#include "egtb_memory.h"

#include "egtb_compress.h"
#include "egtb_gen.h"
#include "egtb_gen_dtm.h"
#include "egtb_gen_wdl_dtc.h"

#include <algorithm>

constexpr size_t MiB = 1024 * 1024;

// DTC and DTM both take a pool of five bitsets for the whole run, of which
// the two unknown ones are still held while the output files are written.
constexpr size_t NUM_RESIDENT_BITSETS = 5;
constexpr size_t NUM_BITSETS_HELD_WHILE_SAVING = 2;

// Beyond this a larger sweep band only makes reads coarser, and they are
// already large.
constexpr size_t MAX_SWEEP_BAND_BYTES = 256 * MiB;

// An LZMA encoder at a 1 MiB dictionary costs roughly 11.5x the dictionary.
constexpr size_t LZMA_ENCODER_BYTES = 12 * MiB;
constexpr size_t EGTB_BLOCK_BYTES = MiB;

// The WDL pass also holds the dictionary training samples.
constexpr size_t WDL_DICT_SAMPLE_BYTES = 32 * MiB;

NODISCARD static size_t page_metadata_bytes_for(size_t num_pages, size_t num_slices, size_t num_tables)
{
	// Per page: a Huge_Array handle and a dirty flag. Per slice: the
	// precomputed page id and in-page base offset.
	return num_tables * (num_pages * 32 + num_slices * 2 * sizeof(size_t));
}

// Per compression worker: an encoder, one uncompressed block and one
// bounded output block.
NODISCARD static size_t compression_bytes_per_worker()
{
	return LZMA_ENCODER_BYTES + 2 * EGTB_BLOCK_BYTES;
}

// Workers that fit `spare` bytes, at least one: compression throttles its
// width to the budget rather than the material being rejected over it.
NODISCARD static size_t compression_workers_for(size_t num_threads, size_t spare)
{
	const size_t per_worker = compression_bytes_per_worker();
	return std::max<size_t>(1, std::min(std::max<size_t>(1, num_threads), spare / per_worker));
}

Generation_Memory plan_generation_memory(
	const Piece_Config& ps,
	Generation_Metric metric,
	size_t num_threads,
	size_t budget_bytes
)
{
	Generation_Memory out;
	out.entry_bytes = metric == Generation_Metric::DTM
		? sizeof(DTM_Final_Entry)
		: sizeof(DTC_Final_Entry);

	bool ok = false;
	const Piece_Config_For_Gen epsi(ps, out_param(ok));
	if (!ok)
	{
		out.too_large = true;
		out.mode = Generation_Mode::REJECTED;
		return out;
	}

	const size_t n = epsi.num_positions();
	out.num_positions = n;
	out.has_check_chase = epsi.both_sides_have_free_attackers();

	const size_t bitset = ceil_div(n, size_t(8));
	out.bitset_bytes = NUM_RESIDENT_BITSETS * bitset;
	out.save_bitset_bytes = NUM_BITSETS_HELD_WHILE_SAVING * bitset;

	// The unpaged WDL/DTC run projects WDL into a packed buffer while writing
	// the output files. A paged run derives the projection as it sweeps the
	// distance table instead, so it holds none of it; DTM reads its material's
	// WDL table through a mapping and never had one.
	out.wdl_buffer_bytes = metric == Generation_Metric::WDL_DTC
		? out.num_tables * ceil_div(n, WDL_ENTRY_PACK_RATIO)
		: 0;

	out.compression_workers = std::max<size_t>(1, num_threads);
	out.compression_buffer_bytes =
		out.compression_workers * compression_bytes_per_worker()
		+ (metric == Generation_Metric::WDL_DTC ? WDL_DICT_SAMPLE_BYTES : 0);
	out.block_store_limit_bytes = DEFAULT_BLOCK_STORE_MEMORY_LIMIT;

	out.flat_table_bytes = out.num_tables * n * out.entry_bytes;

	const Slice_Layout layout(epsi);
	out.num_slices = layout.num_slices();
	out.slice_bytes = layout.slice_size() * out.entry_bytes;

	if (out.flat_total() <= budget_bytes)
	{
		out.mode = Generation_Mode::FLAT;
		return out;
	}

	// A single slice cannot be paged against itself, so such a material has to
	// fit flat. This only happens for a mirror-symmetric group of one
	// placement, which implies a tiny table.
	if (out.num_slices < 2)
	{
		out.mode = Generation_Mode::REJECTED;
		return out;
	}

	const Color slice_color = piece_class_color(epsi.compress_id());

	size_t reach_plies = 1;
	for (const Phase_Pages& pass : generation_pass_shapes(out.has_check_chase))
		update_max(reach_plies, required_reach_plies(pass, slice_color));

	const Slice_Reach reach(epsi.group(epsi.compress_id()), reach_plies);

	// Fewer slices per page always means a smaller floor -- the pinned slice
	// set is the same, but less is dragged in with it -- so the largest bundle
	// that fits is the best choice: it keeps spill I/O coarse.
	const size_t preferred = std::max<size_t>(1, std::min(
		Page_Layout::preferred_slices_per_page(out.slice_bytes),
		out.num_slices / 2));

	auto evaluate = [&](size_t slices_per_page) {
		const Page_Layout pages(out.num_slices, slices_per_page);

		Generation_Memory candidate = out;
		candidate.mode = Generation_Mode::PAGED;
		candidate.pageable = true;
		// Derived on the sweep, not materialized; see DTC_Generator::save_egtb.
		candidate.wdl_buffer_bytes = 0;
		candidate.slices_per_page = pages.slices_per_page();
		candidate.num_pages = pages.num_pages();
		candidate.page_bytes = pages.slices_per_page() * out.slice_bytes;
		candidate.page_metadata_bytes =
			page_metadata_bytes_for(pages.num_pages(), out.num_slices, out.num_tables);

		candidate.floor_pages = generation_floor_page_count(
			pages, reach, slice_color, out.has_check_chase);
		candidate.floor_page_bytes = candidate.floor_pages * candidate.page_bytes;

		// Compression and the final sweep run after the pager is flushed, so
		// they get the pages' budget, throttling to what is left rather than
		// pushing the material over the cap.
		const size_t fixed = candidate.save_bitset_bytes + candidate.page_metadata_bytes;
		const size_t spare = budget_bytes > fixed ? budget_bytes - fixed : 0;

		candidate.compression_workers = compression_workers_for(num_threads, spare / 2);
		candidate.compression_buffer_bytes =
			candidate.compression_workers * compression_bytes_per_worker()
			+ (metric == Generation_Metric::WDL_DTC ? WDL_DICT_SAMPLE_BYTES : 0);

		// One band must hold at least one step of low_weight entries.
		const size_t min_band = layout.low_weight() * out.entry_bytes;
		const size_t band_spare = spare > candidate.compression_buffer_bytes
			? spare - candidate.compression_buffer_bytes
			: 0;
		candidate.sweep_band_bytes = std::min(
			std::max(min_band, std::min(band_spare, MAX_SWEEP_BAND_BYTES)),
			out.flat_table_bytes / out.num_tables);

		const size_t store_spare = band_spare > candidate.sweep_band_bytes
			? band_spare - candidate.sweep_band_bytes
			: 0;
		candidate.block_store_limit_bytes =
			std::min(DEFAULT_BLOCK_STORE_MEMORY_LIMIT, store_spare);

		return candidate;
	};

	Generation_Memory chosen;
	for (size_t spp = preferred; ; spp /= 2)
	{
		chosen = evaluate(spp);
		if (chosen.peak_bytes() <= budget_bytes || spp == 1)
			break;
	}

	chosen.mode = chosen.peak_bytes() <= budget_bytes
		? Generation_Mode::PAGED
		: Generation_Mode::REJECTED;

	// Spend whatever budget is left on extra residency; it is pure cache
	// benefit.
	chosen.capacity_pages = chosen.floor_pages;
	const size_t peak = chosen.generation_peak_bytes();
	if (peak <= budget_bytes && chosen.page_bytes != 0)
	{
		const size_t extra = (budget_bytes - peak) / chosen.page_bytes;
		chosen.capacity_pages = std::min(
			chosen.floor_pages + extra,
			chosen.num_pages * chosen.num_tables);
	}

	return chosen;
}
