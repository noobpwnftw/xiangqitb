#pragma once

#include "egtb_paged.h"

#include "util/defines.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

struct Piece_Config;

// Memory plan for generating one metric (WDL/DTC or DTM) of one material.
//
// The budget is charged for resident memory only: the work/frontier bitsets,
// the packed WDL projection, resident distance pages, page bookkeeping and
// compression buffers. Sub-table readers are memory mapped from the scratch
// files, so the OS can reclaim them under pressure and they are not charged.
enum struct Generation_Mode
{
	// Both distance tables fully resident: the original unbounded path.
	FLAT,
	// Distance tables cut into pages that spill to tmpdir.
	PAGED,
	// The budget is below the paged floor.
	REJECTED
};

struct Generation_Memory
{
	Generation_Mode mode = Generation_Mode::FLAT;

	// The index space does not fit the index type; nothing else is filled in.
	bool too_large = false;

	size_t num_positions = 0;
	size_t entry_bytes = 0;
	size_t num_tables = 2;             // one per side to move

	// Whether the two-ply/three-ply check/chase adjudication passes run. They
	// bind the floor when they do.
	bool has_check_chase = false;

	// Five bitsets are held while generating; three of them are released
	// before the output files are written, leaving the two unknown bitsets.
	size_t bitset_bytes = 0;
	size_t save_bitset_bytes = 0;

	// The packed WDL projection, built while writing the output files.
	size_t wdl_buffer_bytes = 0;

	// Compression, sized for `compression_workers` of them.
	size_t compression_workers = 1;
	size_t compression_buffer_bytes = 0;

	// Buffer the final logical-order sweep uses for one band. Paged only.
	size_t sweep_band_bytes = 0;

	// Uncompressed bytes below which compressed blocks are held in memory
	// rather than spilled. Compressed output never exceeds about 1.1x the
	// input, so comparing against the uncompressed size is conservative.
	size_t block_store_limit_bytes = 0;

	// Unbounded fast path: both tables fully resident.
	size_t flat_table_bytes = 0;

	// Paged path geometry.
	bool pageable = false;
	size_t num_slices = 0;
	size_t slice_bytes = 0;
	size_t slices_per_page = 0;
	size_t page_bytes = 0;
	size_t num_pages = 0;
	size_t page_metadata_bytes = 0;

	// Widest pin set any dispatch needs, over all pass shapes, plus the
	// check/chase look-ahead's transient lease. Counted across both tables.
	size_t floor_pages = 0;
	size_t floor_page_bytes = 0;

	// Pages the cache may keep resident. At least floor_pages; more when the
	// budget allows, which is pure cache benefit.
	size_t capacity_pages = 0;

	NODISCARD size_t distance_bytes() const
	{
		return mode == Generation_Mode::PAGED ? floor_page_bytes : flat_table_bytes;
	}

	// The three peaks a run passes through. Not additive: the bitset pool
	// shrinks before the output files are written, and the pager is flushed
	// before the final sweep.
	NODISCARD size_t generation_peak_bytes() const
	{
		return bitset_bytes + distance_bytes() + page_metadata_bytes;
	}

	NODISCARD size_t finalize_peak_bytes() const
	{
		return save_bitset_bytes + wdl_buffer_bytes + distance_bytes() + page_metadata_bytes;
	}

	NODISCARD size_t compress_peak_bytes() const
	{
		// Paged runs flush every page first, so the pages' budget is free for
		// the band buffer; flat runs still hold the whole table.
		const size_t resident_distance =
			mode == Generation_Mode::PAGED ? sweep_band_bytes : flat_table_bytes;
		return save_bitset_bytes + wdl_buffer_bytes + compression_buffer_bytes
			+ resident_distance + page_metadata_bytes;
	}

	NODISCARD size_t peak_bytes() const
	{
		return std::max(generation_peak_bytes(),
			std::max(finalize_peak_bytes(), compress_peak_bytes()));
	}

	// What the original unbounded path needs, at full thread count.
	NODISCARD size_t flat_total() const
	{
		return std::max(
			bitset_bytes + flat_table_bytes,
			save_bitset_bytes + wdl_buffer_bytes + flat_table_bytes + compression_buffer_bytes);
	}

	// The minimum a paged run needs. A material is rejected only when the
	// budget is below this, not merely because the flat total does not fit.
	NODISCARD size_t paged_floor_total() const
	{
		return pageable ? peak_bytes() : 0;
	}
};

// They differ in entry size and in whether a packed WDL projection is built
// alongside.
enum struct Generation_Metric
{
	WDL_DTC,
	DTM
};

// Plans the page geometry for `ps` under `budget_bytes`, choosing the largest
// page bundle whose floor still fits so that spill I/O stays coarse.
NODISCARD Generation_Memory plan_generation_memory(
	const Piece_Config& ps,
	Generation_Metric metric,
	size_t num_threads,
	size_t budget_bytes
);
