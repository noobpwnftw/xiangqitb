#pragma once

#include "egtb.h"
#include "egtb_paged.h"

#include "util/defines.h"
#include "util/param.h"
#include "util/fixed_vector.h"
#include "util/span.h"
#include "util/thread_pool.h"
#include "util/compress.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>

inline constexpr uint8_t EGTB_SINGULAR_FLAG = 0x80;
inline constexpr uint64_t EGTB_CHECKSUM_INIT_VALUE = 0xf0f0f0f0f0f0;

inline constexpr size_t WDL_BLOCK_SIZE = 64 * 1024;
inline constexpr size_t EGTB_BLOCK_SIZE = 1024 * 1024;

// Compressed blocks stay in memory while the uncompressed table is at most
// this large, and spill to tmpdir above it.
inline constexpr size_t DEFAULT_BLOCK_STORE_MEMORY_LIMIT = 512ull * 1024 * 1024;

// How often the block progress bar prints, in blocks.
NODISCARD inline size_t block_progress_period(size_t block_size, size_t num_workers)
{
	constexpr size_t PRINT_PERIOD_BYTES = 1024 * 1024 * 8;
	return ceil_div(PRINT_PERIOD_BYTES * num_workers, block_size);
}

struct Compressed_EGTB
{
	static Compressed_EGTB make_singular(WDL_Entry sv)
	{
		Compressed_EGTB info{};
		info.set_singular(sv);
		return info;
	}

	Compressed_EGTB(
		Compressed_Block_Store&& compressed_blocks,
		size_t src_blk_sz,
		size_t tail_blk_sz,
		std::optional<LZ4_Dict> d,
		bool is_big
	);

	Compressed_EGTB(const Compressed_EGTB&) = delete;
	Compressed_EGTB& operator=(const Compressed_EGTB&) = delete;
	Compressed_EGTB(Compressed_EGTB&&) noexcept = default;
	Compressed_EGTB& operator=(Compressed_EGTB&&) noexcept = default;

	Compressed_EGTB() :
		m_is_singular(false),
		m_is_big_order(false),
		m_single_val(WDL_Entry::DRAW),
		m_block_size(0),
		m_tail_size(0),
		m_total_compressed_size(0)
	{
	}

	NODISCARD bool is_singular() const
	{
		return m_is_singular;
	}

	NODISCARD WDL_Entry single_val() const
	{
		ASSERT(m_is_singular);
		return m_single_val;
	}

	NODISCARD size_t block_size() const
	{
		return m_block_size;
	}

	NODISCARD size_t tail_size() const
	{
		return m_tail_size;
	}

	NODISCARD const Compressed_Block_Store& compressed_blocks() const
	{
		return m_compressed_blocks;
	}

	NODISCARD size_t total_compressed_size() const
	{
		return m_total_compressed_size;
	}

	NODISCARD const auto& dict() const
	{
		return m_dict;
	}

	NODISCARD bool is_big_order() const
	{
		return m_is_big_order;
	}

	NODISCARD size_t num_blocks() const
	{
		return m_compressed_blocks.num_blocks();
	}

private:
	bool m_is_singular;
	bool m_is_big_order;

	WDL_Entry m_single_val;

	size_t m_block_size;
	size_t m_tail_size;

	Compressed_Block_Store m_compressed_blocks;
	size_t m_total_compressed_size;

	std::optional<LZ4_Dict> m_dict;

	void set_singular(WDL_Entry val)
	{
		m_is_singular = true;
		m_single_val = val;
	}
};

void prepare_evtb_for_compression(
	In_Out_Param<Thread_Pool> thread_pool,
	Span<Packed_WDL_Entries> data
);

// The same preparation applied to one block. Runs of illegal entries are
// filled with neighbouring values so that they compress.
void prepare_packed_wdl_entries_for_compression(Span<Packed_WDL_Entries> data);

// How make_dict_for_evtb samples the projection: `num_blocks` blocks taken at
// every `stride`-th block, and none when the table is too small for a
// dictionary at all. Exposed because a streamed run has to collect the same
// samples while sweeping, the dictionary being needed before the first block.
struct WDL_Dict_Sampling
{
	size_t num_blocks = 0;
	size_t stride = 1;
};

NODISCARD WDL_Dict_Sampling wdl_dict_sampling(size_t packed_bytes);

NODISCARD std::optional<LZ4_Dict> make_dict_for_evtb(
	Const_Span<Packed_WDL_Entries> data
);

// Trains the dictionary on samples already gathered in that layout.
NODISCARD std::optional<LZ4_Dict> make_dict_from_wdl_samples(
	Const_Span<Packed_WDL_Entries> samples
);

NODISCARD Compressed_EGTB save_compress_evtb(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<Packed_WDL_Entries> src,
	Color color,
	const EGTB_Info& info,
	size_t max_workers,
	size_t in_memory_limit,
	const std::filesystem::path& spill_path
);

NODISCARD Compressed_EGTB save_compress_egtb(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<uint8_t> src,
	Color color,
	const EGTB_Info& info,
	bool is_big,
	size_t max_workers,
	size_t in_memory_limit,
	const std::filesystem::path& spill_path
);

// The single value a table collapses to, when it has one.
NODISCARD std::optional<WDL_Entry> singular_wdl_value(Color color, const EGTB_Info& info);
NODISCARD bool egtb_is_singular_draw(Color color, const EGTB_Info& info);

// Compresses a table that is produced a band at a time in logical Board_Index
// order: `next_band` returns the next stretch of output bytes, and an empty
// span when the table is exhausted. Bytes and metadata are identical to what
// the flat path produces.
NODISCARD Compressed_EGTB save_compress_streamed(
	In_Out_Param<Thread_Pool> thread_pool,
	const Compress_Helper& factory,
	size_t block_size,
	size_t total_bytes,
	// Applied in place to each whole block before it is compressed. Only the
	// WDL projection has such a step; null for the distance tables.
	void (*preprocess)(Span<Packed_WDL_Entries>),
	std::optional<LZ4_Dict> dict,
	bool is_big,
	size_t max_workers,
	size_t in_memory_limit,
	const std::filesystem::path& spill_path,
	const std::string& task_name,
	const std::function<Span<uint8_t>()>& next_band
);

void save_evtb_table(
	In_Out_Param<Thread_Pool> thread_pool,
	const Piece_Config& ps,
	const Compressed_EGTB save_info[COLOR_NB],
	std::filesystem::path file_path,
	const Fixed_Vector<Color, 2> table_colors,
	EGTB_Magic magic
);

void save_egtb_table(
	In_Out_Param<Thread_Pool> thread_pool,
	const Piece_Config& ps,
	const Compressed_EGTB save_info[COLOR_NB],
	std::filesystem::path file_path,
	const Fixed_Vector<Color, 2> table_colors,
	EGTB_Magic magic
);

void load_evtb_table(
	Out_Param<WDL_File_For_Probe> evtb,
	const Piece_Config& ps,
	std::filesystem::path sub_evtb,
	const std::filesystem::path tmp[COLOR_NB],
	EGTB_Magic evtb_magic
);

void load_egtb_table(
	Out_Param<DTM_File_For_Probe> egtb,
	const Piece_Config& ps,
	std::filesystem::path sub_evtb,
	const std::filesystem::path tmp[COLOR_NB],
	EGTB_Magic evtb_magic
);