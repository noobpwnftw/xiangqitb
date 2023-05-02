#pragma once

#include "egtb.h"

#include "util/defines.h"
#include "util/param.h"
#include "util/fixed_vector.h"
#include "util/span.h"
#include "util/thread_pool.h"
#include "util/compress.h"

#include <string>
#include <vector>
#include <optional>

constexpr uint8_t EGTB_SINGULAR_FLAG = 0x80;
constexpr uint64_t EGTB_CHECKSUM_INIT_VALUE = 0xf0f0f0f0f0f0;

constexpr size_t WDL_BLOCK_SIZE = 64 * 1024;

struct Compressed_EGTB
{
	static Compressed_EGTB make_singular(WDL_Entry sv)
	{
		Compressed_EGTB info{};
		info.set_singular(sv);
		return info;
	}

	Compressed_EGTB(
		std::vector<std::vector<uint8_t>>&& compressed_blocks,
		size_t src_blk_sz,
		size_t tail_blk_sz,
		std::optional<LZ4_Dict> d,
		bool is_big
	);

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

	NODISCARD const auto& compressed_blocks() const
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
		return m_compressed_blocks.size();
	}

private:
	bool m_is_singular;
	bool m_is_big_order;

	WDL_Entry m_single_val;

	size_t m_block_size;
	size_t m_tail_size;

	std::vector<std::vector<uint8_t>> m_compressed_blocks;
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

NODISCARD std::optional<LZ4_Dict> make_dict_for_evtb(
	Const_Span<Packed_WDL_Entries> data
);

NODISCARD Compressed_EGTB save_compress_evtb(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<Packed_WDL_Entries> src,
	Color color,
	const EGTB_Info& info
);

NODISCARD Compressed_EGTB save_compress_egtb(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<uint8_t> src,
	Color color,
	const EGTB_Info& info,
	bool is_big
);

void save_evtb_table(
	const Piece_Config& ps,
	const Compressed_EGTB save_info[COLOR_NB],
	std::filesystem::path file_path,
	const Fixed_Vector<Color, 2> table_colors,
	EGTB_Magic magic
);

void save_egtb_table(
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