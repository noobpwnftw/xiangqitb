#include "egtb_compress.h"

#include "egtb_gen.h"

#include "util/allocation.h"
#include "util/progress_bar.h"
#include "util/filesystem.h"
#include "util/memory.h"

static void prepare_wdl_entries_for_compression(Span<WDL_Entry> data)
{
	const size_t size = data.size();
	for (size_t begin = 0, end = 0; begin < size; begin = end)
	{
		while (begin < size && data[begin] != WDL_Entry::ILLEGAL)
			++begin;

		if (begin == size)
			break;

		ASSERT(data[begin] == WDL_Entry::ILLEGAL);

		end = begin + 1;
		while (end < size && data[end] == WDL_Entry::ILLEGAL)
			++end;

		ASSERT(data[end - 1] == WDL_Entry::ILLEGAL);
		ASSERT(end == size || data[end] != WDL_Entry::ILLEGAL);

		/*                       begin           end
								 V               V
			0120120121012110210213333333333333333032031032123
		*/

		WDL_Entry fill_value = WDL_Entry::ILLEGAL;
		if (begin > 1 && data[begin - 2] == data[begin - 1])
			fill_value = data[begin - 1];
		else if (end < size - 1 && (data[end] == data[end + 1] || data[end + 1] == WDL_Entry::ILLEGAL))
			fill_value = data[end];
		else if (begin > 0)
			fill_value = data[begin - 1];
		else if (end < size)
			fill_value = data[end];

		if (fill_value != WDL_Entry::ILLEGAL)
			std::fill(data.begin() + begin, data.begin() + end, fill_value);
	}
}

static void prepare_packed_wdl_entries_for_compression(Span<Packed_WDL_Entries> data)
{
	if (data.size() == 0)
		return;

	auto dst_buf = cpp20::make_unique_for_overwrite<WDL_Entry[]>(data.size() * WDL_ENTRY_PACK_RATIO);
	const Span unpacked_span(dst_buf.get(), data.size() * WDL_ENTRY_PACK_RATIO);

	unpack_wdl_entries(data, unpacked_span);

	prepare_wdl_entries_for_compression(unpacked_span);

	pack_wdl_entries(unpacked_span, data);
}

void prepare_evtb_for_compression(In_Out_Param<Thread_Pool> thread_pool, Span<Packed_WDL_Entries> data)
{
	std::atomic<size_t> next_block_id(0);
	thread_pool->run_sync_task_on_all_threads([&](size_t thread_id) {
		for (;;)
		{
			const size_t block_id = next_block_id.fetch_add(1);

			const auto block = data.nth_chunk(block_id, WDL_BLOCK_SIZE);
			if (block.empty())
				return;

			prepare_packed_wdl_entries_for_compression(block);
		}
	});
}

std::optional<LZ4_Dict> make_dict_for_evtb(
	Const_Span<Packed_WDL_Entries> data
)
{
	constexpr size_t DICT_MAX_SIZE = 1024 * 32;
	constexpr size_t MAX_TOTAL_SAMPLES_SIZE = DICT_MAX_SIZE * 1024;
	constexpr size_t SAMPLE_BLOCK_SIZE = 4096;
	constexpr size_t MIN_BLOCKS_TO_MAKE_DICT = 256;

	const size_t block_cnt = data.size() / WDL_BLOCK_SIZE;

	if (block_cnt >= MIN_BLOCKS_TO_MAKE_DICT)
	{
		const size_t num_blocks_to_use = std::min(MAX_TOTAL_SAMPLES_SIZE / WDL_BLOCK_SIZE, block_cnt);
		const size_t split = std::max(block_cnt / num_blocks_to_use, (size_t)1);
		const size_t buf_size = num_blocks_to_use * WDL_BLOCK_SIZE;

		auto dist_buf = cpp20::make_unique_for_overwrite<Packed_WDL_Entries[]>(buf_size);

		for (size_t i = 0; i < num_blocks_to_use; ++i)
			std::memcpy(
				dist_buf.get() + i * WDL_BLOCK_SIZE, 
				data.data() + i * split * WDL_BLOCK_SIZE, 
				WDL_BLOCK_SIZE
			);

		return LZ4_Dict::make(
			Const_Span(reinterpret_cast<const uint8_t*>(dist_buf.get()), buf_size), 
			DICT_MAX_SIZE, 
			SAMPLE_BLOCK_SIZE
		);
	}

	return std::nullopt;
}

Compressed_EGTB save_compress_evtb(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<Packed_WDL_Entries> lpSrcData,
	Color color,
	const EGTB_Info& info
)
{
	const std::string task_name = std::string("save_compress_evtb ") + std::to_string(static_cast<int>(color));

	if (info.draw_cnt[color] + info.lose_cnt[color] == 0)
	{
		printf("%s: singular\n", task_name.c_str());
		return Compressed_EGTB::make_singular(WDL_Entry::WIN);
	}

	if (info.win_cnt[color] + info.lose_cnt[color] == 0)
	{
		printf("%s: singular\n", task_name.c_str());
		return Compressed_EGTB::make_singular(WDL_Entry::DRAW);
	}

	if (info.win_cnt[color] + info.draw_cnt[color] == 0)
	{
		printf("%s: singular\n", task_name.c_str());
		return Compressed_EGTB::make_singular(WDL_Entry::LOSE);
	}

	auto dict = make_dict_for_evtb(lpSrcData);

	auto raw_data = Const_Span(reinterpret_cast<const uint8_t*>(lpSrcData.data()), lpSrcData.size());

	auto compressed_blocks = compress_blocks(
		thread_pool,
		raw_data,
		WDL_BLOCK_SIZE,
		std::make_unique<LZ4_Compress_Helper>(dict.has_value() ? &*dict : nullptr),
		task_name
	);

	return Compressed_EGTB(
		std::move(compressed_blocks),
		WDL_BLOCK_SIZE,
		raw_data.size() % WDL_BLOCK_SIZE,
		std::move(dict),
		false
	);
}

Compressed_EGTB save_compress_egtb(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<uint8_t> lpSrcData,
	Color color,
	const EGTB_Info& info,
	bool is_big
)
{
	static constexpr size_t BLOCK_SIZE = 1024 * 1024;

	const std::string task_name = std::string("save_compress_egtb ") + std::to_string(static_cast<int>(color));

	if (info.win_cnt[color] + info.lose_cnt[color] == 0)
	{
		printf("%s: singular\n", task_name.c_str());
		return Compressed_EGTB::make_singular(WDL_Entry::DRAW);
	}

	auto compressed_blocks = compress_blocks(
		thread_pool,
		lpSrcData,
		BLOCK_SIZE,
		std::make_unique<LZMA_Compress_Helper>(),
		task_name
	);

	return Compressed_EGTB(
		std::move(compressed_blocks),
		BLOCK_SIZE,
		lpSrcData.size() % BLOCK_SIZE,
		std::nullopt,
		is_big
	);
}

void save_evtb_table(
	const Piece_Config& ps,
	const Compressed_EGTB save_info[COLOR_NB],
	std::filesystem::path file_path,
	const Fixed_Vector<Color, 2> table_colors,
	EGTB_Magic magic
)
{
	size_t file_size = 8; // 文件头8字节
	size_t offset_bits[COLOR_NB] = { 4, 4 };

	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			file_size += 2;
		else
		{
			offset_bits[i] = t.total_compressed_size() <= 0xffffffff ? 4 : 6;
			file_size += 20;
		}
	}

	// 字典大小
	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		file_size += 2;
		if (t.dict().has_value())
		{
			file_size += t.dict()->size();
			if (file_size & 1)
				file_size += 1;
		}
	}
	// 偏移量写入
	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		file_size += (offset_bits[i] + 2) * t.num_blocks();
	}

	file_size = ceil_to_multiple(file_size, (size_t)64);

	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		file_size += t.total_compressed_size();
		file_size = ceil_to_multiple(file_size, (size_t)64);
	}

	// file_size is found.

	Memory_Mapped_File write_map;
	if (!write_map.create(file_path.c_str(), file_size + 8))
		abort();

	Serial_Memory_Writer writer(write_map.data_span());

	writer.write<uint32_t>(narrowing_static_cast<uint32_t>(magic));
	writer.write<uint32_t>(narrowing_static_cast<uint32_t>((ps.min_material_key().value() << 2ull) + table_colors.size()));

	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
		{
			writer.write<uint8_t>(narrowing_static_cast<uint8_t>(EGTB_SINGULAR_FLAG));
			writer.write<uint8_t>(narrowing_static_cast<uint8_t>(t.single_val()));
		}
		else
		{
			writer.write<uint8_t>(0);
			writer.write<uint8_t>(narrowing_static_cast<uint8_t>(offset_bits[i]));

			writer.write<uint16_t>(narrowing_static_cast<uint16_t>(t.tail_size()));
			writer.write<uint32_t>(narrowing_static_cast<uint32_t>(t.block_size()));
			writer.write<uint32_t>(narrowing_static_cast<uint32_t>(t.num_blocks()));
			writer.write<uint64_t>(narrowing_static_cast<uint64_t>(t.total_compressed_size()));
		}
	}

	// 字典写入
	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		if (t.dict().has_value())
		{
			writer.write<uint16_t>(narrowing_static_cast<uint16_t>(t.dict()->size()));
			writer.write(Const_Span(t.dict()->data(), t.dict()->size()));
			writer.zero_align(2);
		}
		else
			writer.write<uint16_t>(0);
	}

	// 偏移量写入
	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		size_t offset = 0;
		for (const auto& block : t.compressed_blocks())
		{
			writer.write<uint16_t>(narrowing_static_cast<uint16_t>(block.size()));
			writer.write<uint32_t>(narrowing_static_cast<uint32_t>(offset));

			if (offset_bits[i] == 6)
				writer.write<uint16_t>(narrowing_static_cast<uint16_t>(offset >> 32));

			offset += block.size();
		}
	}

	writer.zero_align(64);

	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		for (const auto& block : t.compressed_blocks())
			writer.write(Const_Span(block));

		writer.zero_align(64);
	}

	if (writer.num_bytes_written() != file_size)
		print_and_abort("file size is wrong.\n");

	writer.write_end_checksum(static_cast<uint64_t>(EGTB_CHECKSUM_INIT_VALUE));

	write_map.close();
}

void save_egtb_table(
	const Piece_Config& ps,
	const Compressed_EGTB save_info[COLOR_NB],
	std::filesystem::path file_path,
	const Fixed_Vector<Color, 2> table_colors,
	EGTB_Magic magic
)
{
	size_t file_size = 8; // 文件头8字节

	for (const Color i : table_colors)
		file_size += save_info[i].is_singular() ? 2 : 22;

	// 偏移量写入
	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		file_size += t.num_blocks() * 8;
	}

	file_size = ceil_to_multiple(file_size, (size_t)64);

	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		file_size += t.total_compressed_size();
		file_size = ceil_to_multiple(file_size, (size_t)64);
	}

	// file_size is found.

	Memory_Mapped_File write_map;
	write_map.create(file_path.c_str(), file_size + 8);

	Serial_Memory_Writer writer(write_map.data_span());

	writer.write<uint32_t>(narrowing_static_cast<uint32_t>(magic));
	writer.write<uint32_t>(narrowing_static_cast<uint32_t>((ps.min_material_key().value() << 2ull) + table_colors.size()));

	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
		{
			writer.write<uint8_t>(narrowing_static_cast<uint8_t>(EGTB_SINGULAR_FLAG));
			writer.write<uint8_t>(narrowing_static_cast<uint8_t>(t.single_val()));
		}
		else
		{
			writer.write<uint8_t>(0);
			writer.write<uint8_t>(t.is_big_order());

			writer.write<uint32_t>(narrowing_static_cast<uint32_t>(t.tail_size()));
			writer.write<uint32_t>(narrowing_static_cast<uint32_t>(t.block_size()));
			writer.write<uint32_t>(narrowing_static_cast<uint32_t>(t.num_blocks()));
			writer.write<uint64_t>(narrowing_static_cast<uint64_t>(t.total_compressed_size()));
		}
	}

	// 偏移量写入
	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		size_t offset = 0;
		for (const auto& block : t.compressed_blocks())
		{
			ASSERT(block.size() < (1 << 20));
			writer.write<uint64_t>((offset << 20) + block.size());
			offset += block.size();
		}
	}

	writer.zero_align(64);

	for (const Color i : table_colors)
	{
		const Compressed_EGTB& t = save_info[i];
		if (t.is_singular())
			continue;

		for (const auto& block : t.compressed_blocks())
			writer.write(Const_Span(block));
		writer.zero_align(64);
	}

	if (writer.num_bytes_written() != file_size)
		print_and_abort("file size is wrong.\n");

	writer.write_end_checksum(static_cast<uint64_t>(EGTB_CHECKSUM_INIT_VALUE));

	write_map.close();
}

Compressed_EGTB::Compressed_EGTB(
	std::vector<std::vector<uint8_t>>&& compressed_blocks,
	size_t src_blk_sz,
	size_t tail_blk_sz,
	std::optional<LZ4_Dict> d,
	bool is_big
) :
	m_is_singular(false),
	m_is_big_order(is_big),
	m_single_val(WDL_Entry::DRAW),
	m_block_size(src_blk_sz),
	m_tail_size(tail_blk_sz),
	m_compressed_blocks(std::move(compressed_blocks)),
	m_total_compressed_size(0),
	m_dict(std::move(d))
{
	for (const auto& block : this->m_compressed_blocks)
		m_total_compressed_size += block.size();
}

void load_evtb_table(
	Out_Param<WDL_File_For_Probe> evtb,
	const Piece_Config& ps,
	std::filesystem::path sub_evtb,
	const std::filesystem::path tmp[COLOR_NB],
	EGTB_Magic evtb_magic
)
{
	Memory_Mapped_File map_file;
	if (!map_file.open_readonly(sub_evtb.c_str()))
		throw std::runtime_error("Could not open WDL file trying to load " + sub_evtb.string());

	const Const_Span<uint8_t> input = map_file.data_span();

	if ((input.size() & 63) != 8)
		throw std::runtime_error("Invalid WDL file size trying to load " + sub_evtb.string());

	Serial_Memory_Reader reader(input);

	if (!reader.is_end_checksum_ok(static_cast<uint64_t>(EGTB_CHECKSUM_INIT_VALUE)))
		throw std::runtime_error("Invalid WDL file checksum trying to load " + sub_evtb.string());

	size_t block_cnt[COLOR_NB]{ 0, 0 };
	size_t block_size[COLOR_NB]{ 0, 0 };
	size_t tail_size[COLOR_NB]{ 0, 0 };

	size_t dict_size[COLOR_NB]{ 0, 0 };
	const uint8_t* lp_dict[COLOR_NB]{ nullptr, nullptr };
	size_t offset_bits[COLOR_NB]{ 0, 0 };
	const uint8_t* data[COLOR_NB]{ nullptr, nullptr };
	size_t data_size[COLOR_NB]{ 0, 0 };
	const uint8_t* offset_tb[COLOR_NB]{ nullptr, nullptr };

	const uint32_t magic = reader.read<uint32_t>();

	if (magic != narrowing_static_cast<uint32_t>(evtb_magic))
		throw std::runtime_error("Invalid WDL file magic trying to load " + sub_evtb.string());

	const uint32_t key_and_table_num = reader.read<uint32_t>();
	const Material_Key key = static_cast<Material_Key>(key_and_table_num >> 2u);
	if (key != ps.min_material_key())
		throw std::runtime_error("Wrong material key in WDL file " + sub_evtb.string());

	const size_t table_num = key_and_table_num & 3;
	const Fixed_Vector<Color, 2> table_colors = egtb_table_colors(table_num);

	for (const Color i : table_colors)
	{
		if (reader.read<uint8_t>() & 0x80)
		{
			evtb->m_is_singular[i] = true;
			evtb->m_single_val[i] = static_cast<WDL_Entry>(reader.read<uint8_t>());
		}
		else
		{
			evtb->m_is_singular[i] = false;

			offset_bits[i] = reader.read<uint8_t>();
			tail_size[i] = reader.read<uint16_t>();
			block_size[i] = reader.read<uint32_t>();
			block_cnt[i] = reader.read<uint32_t>();
			data_size[i] = reader.read<uint64_t>();
		}
	}

	for (const Color i : table_colors)
	{
		if (evtb->m_is_singular[i])
			continue;

		dict_size[i] = reader.read<uint16_t>();
		if (dict_size[i] != 0)
		{
			lp_dict[i] = reader.caret();
			reader.advance(dict_size[i]);
			reader.align(2);
		}
	}

	for (const Color i : table_colors)
	{
		if (evtb->m_is_singular[i])
			continue;

		offset_tb[i] = reader.caret();
		reader.advance((2 + offset_bits[i]) * block_cnt[i]);
	}

	for (const Color i : table_colors)
	{
		if (evtb->m_is_singular[i])
			continue;

		reader.align(64);
		data[i] = reader.caret();
		reader.advance(data_size[i]);
	}

	for (const Color i : table_colors)
	{
		if (evtb->m_is_singular[i])
			continue;

		ASSERT(data[i] != nullptr);
		ASSERT(offset_tb[i] != nullptr);

		Memory_Mapped_File out_map(Memory_Mapped_File::Access_Advice::RANDOM);
		const size_t num_full_sized_blocks =
			tail_size[i] != 0
			? block_cnt[i] - 1
			: block_cnt[i];
		const size_t file_sz = block_size[i] * num_full_sized_blocks + tail_size[i];
		if (file_sz != evtb->uncompressed_file_size(Piece_Config_For_Gen(ps).num_positions()))
			throw std::runtime_error("Invalid decompressed size of WDL table from " + sub_evtb.string());

		out_map.create(tmp[i].c_str(), file_sz);

		Serial_Memory_Writer writer(out_map.data_span());

		// We prepend the dictionary to the output data, because
		// LZ4_decompress_safe_usingDict uses a fast-path when the
		// dictionary is immediately before the output.
		const auto dict = LZ4_Dict::load(Const_Span(lp_dict[i], lp_dict[i] + dict_size[i]));
		LZ4_Decompress_Helper dc_helper(dict, block_size[i]);

		for (size_t idx = 0; idx < block_cnt[i]; ++idx)
		{
			Serial_Memory_Reader block_reader(Const_Span(offset_tb[i] + (offset_bits[i] + 2) * idx, 2 + 4 + 2));
			const size_t size = block_reader.read<uint16_t>();
			size_t data_offset = block_reader.read<uint32_t>();
			if (offset_bits[i] == 6)
			{
				const size_t hi = block_reader.read<uint16_t>();
				data_offset += hi << 32;
			}

			const uint8_t* src_data = data[i] + data_offset;

			const size_t decode_size =
				idx == block_cnt[i] - 1 && tail_size[i]
				? tail_size[i]
				: block_size[i];

			const Const_Span<uint8_t> decompressed = dc_helper.decompress(Const_Span(src_data, size), decode_size);
			writer.write(decompressed);
		}

		evtb->m_files[i] = std::move(out_map);
	}
}

void load_egtb_table(
	Out_Param<DTM_File_For_Probe> egtb,
	const Piece_Config& ps,
	std::filesystem::path sub_evtb,
	const std::filesystem::path tmp[COLOR_NB],
	EGTB_Magic egtb_magic
)
{
	Memory_Mapped_File map_file;
	if (!map_file.open_readonly(sub_evtb.c_str()))
		throw std::runtime_error("Could not open DTM file trying to load " + sub_evtb.string());

	const Const_Span<uint8_t> input = map_file.data_span();

	if ((input.size() & 63) != 8)
		throw std::runtime_error("Invalid DTM file size trying to load " + sub_evtb.string());

	Serial_Memory_Reader reader(input);

	if (!reader.is_end_checksum_ok(static_cast<uint64_t>(EGTB_CHECKSUM_INIT_VALUE)))
		throw std::runtime_error("Invalid DTM file checksum trying to load " + sub_evtb.string());

	size_t block_cnt[COLOR_NB]{ 0, 0 };
	size_t block_size[COLOR_NB]{ 0, 0 };
	size_t tail_size[COLOR_NB]{ 0, 0 };

	const uint8_t* data[COLOR_NB]{ nullptr, nullptr };
	size_t data_size[COLOR_NB]{ 0, 0 };
	const uint8_t* offset_tb[COLOR_NB]{ nullptr, nullptr };

	const uint32_t magic = reader.read<uint32_t>();
	if (magic != narrowing_static_cast<uint32_t>(egtb_magic))
		throw std::runtime_error("Invalid DTM file magic trying to load " + sub_evtb.string());

	const uint32_t key_and_table_num = reader.read<uint32_t>();
	const Material_Key key = Material_Key(key_and_table_num >> 2);
	if (key != ps.min_material_key())
		throw std::runtime_error("Wrong material key in DTM file " + sub_evtb.string());

	const size_t table_num = key_and_table_num & 3;
	const Fixed_Vector<Color, 2> table_colors = egtb_table_colors(table_num);

	for (const Color i : table_colors)
	{
		if (reader.read<uint8_t>() & narrowing_static_cast<uint8_t>(EGTB_SINGULAR_FLAG))
		{
			egtb->m_is_singular_draw[i] = true;
			const WDL_Entry single_val = static_cast<WDL_Entry>(reader.read<uint8_t>());
			if (single_val != WDL_Entry::DRAW)
				throw std::runtime_error("Invalid single_val (not draw) in DTM table.");
		}
		else
		{
			egtb->m_is_singular_draw[i] = false;

			reader.advance(1);
			tail_size[i] = reader.read<uint32_t>();
			block_size[i] = reader.read<uint32_t>();
			block_cnt[i] = reader.read<uint32_t>();
			data_size[i] = reader.read<uint64_t>();
		}
	}

	for (const Color i : table_colors)
	{
		if (egtb->m_is_singular_draw[i])
			continue;

		offset_tb[i] = reader.caret();
		reader.advance(block_cnt[i] * 8);
	}

	for (const Color i : table_colors)
	{
		if (egtb->m_is_singular_draw[i])
			continue;

		reader.align(64);
		data[i] = reader.caret();
		reader.advance(data_size[i]);
	}

	for (const Color i : table_colors)
	{
		if (egtb->m_is_singular_draw[i])
			continue;

		ASSERT(data[i] != nullptr);
		ASSERT(offset_tb[i] != nullptr);

		Memory_Mapped_File out_map(Memory_Mapped_File::Access_Advice::RANDOM);
		const size_t num_full_sized_blocks =
			tail_size[i] != 0
			? block_cnt[i] - 1
			: block_cnt[i];
		const size_t file_sz = block_size[i] * num_full_sized_blocks + tail_size[i];
		if (file_sz != egtb->uncompressed_file_size(Piece_Config_For_Gen(ps).num_positions()))
			throw std::runtime_error("Invalid decompressed size of DTM table from " + sub_evtb.string());

		out_map.create(tmp[i].c_str(), file_sz);

		Serial_Memory_Writer writer(out_map.data_span());

		LZMA_Decompress_Helper dc_helper(block_size[i] * DTM_File_For_Probe::ENTRY_SIZE);

		for (size_t idx = 0; idx < block_cnt[i]; ++idx)
		{
			const uint8_t* offset = offset_tb[i] + idx * 8;

			const size_t data_size_and_offset = reinterpret_cast<const uint64_t*>(offset)[0];
			const size_t data_size = data_size_and_offset & 0xFFFFF;
			const size_t data_offset = data_size_and_offset >> 20;

			const uint8_t* p_src = data[i] + data_offset;

			const size_t decode_size =
				idx == block_cnt[i] - 1 && tail_size[i]
				? tail_size[i]
				: block_size[i];

			const Const_Span<uint8_t> decompressed = dc_helper.decompress(Const_Span(p_src, data_size), decode_size);
			writer.write(decompressed);
		}

		egtb->m_files[i] = std::move(out_map);
	}
}
