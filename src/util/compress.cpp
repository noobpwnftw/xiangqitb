#include "compress.h"

#include "util/defines.h"
#include "util/fixed_vector.h"
#include "util/allocation.h"
#include "util/progress_bar.h"

#include <algorithm>
#include <memory>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstdio>

LZ4_Dict::LZ4_Dict(
	Const_Span<uint8_t> data,
	size_t dict_size,
	size_t sample_size
) :
	m_dict(dict_size)
{
	if (data.size() % sample_size != 0)
		throw std::runtime_error("LZ4 dict sample size must divide the data size.");

	const size_t sample_count = data.size() / sample_size;

	if (sample_count == 0)
		throw std::runtime_error("LZ4 dict no samples.");

	const std::vector<size_t> sample_sizes(sample_count, sample_size);

	const size_t new_size = ZDICT_trainFromBuffer(
		m_dict.data(),
		m_dict.size(),
		data.data(),
		sample_sizes.data(),
		narrowing_static_cast<unsigned int>(sample_count)
	);

	if (ZDICT_isError(new_size))
		m_dict.clear();
	else
	{
		ASSUME(new_size <= m_dict.size());
		m_dict.resize(new_size);
	}
}

LZ4_Decompress_Helper::LZ4_Decompress_Helper(const LZ4_Dict& dict, size_t max_output_size) :
	m_output_buffer(cpp20::make_unique_for_overwrite<uint8_t[]>(dict.size() + max_output_size)),
	m_dict_size(dict.size()),
	m_max_output_size(max_output_size)
{
	if (dict.size())
		std::memcpy(m_output_buffer.get(), dict.data(), dict.size());
}

std::vector<uint8_t> LZ4_Compress_Helper::compress(
	Const_Span<uint8_t> src
)
{
	const size_t bound_size = compress_bound(src.size());
	auto compressed_block_buffer = cpp20::make_unique_for_overwrite<uint8_t[]>(bound_size);
	const size_t out_sz = compress(
		Span(compressed_block_buffer.get(), bound_size),
		src
	);
	return std::vector(compressed_block_buffer.get(), compressed_block_buffer.get() + out_sz);
}

std::vector<uint8_t> LZMA_Compress_Helper::compress(Const_Span<uint8_t> src)
{
	const size_t bound_size = compress_bound(src.size());
	auto compressed_block_buffer = cpp20::make_unique_for_overwrite<uint8_t[]>(bound_size);
	const size_t out_sz = compress(
		Span(compressed_block_buffer.get(), bound_size),
		src
	);
	return std::vector(compressed_block_buffer.get(), compressed_block_buffer.get() + out_sz);
}

LZMA_Decompress_Helper::LZMA_Decompress_Helper(size_t max_output_size) :
	m_output_buffer(cpp20::make_unique_for_overwrite<uint8_t[]>(max_output_size)),
	m_max_output_size(max_output_size)
{
}

std::vector<std::vector<uint8_t>> compress_blocks(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<uint8_t> src,
	size_t block_size,
	std::unique_ptr<Compress_Helper> compressor_factory,
	std::string task_name
)
{
	std::vector<std::vector<uint8_t>> compressed_blocks(ceil_div(src.size(), block_size));
	std::atomic<size_t> next_block_id(0);

	constexpr size_t PRINT_PERIOD_BYTES = 1024 * 1024 * 8;
	const size_t PRINT_PERIOD = ceil_div(PRINT_PERIOD_BYTES * thread_pool->num_workers(), block_size);
	Concurrent_Progress_Bar progress_bar(compressed_blocks.size(), PRINT_PERIOD, task_name);

	thread_pool->run_sync_task_on_all_threads([&](size_t thread_id) {
		std::unique_ptr<Compress_Helper> c_helper = compressor_factory->clone();

		const size_t bound_size = c_helper->compress_bound(block_size);

		auto compressed_block_buffer = cpp20::make_unique_for_overwrite<uint8_t[]>(bound_size);

		for (;;)
		{
			const size_t block_id = next_block_id.fetch_add(1);

			const auto block = src.nth_chunk(block_id, block_size);
			if (block.empty())
				return;

			const size_t out_sz = c_helper->compress(
				Span(compressed_block_buffer.get(), bound_size),
				block
			);

			compressed_blocks[block_id] = std::vector(compressed_block_buffer.get(), compressed_block_buffer.get() + out_sz);

			progress_bar += 1;
		}
	});

	progress_bar.set_finished();

	return compressed_blocks;
}
