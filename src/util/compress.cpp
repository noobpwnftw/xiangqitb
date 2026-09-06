#include "compress.h"

#include "util/defines.h"
#include "util/fixed_vector.h"
#include "util/allocation.h"
#include "util/progress_bar.h"
#include "util/utility.h"

#include <algorithm>
#include <thread>
#include <memory>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>

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

void Compressed_Block_Store::open(
	size_t num_blocks,
	size_t uncompressed_bytes,
	size_t in_memory_limit,
	std::filesystem::path spill_path
)
{
	close();

	if (uncompressed_bytes <= in_memory_limit)
	{
		m_blocks.resize(num_blocks);
		return;
	}

	m_offsets.assign(num_blocks, 0);
	m_sizes.assign(num_blocks, 0);
	m_path = std::move(spill_path);
}

void Compressed_Block_Store::close()
{
	m_blocks.clear();
	m_blocks.shrink_to_fit();
	m_offsets.clear();
	m_offsets.shrink_to_fit();
	m_sizes.clear();
	m_sizes.shrink_to_fit();
	m_total_size = 0;
	m_map.close_file();
	m_out.close_file();
	m_tmp_files.clear();
	m_path.clear();
	m_writes_in_flight.store(0, std::memory_order_relaxed);
	m_finalized.store(false, std::memory_order_relaxed);
}

void Compressed_Block_Store::swap(Compressed_Block_Store& other) noexcept
{
	using std::swap;
	swap(m_blocks, other.m_blocks);
	swap(m_offsets, other.m_offsets);
	swap(m_sizes, other.m_sizes);
	swap(m_total_size, other.m_total_size);
	swap(m_tmp_files, other.m_tmp_files);
	swap(m_path, other.m_path);
	swap(m_out, other.m_out);
	swap(m_map, other.m_map);

	const size_t in_flight = m_writes_in_flight.load(std::memory_order_relaxed);
	m_writes_in_flight.store(other.m_writes_in_flight.load(std::memory_order_relaxed), std::memory_order_relaxed);
	other.m_writes_in_flight.store(in_flight, std::memory_order_relaxed);

	const bool finalized = m_finalized.load(std::memory_order_relaxed);
	m_finalized.store(other.m_finalized.load(std::memory_order_relaxed), std::memory_order_relaxed);
	other.m_finalized.store(finalized, std::memory_order_relaxed);
}

size_t Compressed_Block_Store::total_bytes() const
{
	if (is_spilled())
		return m_total_size;

	size_t total = 0;
	for (const auto& b : m_blocks)
		total += b.size();
	return total;
}

void Compressed_Block_Store::store(size_t block_id, Const_Span<uint8_t> data)
{
	if (data.size() == 0)
	{
		clear(block_id);
		return;
	}

	if (!is_spilled())
	{
		// Distinct block ids only, so the slot itself needs no lock.
		m_blocks[block_id].assign(data.begin(), data.end());
		return;
	}

	uint64_t offset;
	{
		std::lock_guard<std::mutex> lock(m_mutex);

		if (m_finalized.load(std::memory_order_relaxed))
			print_and_abort("Write to a compressed block spill file already being read: %s\n", m_path.string().c_str());

		if (!m_out.is_open())
		{
			std::error_code ec;
			std::filesystem::create_directories(m_path.parent_path(), ec);
			if (!m_out.create(m_path))
				print_and_abort("Could not open compressed block spill file: %s\n", m_path.string().c_str());
			m_tmp_files.track_path(m_path);
		}

		offset = m_total_size;
		m_total_size += data.size();
		m_offsets[block_id] = offset;
		m_sizes[block_id] = data.size();
		m_writes_in_flight.fetch_add(1, std::memory_order_relaxed);
	}

	// Reserved ranges are disjoint, so the writes themselves need no lock.
	const bool ok = m_out.write_at(offset, data);
	m_writes_in_flight.fetch_sub(1, std::memory_order_release);

	if (!ok)
		print_and_abort("Write error on compressed block spill file: %s\n", m_path.string().c_str());
}

void Compressed_Block_Store::clear(size_t block_id)
{
	if (!is_spilled())
	{
		m_blocks[block_id].clear();
		return;
	}

	std::lock_guard<std::mutex> lock(m_mutex);
	m_offsets[block_id] = 0;
	m_sizes[block_id] = 0;
}

Const_Span<uint8_t> Compressed_Block_Store::block(size_t block_id) const
{
	if (!is_spilled())
	{
		const auto& b = m_blocks[block_id];
		return b.empty() ? Const_Span<uint8_t>() : Const_Span<uint8_t>(b.data(), b.size());
	}

	const size_t size = m_sizes[block_id];
	if (size == 0)
		return {};

	if (!m_finalized.load(std::memory_order_acquire))
	{
		std::lock_guard<std::mutex> lock(m_mutex);

		if (!m_finalized.load(std::memory_order_relaxed))
		{
			if (m_out.is_open())
			{
				while (m_writes_in_flight.load(std::memory_order_acquire) != 0)
					std::this_thread::yield();
				if (!m_out.flush())
					print_and_abort("Write error on compressed block spill file: %s\n", m_path.string().c_str());
				m_out.close_file();
			}

			if (m_map.data() == nullptr && !m_map.open_readonly(m_path))
				print_and_abort("Could not mmap compressed block spill file: %s\n", m_path.string().c_str());

			m_finalized.store(true, std::memory_order_release);
		}
	}

	return Const_Span<uint8_t>(m_map.data() + m_offsets[block_id], size);
}

void compress_blocks_into(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<uint8_t> src,
	size_t block_size,
	const Compress_Helper& compressor_factory,
	size_t first_block_id,
	size_t max_workers,
	In_Out_Param<Block_Sink> sink,
	In_Out_Param<Concurrent_Progress_Bar> progress_bar
)
{
	const size_t num_blocks = ceil_div(src.size(), block_size);
	if (num_blocks == 0)
		return;

	const size_t capped = max_workers == 0
		? thread_pool->num_workers()
		: std::min(thread_pool->num_workers(), max_workers);
	const size_t workers = std::max<size_t>(1, std::min(capped, num_blocks));

	std::atomic<size_t> next_block_id(0);

	thread_pool->run_sync_task_on_multiple_threads(workers, [&](size_t thread_id) {
		std::unique_ptr<Compress_Helper> c_helper = compressor_factory.clone();

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

			sink->store(first_block_id + block_id,
				Const_Span<uint8_t>(compressed_block_buffer.get(), out_sz));

			*progress_bar += 1;
		}
	});
}
