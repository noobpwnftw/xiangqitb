#include "egtb_paged.h"

#include "util/filesystem.h"
#include "util/utility.h"

#include "lz4/lz4.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>

namespace {

// A spilled page is a sequence of independently compressed chunks, each with
// its own header, so a page can be written and read as a stream and a partial
// read has to touch only the chunks it falls in. The magic and the sizes keep
// a stale file from another material or metric from being taken for a live one.
struct Spill_Header
{
	uint64_t magic;
	uint64_t uncompressed_size;
	uint64_t compressed_size;
};
static_assert(sizeof(Spill_Header) == 24);

constexpr size_t SPILL_CHUNK_BYTES = 64ull * 1024ull * 1024ull;
static_assert(SPILL_CHUNK_BYTES <= static_cast<size_t>(LZ4_MAX_INPUT_SIZE));

NODISCARD size_t spill_scratch_bytes(size_t bytes)
{
	const size_t chunk = bytes < SPILL_CHUNK_BYTES ? bytes : SPILL_CHUNK_BYTES;
	return static_cast<size_t>(LZ4_compressBound(narrowing_static_cast<int>(chunk)));
}

void decompress_chunk_into(
	Const_Span<uint8_t> src,
	Span<uint8_t> dst,
	const std::string& name
)
{
	const int ret = LZ4_decompress_safe(
		reinterpret_cast<const char*>(src.data()),
		reinterpret_cast<char*>(dst.data()),
		narrowing_static_cast<int>(src.size()),
		narrowing_static_cast<int>(dst.size()));

	if (ret <= 0 || static_cast<size_t>(ret) != dst.size())
		print_and_abort("Spill file LZ4 decompress failed: %s\n", name.c_str());
}

}  // namespace

void load_page_raw(
	Span<uint8_t> data,
	const std::filesystem::path& path,
	uint64_t magic
)
{
	const size_t bytes = data.size();
	const std::string name = path.string();

	std::ifstream fp(path, std::ios::binary);
	if (!fp)
		print_and_abort("Could not open scratch page for reading: %s\n", name.c_str());

	Huge_Array<uint8_t> in(For_Overwrite_Tag{}, spill_scratch_bytes(bytes));
	size_t off = 0;
	while (off < bytes)
	{
		Spill_Header header;
		fp.read(reinterpret_cast<char*>(&header), sizeof(header));
		if (!fp)
			print_and_abort("Spill file truncated header: %s\n", name.c_str());

		if (header.magic != magic)
			print_and_abort("Stale or mismatched scratch page: %s\n", name.c_str());

		if (header.uncompressed_size > bytes - off)
			print_and_abort("Spill file size mismatch: %s\n", name.c_str());

		if (header.compressed_size > in.size())
			in = Huge_Array<uint8_t>(For_Overwrite_Tag{}, header.compressed_size);

		fp.read(reinterpret_cast<char*>(in.data()), header.compressed_size);
		if (!fp)
			print_and_abort("Spill file truncated payload: %s\n", name.c_str());

		decompress_chunk_into(
			Const_Span<uint8_t>(in.data(), header.compressed_size),
			Span<uint8_t>(data.data() + off, header.uncompressed_size),
			name);

		off += header.uncompressed_size;
	}
}

// Marks the pages the closed `slice_plies` reach of `slice` spans, appending
// the newly marked ones to `touched` so that both the count and the reset stay
// proportional to the reach rather than to the page count.
static void mark_reach(
	std::vector<uint8_t>& needed,
	std::vector<size_t>& touched,
	const Page_Layout& pages,
	const Slice_Reach& reach,
	size_t slice,
	int slice_plies
)
{
	if (slice_plies < 0)
		return;

	const auto mark = [&](size_t target_slice) {
		const size_t page = pages.page_of_slice(target_slice);
		if (needed[page])
			return;
		needed[page] = 1;
		touched.emplace_back(page);
	};

	if (slice_plies == 0)
	{
		mark(slice);
		return;
	}

	for (const int32_t target : reach.closed(static_cast<size_t>(slice_plies), slice))
		mark(static_cast<size_t>(target));
}

size_t max_dispatch_page_count(
	const Page_Layout& pages,
	const Slice_Reach& reach,
	Color slice_color,
	Phase_Pages pass
)
{
	std::vector<uint8_t> me_needed(pages.num_pages(), 0);
	std::vector<uint8_t> opp_needed(pages.num_pages(), 0);
	std::vector<size_t> me_touched;
	std::vector<size_t> opp_touched;

	size_t worst = 0;
	int prev_me_plies = PHASE_PLIES_NONE - 1;
	int prev_opp_plies = PHASE_PLIES_NONE - 1;

	// A pass runs for both mover colors over the course of a generation, and
	// the color decides which plies can change the slice, so the budget has
	// to cover the wider of the two.
	for (const Color mover : { WHITE, BLACK })
	{
		const int me_plies = dispatch_slice_plies(
			pass.me, mover, slice_color, pass.plies_start_with_mover);
		const int opp_plies = dispatch_slice_plies(
			pass.opp, mover, slice_color, pass.plies_start_with_mover);

		// The two movers differ only where the slice color distinguishes them.
		if (me_plies == prev_me_plies && opp_plies == prev_opp_plies)
			continue;
		prev_me_plies = me_plies;
		prev_opp_plies = opp_plies;

		for (size_t page = 0; page < pages.num_pages(); ++page)
		{
			me_touched.clear();
			opp_touched.clear();

			const size_t slice_end = pages.end_slice_of_page(page);
			for (size_t slice = pages.first_slice_of_page(page); slice < slice_end; ++slice)
			{
				mark_reach(me_needed, me_touched, pages, reach, slice, me_plies);
				mark_reach(opp_needed, opp_touched, pages, reach, slice, opp_plies);
			}

			update_max(worst, me_touched.size() + opp_touched.size());

			for (const size_t p : me_touched)
				me_needed[p] = 0;
			for (const size_t p : opp_touched)
				opp_needed[p] = 0;
		}
	}

	return worst;
}

size_t max_lease_page_count(const Page_Layout& pages, const Slice_Reach& reach)
{
	size_t worst = 0;
	for (size_t slice = 0; slice < reach.num_slices(); ++slice)
	{
		// The reach is sorted, and page ids are monotonic in the slice, so
		// consecutive duplicates are adjacent.
		size_t count = 0;
		size_t last = std::numeric_limits<size_t>::max();
		for (const int32_t target : reach.closed(1, slice))
		{
			const size_t page = pages.page_of_slice(static_cast<size_t>(target));
			if (page == last)
				continue;
			last = page;
			count += 1;
		}
		update_max(worst, count);
	}
	return worst;
}

size_t generation_floor_page_count(
	const Page_Layout& pages,
	const Slice_Reach& reach,
	Color slice_color,
	bool has_check_chase
)
{
	size_t worst = 0;
	for (const Phase_Pages& pass : generation_pass_shapes(has_check_chase))
		update_max(worst, max_dispatch_page_count(pages, reach, slice_color, pass));

	// The check/chase look-ahead leases one slice's one-ply reach on top of its
	// dispatch's pin set; see DTM_Generator::check_double_chase_win.
	if (has_check_chase)
		worst += max_lease_page_count(pages, reach);

	return worst;
}

Packed_Page pack_page_raw(Const_Span<uint8_t> data)
{
	const size_t bytes = data.size();
	const size_t num_chunks = ceil_div(bytes, SPILL_CHUNK_BYTES);
	Huge_Array<uint8_t> scratch(For_Overwrite_Tag{}, spill_scratch_bytes(bytes));

	Packed_Page packed;
	packed.chunks.resize(num_chunks);
	for (size_t i = 0; i < num_chunks; ++i)
	{
		const size_t off = i * SPILL_CHUNK_BYTES;
		const size_t remain = bytes - off;
		const size_t chunk_bound = spill_scratch_bytes(remain);

		const int src_size = narrowing_static_cast<int>(
			remain < SPILL_CHUNK_BYTES ? remain : SPILL_CHUNK_BYTES);

		const int n = LZ4_compress_default(
			reinterpret_cast<const char*>(data.data() + off),
			reinterpret_cast<char*>(scratch.data()),
			src_size,
			narrowing_static_cast<int>(chunk_bound));
		if (n <= 0)
			print_and_abort("LZ4 compress failed for a page held in memory\n");

		Packed_Page::Chunk& chunk = packed.chunks[i];
		chunk.data = Huge_Array<uint8_t>(For_Overwrite_Tag{}, static_cast<size_t>(n));
		std::memcpy(chunk.data.data(), scratch.data(), static_cast<size_t>(n));
		chunk.uncompressed_size = static_cast<uint32_t>(src_size);
	}
	return packed;
}

void unpack_page_raw(Span<uint8_t> data, Packed_Page& packed)
{
	static const std::string name = "a page held in memory";

	size_t off = 0;
	for (Packed_Page::Chunk& chunk : packed.chunks)
	{
		if (chunk.uncompressed_size > data.size() - off)
			print_and_abort("Packed page size mismatch: %s\n", name.c_str());

		decompress_chunk_into(
			Const_Span<uint8_t>(chunk.data.data(), chunk.data.size()),
			Span<uint8_t>(data.data() + off, chunk.uncompressed_size),
			name);

		off += chunk.uncompressed_size;
		chunk.data.clear();
	}

	if (off != data.size())
		print_and_abort("Packed page shorter than the page: %s\n", name.c_str());
}

void save_packed_page(
	const Packed_Page& packed,
	const std::filesystem::path& path,
	uint64_t magic
)
{
	const std::string name = path.string();

	std::ofstream fp(path, std::ios::binary | std::ios::trunc);
	if (!fp)
		print_and_abort("Could not open scratch page for writing: %s\n", name.c_str());

	for (const Packed_Page::Chunk& chunk : packed.chunks)
	{
		const Spill_Header header{
			magic,
			chunk.uncompressed_size,
			chunk.data.size()
		};

		fp.write(reinterpret_cast<const char*>(&header), sizeof(header));
		fp.write(reinterpret_cast<const char*>(chunk.data.data()),
			narrowing_static_cast<std::streamsize>(chunk.data.size()));
	}
	if (!fp)
		print_and_abort("Could not write scratch page: %s\n", name.c_str());

	fp.close();
	if (!fp)
		print_and_abort("Could not close scratch page: %s\n", name.c_str());
}

Page_Cache::Page_Cache(std::vector<Pageable_Table*> tables, size_t capacity_bytes) :
	m_tables(std::move(tables)),
	m_capacity_bytes(capacity_bytes)
{
	// Anything the generator's pager left behind is tracked unpinned, in
	// whichever form it left it, so the cap applies to it too.
	for (size_t ti = 0; ti < m_tables.size(); ++ti)
	{
		if (m_tables[ti] == nullptr)
			continue;
		for (size_t p = 0; p < m_tables[ti]->num_pages(); ++p)
		{
			const size_t bytes = m_tables[ti]->page_ram_bytes(p);
			if (bytes == 0)
				continue;

			const bool packed = !m_tables[ti]->is_page_resident(p);
			const Key key{ ti, p };
			m_lru.push_back(key);
			m_entries[key] = Entry{
				0, packed ? State::PACKED : State::RESIDENT, std::prev(m_lru.end())
			};
			m_ram_bytes += bytes;
		}
	}
}

void Page_Cache::acquire(size_t table_idx, size_t page)
{
	std::unique_lock<std::mutex> lock(m_mutex);
	const Key key{ table_idx, page };

	for (;;)
	{
		auto it = m_entries.find(key);
		if (it == m_entries.end())
		{
			// Reserved pinned before the lock is dropped, so a concurrent
			// sweep can neither evict nor double-load it.
			m_lru.push_back(key);
			m_entries.emplace(key, Entry{ 1, State::LOADING, std::prev(m_lru.end()) });
			const size_t was_bytes = m_tables[table_idx]->page_ram_bytes(page);
			m_ram_bytes = m_ram_bytes - was_bytes + m_tables[table_idx]->page_bytes(page);

			// One page pinned, so at most a page or two to give up: not worth
			// a fan-out even where one would be legal, and this path has no
			// pool in hand anyway.
			sweep(lock, table_idx);
			load_pinned(lock, key);
			return;
		}

		if (it->second.state == State::RESIDENT)
		{
			it->second.pins += 1;
			// Splicing keeps the iterator valid, so the entry needs no update.
			m_lru.splice(m_lru.end(), m_lru, it->second.lru);
			return;
		}

		// Compressed in place: a hit, but it has to be brought back first.
		if (it->second.state == State::PACKED)
		{
			it->second.pins += 1;
			it->second.state = State::LOADING;
			m_lru.splice(m_lru.end(), m_lru, it->second.lru);
			const size_t was_bytes = m_tables[table_idx]->page_ram_bytes(page);
			m_ram_bytes = m_ram_bytes - was_bytes + m_tables[table_idx]->page_bytes(page);
			sweep(lock, table_idx);
			load_pinned(lock, key);
			return;
		}

		// LOADING or EVICTING: another thread owns the transition.
		m_state_changed.wait(lock);
	}
}

void Page_Cache::load_pinned(std::unique_lock<std::mutex>& lock, const Key& key)
{
	Pageable_Table& table = *m_tables[key.first];

	if (!table.is_page_resident(key.second))
	{
		lock.unlock();
		table.load_page(key.second);
		lock.lock();
	}

	m_entries.find(key)->second.state = State::RESIDENT;
	m_state_changed.notify_all();
}

void Page_Cache::acquire_all(
	In_Out_Param<Thread_Pool> thread_pool,
	Const_Span<Key> keys,
	size_t protected_dirty_table
)
{
	// Pages this call owns the LOADING state of, and the subset of them that is
	// not resident yet.
	std::vector<Key> reserved;
	std::vector<Key> to_read;

	// Pages another dispatch is loading or evicting right now. Waiting for them
	// is left until this set's own reads are done, so nothing blocks behind a
	// transition this thread could be filling instead.
	std::vector<Key> to_wait;

	{
		std::unique_lock<std::mutex> lock(m_mutex);

		for (const Key& key : keys)
		{
			auto it = m_entries.find(key);
			if (it == m_entries.end())
			{
				m_lru.push_back(key);
				m_entries.emplace(key, Entry{ 1, State::LOADING, std::prev(m_lru.end()) });
				reserved.emplace_back(key);
				Pageable_Table& table = *m_tables[key.first];
				const size_t was_bytes = table.page_ram_bytes(key.second);
				m_ram_bytes = m_ram_bytes - was_bytes + table.page_bytes(key.second);
				if (!m_tables[key.first]->is_page_resident(key.second))
					to_read.emplace_back(key);
			}
			else if (it->second.state == State::RESIDENT)
			{
				it->second.pins += 1;
				m_lru.splice(m_lru.end(), m_lru, it->second.lru);
			}
			else if (it->second.state == State::PACKED)
			{
				it->second.pins += 1;
				it->second.state = State::LOADING;
				m_lru.splice(m_lru.end(), m_lru, it->second.lru);
				reserved.emplace_back(key);
				to_read.emplace_back(key);
				Pageable_Table& table = *m_tables[key.first];
				const size_t was_bytes = table.page_ram_bytes(key.second);
				m_ram_bytes = m_ram_bytes - was_bytes + table.page_bytes(key.second);
			}
			else
				to_wait.emplace_back(key);
		}

		// One sweep for the whole set: the reservations above are pinned, so
		// what it frees is what the set needs on top of them.
		sweep(lock, thread_pool, protected_dirty_table);
	}

	if (to_read.size() == 1)
	{
		// Not worth a fan-out.
		m_tables[to_read[0].first]->load_page(to_read[0].second);
	}
	else if (!to_read.empty())
	{
		std::atomic<size_t> next(0);
		thread_pool->run_sync_task_on_multiple_threads(
			std::min(MAX_PAGING_THREADS, thread_pool->num_workers()), [&](size_t) {
			for (;;)
			{
				const size_t i = next.fetch_add(1, std::memory_order_relaxed);
				if (i >= to_read.size())
					return;
				m_tables[to_read[i].first]->load_page(to_read[i].second);
			}
		});
	}

	{
		std::lock_guard<std::mutex> lock(m_mutex);
		for (const Key& key : reserved)
			m_entries.find(key)->second.state = State::RESIDENT;
		m_state_changed.notify_all();
	}

	for (const Key& key : to_wait)
		acquire(key.first, key.second);
}

void Page_Cache::release(size_t table_idx, size_t page)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	auto it = m_entries.find({ table_idx, page });
	ASSERT(it != m_entries.end());
	ASSERT(it->second.pins > 0);
	if (it != m_entries.end() && it->second.pins > 0)
		it->second.pins -= 1;
}

std::vector<Page_Cache::Victim> Page_Cache::select_victims(
	size_t budget_bytes,
	size_t protected_dirty_table,
	bool spill_compressed
)
{
	size_t live = m_ram_bytes - m_evicting_bytes;
	if (live <= budget_bytes)
		return {};

	std::vector<Victim> victims;

	// Compressing a resident page gives back an amount nothing knows until it
	// has run, so it is credited half its size and settled for real in
	// finish_eviction. A page leaving memory outright is credited all of it.
	const auto take = [&](const Key& key, Entry& entry, bool resident) {
		const size_t was_bytes = m_tables[key.first]->page_ram_bytes(key.second);
		const size_t freed = resident ? was_bytes / 2 : was_bytes;
		entry.state = State::EVICTING;
		m_evicting_bytes += freed;
		live -= std::min(live, freed);
		victims.push_back(Victim{ key, was_bytes, resident });
	};

	// Compressing costs no I/O, so resident pages give up a level first. Clean
	// ones are cheap to drop later, so take them least recently used first.
	for (const Key& key : m_lru)
	{
		if (live <= budget_bytes)
			return victims;

		Entry& entry = m_entries.find(key)->second;
		if (entry.state != State::RESIDENT || entry.pins != 0
		    || m_tables[key.first]->is_page_dirty(key.second))
			continue;

		take(key, entry, true);
	}

	// A forward dispatch sweep next reuses the most recently acquired dirty
	// page last, so walk dirty candidates from most to least recent. Keep the
	// protected table's dirty pages until the other candidates are gone,
	// then evict those least recently used first.
	for (const bool protected_dirty : { false, true })
	{
		auto oldest = m_lru.begin();
		auto newest = m_lru.rbegin();
		for (size_t remaining = m_lru.size(); remaining != 0; --remaining)
		{
			if (live <= budget_bytes)
				return victims;

			const Key& key = protected_dirty ? *oldest++ : *newest++;
			Entry& entry = m_entries.find(key)->second;
			if (entry.state != State::RESIDENT || entry.pins != 0
			    || !m_tables[key.first]->is_page_dirty(key.second)
			    || (key.first == protected_dirty_table) != protected_dirty)
				continue;

			take(key, entry, true);
		}
	}

	// Only now does anything go to disk, least recently used first.
	if (spill_compressed)
		for (const Key& key : m_lru)
		{
			if (live <= budget_bytes)
				return victims;

			Entry& entry = m_entries.find(key)->second;
			if (entry.state != State::PACKED || entry.pins != 0)
				continue;

			take(key, entry, false);
		}

	return victims;
}

void Page_Cache::evict_victims(const std::vector<Victim>& victims)
{
	for (const Victim& victim : victims)
		m_tables[victim.key.first]->evict_page(victim.key.second);
}

void Page_Cache::evict_victims(
	In_Out_Param<Thread_Pool> thread_pool,
	const std::vector<Victim>& victims
)
{
	// A demotion compresses the page or writes it out, so with several to do
	// this is worth spreading over the pool rather than paying serially here.
	if (victims.size() < 2)
	{
		evict_victims(victims);
		return;
	}

	std::atomic<size_t> next(0);
	thread_pool->run_sync_task_on_multiple_threads(
		std::min(MAX_PAGING_THREADS, thread_pool->num_workers()), [&](size_t) {
		for (;;)
		{
			const size_t i = next.fetch_add(1, std::memory_order_relaxed);
			if (i >= victims.size())
				return;
			m_tables[victims[i].key.first]->evict_page(victims[i].key.second);
		}
	});
}

void Page_Cache::finish_eviction(const std::vector<Victim>& victims)
{
	for (const Victim& victim : victims)
	{
		const size_t now_bytes = m_tables[victim.key.first]->page_ram_bytes(victim.key.second);
		m_ram_bytes = m_ram_bytes - victim.was_bytes + now_bytes;
		m_evicting_bytes -= (victim.was_resident && now_bytes != 0)
			? victim.was_bytes / 2 : victim.was_bytes;

		auto it = m_entries.find(victim.key);
		if (now_bytes != 0)
		{
			// Compressed, and kept: a later acquire unpacks it from memory.
			it->second.state = State::PACKED;
		}
		else
		{
			m_lru.erase(it->second.lru);
			m_entries.erase(it);
		}
	}

	m_state_changed.notify_all();
}

void Page_Cache::sweep(
	std::unique_lock<std::mutex>& lock,
	size_t protected_dirty_table
)
{
	for (const bool spill : { false, true })
		for (;;)
		{
			// Resident-page compression is selected using a provisional half-page
			// credit. Settle its actual packed size, then select again: a weakly
			// compressible page may still leave us over budget and must be allowed
			// to proceed to the packed-to-disk stage below.
			const std::vector<Victim> victims = select_victims(
				m_capacity_bytes, protected_dirty_table, spill);
			if (victims.empty())
				break;

			lock.unlock();
			evict_victims(victims);
			lock.lock();

			finish_eviction(victims);
		}
}

void Page_Cache::sweep(
	std::unique_lock<std::mutex>& lock,
	In_Out_Param<Thread_Pool> thread_pool,
	size_t protected_dirty_table
)
{
	for (const bool spill : { false, true })
		for (;;)
		{
			const std::vector<Victim> victims = select_victims(
				m_capacity_bytes, protected_dirty_table, spill);
			if (victims.empty())
				break;

			lock.unlock();
			evict_victims(thread_pool, victims);
			lock.lock();

			finish_eviction(victims);
		}
}

void Page_Cache::flush_all(In_Out_Param<Thread_Pool> thread_pool, size_t retain_bytes)
{
	// Every page leaves the resident state, whatever the retention budget: the
	// sweep's band buffer is sized on the pages' share of memory being free.
	for (const bool spill : { false, true })
		for (;;)
		{
			std::vector<Victim> victims;
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				victims = select_victims(
					spill ? retain_bytes : 0, m_tables.size(), spill);
			}

			if (victims.empty())
				break;

			evict_victims(thread_pool, victims);

			std::lock_guard<std::mutex> lock(m_mutex);
			finish_eviction(victims);
		}
}

namespace {

// One step of a band inside one page: where it starts in the uncompressed page,
// and where it belongs in the band buffer. All steps are `step_bytes` long.
struct Band_Piece
{
	uint64_t page_offset;
	size_t band_offset;
};

}  // namespace

std::filesystem::path Paged_Logical_Reader::page_path(size_t page) const
{
	return m_scratch_prefix.string() + ".p" + std::to_string(page);
}

Paged_Logical_Reader::Paged_Logical_Reader(
	const Pageable_Table& table,
	const Slice_Layout& slices,
	const Page_Layout& pages,
	std::filesystem::path scratch_prefix,
	uint64_t magic,
	size_t entry_bytes,
	size_t band_budget_bytes,
	size_t step_alignment
) :
	m_table(&table),
	m_slices(slices),
	m_pages(pages),
	m_scratch_prefix(std::move(scratch_prefix)),
	m_magic(magic),
	m_entry_bytes(entry_bytes)
{
	const size_t step_bytes = m_slices.low_weight() * entry_bytes;

	// Steps needed for the alignment: a step is low_weight entries, so once
	// that many entries make a whole number of output bytes, so does a step.
	const size_t align = std::max<size_t>(1,
		step_alignment / std::gcd(step_alignment, std::max<size_t>(1, m_slices.low_weight())));

	m_num_steps = m_slices.num_slices() * m_slices.num_high();
	m_steps_per_band = std::max<size_t>(1, band_budget_bytes / std::max<size_t>(1, step_bytes));
	m_steps_per_band = std::max(align, (m_steps_per_band / align) * align);
	m_steps_per_band = std::min(m_steps_per_band, m_num_steps);

	m_band_bytes = m_steps_per_band * step_bytes;
	m_total_bytes = m_slices.num_entries() * entry_bytes;

	m_band.resize(m_band_bytes);
}

Span<uint8_t> Paged_Logical_Reader::next_band(In_Out_Param<Thread_Pool> thread_pool)
{
	if (m_next_step >= m_num_steps)
		return Span<uint8_t>(m_band.data(), size_t(0));

	const size_t step_begin = m_next_step;
	const size_t step_end = std::min(step_begin + m_steps_per_band, m_num_steps);

	const size_t w = m_slices.low_weight();
	const size_t r = m_slices.num_slices();
	const size_t step_bytes = w * m_entry_bytes;

	std::atomic<size_t> next_page(0);

	// A page holds at most this many bytes, so one chunk of it is at most this
	// large however the pages divide the slices.
	const size_t max_page_bytes =
		m_pages.slices_per_page() * m_slices.slice_size() * m_entry_bytes;

	thread_pool->run_sync_task_on_multiple_threads(
		std::min(MAX_PAGING_THREADS, thread_pool->num_workers()), [&](size_t) {
		// One chunk in each form, allocated once per worker: a resize inside
		// the chunk loop would zero-fill before every read.
		Huge_Array<uint8_t> compressed(For_Overwrite_Tag{}, spill_scratch_bytes(max_page_bytes));
		Huge_Array<uint8_t> plain(For_Overwrite_Tag{},
			max_page_bytes < SPILL_CHUNK_BYTES ? max_page_bytes : SPILL_CHUNK_BYTES);
		std::vector<Band_Piece> pieces;

		for (;;)
		{
			const size_t page = next_page.fetch_add(1, std::memory_order_relaxed);
			if (page >= m_pages.num_pages())
				return;

			const size_t first_slice = m_pages.first_slice_of_page(page);
			const size_t slice_end = m_pages.end_slice_of_page(page);
			const size_t page_bytes =
				(slice_end - first_slice) * m_slices.slice_size() * m_entry_bytes;

			pieces.clear();
			for (size_t slice = first_slice; slice < slice_end; ++slice)
			{
				// Steps of this slice are those congruent to it modulo the
				// slice count, so within the band they form a contiguous
				// range of the slice's storage.
				const size_t offset_in_cycle = step_begin % r;
				const size_t first_step = slice >= offset_in_cycle
					? step_begin + (slice - offset_in_cycle)
					: step_begin + (r - offset_in_cycle) + slice;
				if (first_step >= step_end)
					continue;

				const size_t count = (step_end - 1 - first_step) / r + 1;
				const size_t high_begin = first_step / r;

				const size_t in_page_entry =
					(slice - first_slice) * m_slices.slice_size() + w * high_begin;

				// Ascending in both the slice and the step, so the pieces come
				// out ordered by page offset and the walk below never rewinds.
				for (size_t k = 0; k < count; ++k)
					pieces.push_back(Band_Piece{
						static_cast<uint64_t>((in_page_entry + k * w) * m_entry_bytes),
						(first_step + k * r - step_begin) * step_bytes
					});
			}

			if (pieces.empty())
				continue;

			// Copies the stretches of one decompressed chunk that the band
			// wants, and reports whether the walk can move past it.
			size_t next_piece = 0;
			const auto gather_chunk = [&](
				Const_Span<uint8_t> chunk, uint64_t chunk_begin)
			{
				const uint64_t chunk_end = chunk_begin + chunk.size();
				while (next_piece < pieces.size()
					&& pieces[next_piece].page_offset < chunk_end)
				{
					const Band_Piece& piece = pieces[next_piece];
					const uint64_t piece_end = piece.page_offset + step_bytes;
					const uint64_t from = std::max(piece.page_offset, chunk_begin);
					const uint64_t to = std::min(piece_end, chunk_end);

					std::memcpy(
						m_band.data() + piece.band_offset + (from - piece.page_offset),
						chunk.data() + (from - chunk_begin),
						to - from);

					// A step straddling the boundary is finished by the next
					// chunk.
					if (piece_end > chunk_end)
						break;

					++next_piece;
				}
			};

			// Still uncompressed: the stretches are just copied out.
			const Const_Span<uint8_t> resident = m_table->resident_page_span(page);
			if (!resident.empty())
			{
				gather_chunk(resident, 0);
				continue;
			}

			// Compressed in memory. The records are the chunks a spilled page
			// is made of, so the walk below is the same one, without the reads.
			if (const Packed_Page* const packed = m_table->packed_page(page))
			{
				static const std::string held = "a page held in memory";
				uint64_t chunk_begin = 0;
				for (const Packed_Page::Chunk& chunk : packed->chunks)
				{
					if (next_piece >= pieces.size())
						break;

					if (chunk.uncompressed_size > page_bytes - chunk_begin)
						print_and_abort("Packed page size mismatch: %s\n", held.c_str());

					const uint64_t chunk_end = chunk_begin + chunk.uncompressed_size;

					if (pieces[next_piece].page_offset < chunk_end)
					{
						decompress_chunk_into(
							Const_Span<uint8_t>(chunk.data.data(), chunk.data.size()),
							Span<uint8_t>(plain.data(), chunk.uncompressed_size),
							held);
						gather_chunk(
							Const_Span<uint8_t>(plain.data(), chunk.uncompressed_size),
							chunk_begin);
					}

					chunk_begin = chunk_end;
				}
				continue;
			}

			const std::filesystem::path path = page_path(page);
			const std::string name = path.string();

			std::ifstream fp(path, std::ios::binary);
			if (!fp)
				print_and_abort("Missing spilled distance page: %s\n", name.c_str());

			// The page is a stream of compressed chunks, so the wanted stretches
			// are gathered by walking it forward, reading a chunk's payload only
			// when a stretch falls in it and skipping past the rest.
			uint64_t chunk_begin = 0;
			while (next_piece < pieces.size())
			{
				Spill_Header header;
				fp.read(reinterpret_cast<char*>(&header), sizeof(header));
				if (!fp)
					print_and_abort("Spill file truncated header: %s\n", name.c_str());

				if (header.magic != m_magic)
					print_and_abort("Stale or mismatched scratch page: %s\n", name.c_str());

				if (header.uncompressed_size > page_bytes - chunk_begin)
					print_and_abort("Spill file size mismatch: %s\n", name.c_str());

				const uint64_t chunk_end = chunk_begin + header.uncompressed_size;

				if (pieces[next_piece].page_offset >= chunk_end)
				{
					fp.seekg(static_cast<std::streamoff>(header.compressed_size), std::ios::cur);
					if (!fp)
						print_and_abort("Spill file truncated payload: %s\n", name.c_str());
				}
				else
				{
					fp.read(reinterpret_cast<char*>(compressed.data()), header.compressed_size);
					if (!fp)
						print_and_abort("Spill file truncated payload: %s\n", name.c_str());

					decompress_chunk_into(
						Const_Span<uint8_t>(compressed.data(), header.compressed_size),
						Span<uint8_t>(plain.data(), header.uncompressed_size),
						name);

					gather_chunk(
						Const_Span<uint8_t>(plain.data(), header.uncompressed_size),
						chunk_begin);
				}

				chunk_begin = chunk_end;
			}
		}
	});

	m_next_step = step_end;
	return Span<uint8_t>(m_band.data(), (step_end - step_begin) * step_bytes);
}
