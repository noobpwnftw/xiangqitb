#pragma once

#include "egtb_slice.h"

#include "util/defines.h"
#include "util/math.h"
#include "util/param.h"
#include "util/span.h"
#include "util/thread_pool.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Bounded-memory paging for the DTC/DTM distance arrays, shared by both
// metrics.
//
// A distance array is cut along the slice dimension (see Slice_Layout) into
// pages of adjacent slices. A page is the unit of residency: either fully in
// memory or spilled under `tmpdir`. Pages are pinned for one dispatch and
// released afterwards, so a worker never holds a pointer into a page that
// could be evicted underneath it.

// Adjacent slices are bundled up to this; the planner shrinks the bundle when
// the budget demands.
inline constexpr size_t PREFERRED_PAGE_BYTES = 64ull * 1024 * 1024;

struct Page_Layout
{
	Page_Layout() = default;

	Page_Layout(size_t num_slices, size_t slices_per_page) :
		m_num_slices(num_slices),
		m_slices_per_page(std::max<size_t>(1, std::min(slices_per_page, num_slices))),
		m_num_pages(ceil_div(num_slices, m_slices_per_page))
	{
	}

	// The largest bundle of adjacent slices that keeps a page at or under
	// `PREFERRED_PAGE_BYTES`, but at least one slice.
	NODISCARD static size_t preferred_slices_per_page(size_t bytes_per_slice)
	{
		if (bytes_per_slice == 0 || bytes_per_slice >= PREFERRED_PAGE_BYTES)
			return 1;
		return PREFERRED_PAGE_BYTES / bytes_per_slice;
	}

	NODISCARD size_t num_slices() const { return m_num_slices; }
	NODISCARD size_t slices_per_page() const { return m_slices_per_page; }
	NODISCARD size_t num_pages() const { return m_num_pages; }

	NODISCARD size_t page_of_slice(size_t slice) const
	{
		ASSERT(slice < m_num_slices);
		return slice / m_slices_per_page;
	}

	NODISCARD size_t first_slice_of_page(size_t page) const
	{
		ASSERT(page < m_num_pages);
		return page * m_slices_per_page;
	}

	NODISCARD size_t end_slice_of_page(size_t page) const
	{
		return std::min(first_slice_of_page(page) + m_slices_per_page, m_num_slices);
	}

	NODISCARD size_t slices_in_page(size_t page) const
	{
		return end_slice_of_page(page) - first_slice_of_page(page);
	}

private:
	size_t m_num_slices = 1;
	size_t m_slices_per_page = 1;
	size_t m_num_pages = 1;
};

// Spills a page as a stream of independently compressed chunks. Each chunk's
// header carries `magic`, which keeps a stale file from another material or
// metric from being mistaken for a live one.
void save_page_raw(
	Const_Span<uint8_t> data,
	const std::filesystem::path& path,
	uint64_t magic
);

void load_page_raw(
	Span<uint8_t> data,
	const std::filesystem::path& path,
	uint64_t magic
);

// The part of a paged table the cache needs, entry type erased so DTC and DTM
// share one implementation.
struct Pageable_Table
{
	virtual ~Pageable_Table() = default;

	NODISCARD virtual size_t num_pages() const = 0;
	NODISCARD virtual bool is_page_resident(size_t page) const = 0;

	// Whether evicting the page would have to write it out.
	NODISCARD virtual bool is_page_dirty(size_t page) const = 0;

	// Allocates the page and, if it was spilled before, reads it back.
	virtual void load_page(size_t page) = 0;

	// Writes the page out if it is dirty, then frees it.
	virtual void evict_page(size_t page) = 0;
};

// The deepest quiet ply at which a pass dereferences a table, counted from the
// dispatched position (0 being that position), or PHASE_PLIES_NONE.
using Phase_Plies = int;
inline constexpr Phase_Plies PHASE_PLIES_NONE = -1;

// What one pass may touch, split by table.
struct Phase_Pages
{
	Phase_Plies me = PHASE_PLIES_NONE;
	Phase_Plies opp = PHASE_PLIES_NONE;

	// Whether the look-ahead starts with a move by the dispatched side to
	// move. Set through the factory functions below, so every pass states it.
	bool plies_start_with_mover = true;
};

// A pass whose look-ahead starts with a move by the dispatched side to move,
// so its quiet plies alternate colors from there.
NODISCARD constexpr Phase_Pages phase_pages(
	Phase_Plies me,
	Phase_Plies opp = PHASE_PLIES_NONE)
{
	return Phase_Pages{ me, opp, true };
}

// A pass that takes each color in turn as the mover within a single visit, so
// its first ply can be by either side.
NODISCARD constexpr Phase_Pages phase_pages_both_movers(
	Phase_Plies plies)
{
	return Phase_Pages{ plies, plies, false };
}

// How many of `quiet_plies` from the dispatched position can change the slice.
//
// Quiet plies alternate colors starting with `mover`, and only a move of the
// slice group changes the slice; that group belongs to one color. So only
// every other ply counts. Both tables share one slice geometry, so which is
// read does not matter.
//
// Returns -1 for an untouched table, 0 for the dispatched slice only, else the
// number of slice-group plies whose closed reach must be pinned.
NODISCARD constexpr int slice_plies_in_lookahead(
	Phase_Plies quiet_plies, Color mover, Color slice_color)
{
	if (quiet_plies < 0)
		return -1;

	return slice_color == mover
		? (quiet_plies + 1) / 2      // plies 1, 3, 5, ... are the mover's
		: quiet_plies / 2;           // plies 2, 4, 6, ... are the opponent's
}

// Closure depth a table needs for one dispatch, unioning over both movers when
// the pass does not start from the dispatched side.
NODISCARD constexpr int dispatch_slice_plies(
	Phase_Plies quiet_plies, Color mover, Color slice_color, bool plies_start_with_mover)
{
	const int from_mover = slice_plies_in_lookahead(quiet_plies, mover, slice_color);
	if (plies_start_with_mover)
		return from_mover;

	const int from_other = slice_plies_in_lookahead(quiet_plies, color_opp(mover), slice_color);
	return from_mover > from_other ? from_mover : from_other;
}

// Deepest closure any dispatch of `pass` can need, over both tables and both
// movers. This is what Slice_Reach has to be built to.
NODISCARD constexpr size_t required_reach_plies(Phase_Pages pass, Color slice_color)
{
	int deepest = 1;
	for (const Color mover : { WHITE, BLACK })
		for (const Phase_Plies plies : { pass.me, pass.opp })
		{
			const int needed = dispatch_slice_plies(
				plies, mover, slice_color, pass.plies_start_with_mover);
			if (needed > deepest)
				deepest = needed;
		}
	return static_cast<size_t>(deepest);
}

// The pass shapes a generation runs, in quiet plies per table. The floor is
// the maximum over these rather than over one hand-picked "widest" shape, so
// it cannot drift from what the dispatchers declare.
struct Generation_Pass_Shapes
{
	std::array<Phase_Pages, 4> shapes{};
	size_t count = 0;

	NODISCARD const Phase_Pages* begin() const { return shapes.data(); }
	NODISCARD const Phase_Pages* end() const { return shapes.data() + count; }
};

NODISCARD constexpr Generation_Pass_Shapes generation_pass_shapes(bool has_check_chase)
{
	// Initialization and the final sweep touch both tables at the dispatched
	// slice; retrograde propagation adds the opponent one quiet ply on. With
	// check/chase adjudication there are two more: the fake-removal passes,
	// which read their own table two quiet plies on, and the flagging pass,
	// which takes each color as the mover within one visit.
	if (has_check_chase)
		return Generation_Pass_Shapes{ {
			phase_pages(0, 0),
			phase_pages(0, 1),
			phase_pages(2, 1),
			phase_pages_both_movers(1),
		}, 4 };

	return Generation_Pass_Shapes{ {
		phase_pages(0, 0),
		phase_pages(0, 1),
		Phase_Pages{},
		Phase_Pages{},
	}, 2 };
}

// Widest resident page set one dispatch of `pass` holds, across both tables
// and both movers. Swept exactly rather than bounded by the maximum reach,
// because a page bundles several slices whose reaches overlap.
NODISCARD size_t max_dispatch_page_count(
	const Page_Layout& pages,
	const Slice_Reach& reach,
	Color slice_color,
	Phase_Pages pass
);

// Pages one slice's closed one-ply reach can span: what the check/chase
// look-ahead's transient lease adds on top of its dispatch's pin set.
NODISCARD size_t max_lease_page_count(
	const Page_Layout& pages,
	const Slice_Reach& reach
);

// Pages a generation can hold resident at once: the widest dispatch pin set
// over all pass shapes, plus that transient lease.
NODISCARD size_t generation_floor_page_count(
	const Page_Layout& pages,
	const Slice_Reach& reach,
	Color slice_color,
	bool has_check_chase
);

// Pin/lease cache over a fixed set of paged tables.
//
// Residency is capped at `capacity` pages. Pinned pages are never evicted;
// unpinned ones are dropped in use order once the cap is exceeded. Dirty pages
// in the current source table are kept until the other candidates are gone.
// The mutex is taken on acquire/release only -- entry access never touches it.
struct Page_Cache
{
	using Key = std::pair<size_t, size_t>;

	Page_Cache(std::vector<Pageable_Table*> tables, size_t capacity);

	Page_Cache(const Page_Cache&) = delete;
	Page_Cache& operator=(const Page_Cache&) = delete;

	void acquire(size_t table_idx, size_t page);

	// Pins a whole dispatch's set at once. The pages that have to be read are
	// read on the pool rather than one at a time by the calling thread, and the
	// budget is applied to the set as a whole, the way a working set is.
	void acquire_all(
		In_Out_Param<Thread_Pool> thread_pool,
		Const_Span<Key> keys,
		size_t protected_dirty_table
	);

	void release(size_t table_idx, size_t page);

	// Evicts every unpinned page, flushing the dirty ones.
	void flush_all(In_Out_Param<Thread_Pool> thread_pool);

	NODISCARD size_t capacity() const { return m_capacity; }

private:
	enum struct State : uint8_t { LOADING, RESIDENT, EVICTING };

	struct Entry
	{
		size_t pins = 0;
		State state = State::LOADING;
		// Where this page sits in m_lru. Valid until the page is selected for
		// eviction, which is also the only thing that erases the node.
		std::list<Key>::iterator lru;
	};

	std::vector<Pageable_Table*> m_tables;
	size_t m_capacity;

	mutable std::mutex m_mutex;
	std::condition_variable m_state_changed;
	std::map<Key, Entry> m_entries;
	// Use order, least recently acquired at the front.
	std::list<Key> m_lru;
	// Entries being written out right now. They still occupy m_entries, but
	// their memory is already spoken for, so the budget must not count them.
	size_t m_evicting = 0;

	// Brings residency back to the budget; `lock` is released while the victims
	// are written out. The pool-less form is for acquire(), which pins one page
	// and so has at most a page or two to give up, and which must stay callable
	// without a pool in hand: it is the general single-page path, and a fan-out
	// from a pool worker would deadlock.
	void sweep(std::unique_lock<std::mutex>& lock, size_t protected_dirty_table);
	void sweep(
		std::unique_lock<std::mutex>& lock,
		In_Out_Param<Thread_Pool> thread_pool,
		size_t protected_dirty_table
	);

	// Marks the pages that have to go to bring residency down to `max_live` as
	// EVICTING and returns them. Called with the lock held; nothing else
	// touches a page once it is marked.
	NODISCARD std::vector<Key> select_victims(
		size_t max_live,
		size_t protected_dirty_table
	);


	// Writes the victims out. Called with the lock released. The pooled form
	// spreads them over the pool and defers to the other one when there are too
	// few to be worth a fan-out.
	void evict_victims(const std::vector<Key>& victims);
	void evict_victims(In_Out_Param<Thread_Pool> thread_pool, const std::vector<Key>& victims);

	// Drops the evicted entries. Called with the lock held.
	void finish_eviction(const std::vector<Key>& victims);
};

// Holds a set of pins for one dispatch. Taken by the dispatching thread before
// the parallel region and released after it joins, so workers never race the
// pager and never outlive a lease.
struct Pinned_Pages
{
	Pinned_Pages() = default;

	explicit Pinned_Pages(Page_Cache& cache) :
		m_cache(&cache)
	{
	}

	Pinned_Pages(const Pinned_Pages&) = delete;
	Pinned_Pages& operator=(const Pinned_Pages&) = delete;

	Pinned_Pages(Pinned_Pages&& other) noexcept :
		m_cache(std::exchange(other.m_cache, nullptr)),
		m_held(std::move(other.m_held))
	{
	}

	~Pinned_Pages()
	{
		release_all();
	}

	void pin(size_t table_idx, size_t page)
	{
		ASSERT(m_cache != nullptr);
		m_cache->acquire(table_idx, page);
		m_held.emplace_back(table_idx, page);
	}

	// Pins `keys` as one set; see Page_Cache::acquire_all.
	void pin_all(
		In_Out_Param<Thread_Pool> thread_pool,
		Const_Span<Page_Cache::Key> keys,
		size_t protected_dirty_table
	)
	{
		ASSERT(m_cache != nullptr);
		m_cache->acquire_all(thread_pool, keys, protected_dirty_table);
		for (const Page_Cache::Key& key : keys)
			m_held.emplace_back(key);
	}

	void release_all()
	{
		if (m_cache == nullptr)
			return;
		for (const auto& [table_idx, page] : m_held)
			m_cache->release(table_idx, page);
		m_held.clear();
	}

	NODISCARD size_t size() const { return m_held.size(); }

private:
	Page_Cache* m_cache = nullptr;
	std::vector<std::pair<size_t, size_t>> m_held;
};

// Streams a spilled paged distance table in logical Board_Index order.
//
// Logical order walks *steps*: step t covers the `low_weight` consecutive
// indices [w*t, w*(t+1)), which live in slice t % r at in-slice offset
// w * (t / r). So any run of consecutive steps is a contiguous logical range,
// and within it the steps of one slice form a contiguous range of that slice's
// storage. A band is therefore one contiguous chunk per slice, and across
// bands those chunks tile each slice front to back: the table is read exactly
// once however small the band is, so the band size trades read size against
// buffer memory, never against total I/O.
//
// Every page must already be spilled; flush the page cache first.
struct Paged_Logical_Reader
{
	Paged_Logical_Reader(
		const Slice_Layout& slices,
		const Page_Layout& pages,
		std::filesystem::path scratch_prefix,
		uint64_t magic,
		size_t entry_bytes,
		size_t band_budget_bytes,
		// Bands cover a multiple of this many steps, so that a consumer
		// packing several entries per output byte always gets whole bytes.
		size_t step_alignment
	);

	NODISCARD size_t total_bytes() const { return m_total_bytes; }

	// Reads the next band into the internal buffer and returns a view of it,
	// empty once the table is exhausted. The buffer is the reader's, valid
	// until the next call, and the caller may rewrite it in place. Must be
	// called sequentially.
	NODISCARD Span<uint8_t> next_band(In_Out_Param<Thread_Pool> thread_pool);

	Paged_Logical_Reader(const Paged_Logical_Reader&) = delete;
	Paged_Logical_Reader& operator=(const Paged_Logical_Reader&) = delete;

private:
	Slice_Layout m_slices;
	Page_Layout m_pages;
	std::filesystem::path m_scratch_prefix;
	uint64_t m_magic;
	size_t m_entry_bytes;

	size_t m_steps_per_band = 1;
	size_t m_num_steps = 0;
	size_t m_next_step = 0;
	size_t m_band_bytes = 0;
	size_t m_total_bytes = 0;

	std::vector<uint8_t> m_band;

	NODISCARD std::filesystem::path page_path(size_t page) const;
};
