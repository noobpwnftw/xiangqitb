#pragma once

#include "egtb.h"

#include "util/defines.h"
#include "util/division.h"
#include "util/param.h"
#include "util/span.h"

#include <cstdint>
#include <vector>

struct Piece_Config_For_Gen;

// Physical slice layout for pageable distance storage.
//
// The slice dimension is the compression group (Piece_Config_For_Gen::compress_id),
// which is NOT guaranteed to be the outermost index dimension -- for symmetric
// materials such as KAABBNCKAABBN the equal-ratio tie picks WHITE_DEFENDERS,
// the innermost one. So rather than assume a contiguous run of indices per
// slice, the group's mixed-radix digit is lifted out of the logical index:
//
//     low    = index % w          offset = low + w * high
//     slice  = (index / w) % r    index  = low + w * (slice + r * high)
//     high   = index / (w * r)
//
// where `w` is the group's index weight and `r` its radix. The logical
// numbering is untouched; only the physical placement changes, so a slice's
// entries are contiguous even where they interleave in flat index order.
struct Slice_Layout
{
	Slice_Layout() = default;

	explicit Slice_Layout(const Piece_Config_For_Gen& epsi);

	NODISCARD size_t num_slices() const { return m_num_slices; }

	// Entries in one slice. Every slice has exactly this many.
	NODISCARD size_t slice_size() const { return m_slice_size; }

	NODISCARD size_t num_entries() const { return m_num_entries; }

	NODISCARD size_t low_weight() const { return m_low_weight; }
	NODISCARD size_t num_high() const { return m_num_high; }

	struct Location
	{
		size_t slice;
		size_t offset;
	};

	NODISCARD INLINE Location locate(Board_Index index) const
	{
		const size_t i = static_cast<size_t>(index);
		ASSERT(i < m_num_entries);

		const size_t q = div_low(i);
		const size_t low = i - q * m_low_weight;

		// One `high` step means the group is the outermost index dimension:
		// the quotient is already the slice, and the second division would
		// only ever yield zero.
		if (m_num_high == 1)
			return Location{ q, low };

		const size_t high = div_slices(q);
		return Location{ q - high * m_num_slices, low + m_low_weight * high };
	}

	NODISCARD INLINE size_t slice_of(Board_Index index) const
	{
		const size_t i = static_cast<size_t>(index);
		ASSERT(i < m_num_entries);
		const size_t q = div_low(i);
		return m_num_high == 1 ? q : q - div_slices(q) * m_num_slices;
	}

	// First logical index of the run holding slices [slice, ...) at `high`.
	// Adjacent slices are adjacent in the logical index at fixed `high`, which
	// is what lets a page of adjacent slices sweep as few long runs.
	NODISCARD INLINE Board_Index run_begin(size_t slice, size_t high) const
	{
		ASSERT(slice < m_num_slices);
		ASSERT(high < m_num_high);
		return static_cast<Board_Index>(m_low_weight * (slice + m_num_slices * high));
	}

private:
	size_t m_num_slices = 1;
	size_t m_low_weight = 1;
	size_t m_num_high = 1;
	size_t m_slice_size = 1;
	size_t m_num_entries = 1;
	Divider<uint64_t> m_low_weight_div{};
	Divider<uint64_t> m_num_slices_div{};

	// Precomputed reciprocals. Divider rejects a divisor of 1, which is
	// exactly the case needing no division at all.
	NODISCARD INLINE size_t div_low(size_t v) const
	{
		return m_low_weight > 1 ? static_cast<size_t>(v / m_low_weight_div) : v;
	}

	NODISCARD INLINE size_t div_slices(size_t v) const
	{
		return m_num_slices > 1 ? static_cast<size_t>(v / m_num_slices_div) : v;
	}
};

// The slice graph of the slice group, closed to a given number of slice-group
// plies. A slice-group ply is not a quiet ply; see slice_plies_in_lookahead.
//
// Conservative by construction: blockers from other groups and full-position
// legality are ignored, so a pair may be listed as adjacent when no legal
// position realizes the move. It is never an under-approximation, which is
// what pinning correctness needs.
//
// A move into the group's secondary mirror half canonicalizes by reflecting
// the WHOLE board, so the recorded target is the canonical (primary-half)
// slice after that reflection -- what quiet_index() computes.
struct Slice_Reach
{
	Slice_Reach() = default;

	// `max_plies` is stated by the caller rather than defaulted: every level
	// costs time and memory.
	Slice_Reach(const Piece_Group& slice_group, size_t max_plies);

	NODISCARD size_t num_slices() const { return m_num_slices; }

	// Closed reach after at most `plies` slice-group moves, sorted ascending,
	// always containing `slice` itself.
	NODISCARD Const_Span<int32_t> closed(size_t plies, size_t slice) const
	{
		ASSERT(plies >= 1 && plies <= m_levels.size());
		const Level& level = m_levels[plies - 1];
		ASSERT(slice + 1 < level.off.size());
		return Const_Span<int32_t>(
			level.data.data() + level.off[slice],
			level.off[slice + 1] - level.off[slice]);
	}

	NODISCARD size_t max_closed(size_t plies) const
	{
		ASSERT(plies >= 1 && plies <= m_levels.size());
		return m_levels[plies - 1].max_size;
	}

private:
	struct Level
	{
		std::vector<int32_t>  data;   // CSR values, sorted per slice
		std::vector<uint32_t> off;    // CSR offsets, size num_slices + 1
		size_t max_size = 1;
	};

	size_t m_num_slices = 1;
	std::vector<Level> m_levels;
};
