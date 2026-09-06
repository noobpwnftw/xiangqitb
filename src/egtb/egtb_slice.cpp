#include "egtb_slice.h"

#include "egtb_gen.h"

#include "chess/attack.h"
#include "chess/bitboard.h"
#include "chess/move.h"

#include "util/math.h"

#include <algorithm>

// Every square the piece could move to from `from`, given only the occupancy
// its own group contributes. Still a superset of every real move set -- extra
// blockers only remove moves -- but tighter than ignoring occupancy entirely,
// since a bishop's eye or a knight's leg is often occupied by a group mate.
NODISCARD static Bitboard group_quiet_targets(Piece piece, Square from, const Bitboard& block)
{
	switch (piece_type(piece))
	{
	case KING:
		return king_attack_bb(from);

	case ADVISOR:
		return advisor_attack_bb(from);

	case BISHOP:
		return bishop_attack_bb(from, block);

	case KNIGHT:
		return knight_attack_bb(from, block);

	// A cannon's quiet moves are a rook's; its captures need a screen and are
	// a subset of the rank/file cross.
	case ROOK:
	case CANNON:
		return rook_attack_bb(from, block);

	case PAWN:
		return pawn_attack_bb(from, piece_color(piece));

	default:
		ASSUME(false);
		return Bitboard::make_empty();
	}
}

Slice_Layout::Slice_Layout(const Piece_Config_For_Gen& epsi) :
	m_num_slices(epsi.num_positions_in_group(epsi.compress_id())),
	m_low_weight(epsi.weight(epsi.compress_id())),
	m_num_high(epsi.num_positions() / (m_low_weight * m_num_slices)),
	m_slice_size(m_low_weight * m_num_high),
	m_num_entries(epsi.num_positions())
{
	ASSERT(m_low_weight * m_num_slices * m_num_high == m_num_entries);
	if (m_low_weight > 1)
		m_low_weight_div = Divider<uint64_t>(m_low_weight);
	if (m_num_slices > 1)
		m_num_slices_div = Divider<uint64_t>(m_num_slices);
}

Slice_Reach::Slice_Reach(const Piece_Group& group, size_t max_plies) :
	m_num_slices(group.compress_size())
{
	ASSERT(max_plies >= 1);

	const size_t num_slices = m_num_slices;

	m_levels.resize(max_plies);

	Level& first = m_levels[0];
	first.off.resize(num_slices + 1, 0);
	first.data.reserve(num_slices * 4);

	std::vector<int32_t> targets;
	for (size_t s = 0; s < num_slices; ++s)
	{
		targets.clear();
		targets.emplace_back(static_cast<int32_t>(s));

		const Piece_Group::Placement& placement =
			group.squares(static_cast<Piece_Group::Placement_Index>(s));

		Bitboard occupancy = Bitboard::make_empty();
		for (const Square sq : placement)
			occupancy |= square_bb(sq);

		for (size_t j = 0; j < group.size(); ++j)
		{
			const Piece piece = group.piece(j);
			const Square from = placement[j];

			Bitboard to_bb = group_quiet_targets(piece, from, occupancy);
			while (to_bb)
			{
				const Square to = to_bb.pop_first_square();

				// A destination outside the piece's square set cannot be
				// indexed, and one held by a group member is not a quiet
				// move. Both are exact, not merely conservative.
				if (possible_sq_index(piece, to) == -1)
					continue;
				if (std::find(placement.begin(), placement.end(), to) != placement.end())
					continue;

				const Piece_Group::Full_Placement_Index ix =
					group.compound_index_after_quiet_move(
						static_cast<Piece_Group::Placement_Index>(s), Move(from, to));

				// The secondary half mirrors the whole board, so the canonical
				// slice is the target's mirror image.
				const Piece_Group::Placement_Index canon =
					ix.base() < num_slices ? ix.base() : ix.mirr();
				ASSERT(canon < num_slices);

				targets.emplace_back(static_cast<int32_t>(canon));
			}
		}

		std::sort(targets.begin(), targets.end());
		targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

		first.data.insert(first.data.end(), targets.begin(), targets.end());
		first.off[s + 1] = static_cast<uint32_t>(first.data.size());

		update_max(first.max_size, targets.size());
	}

	// Wider closures: the union of the one-ply reaches of the previous level.
	std::vector<uint8_t> seen(num_slices, 0);
	std::vector<int32_t> touched;
	for (size_t plies = 2; plies <= max_plies; ++plies)
	{
		Level& level = m_levels[plies - 1];
		level.off.assign(num_slices + 1, 0);
		level.data.reserve(m_levels[plies - 2].data.size() * 2);

		for (size_t s = 0; s < num_slices; ++s)
		{
			touched.clear();
			for (const int32_t mid : closed(plies - 1, s))
				for (const int32_t dst : closed(1, static_cast<size_t>(mid)))
					if (!seen[dst])
					{
						seen[dst] = 1;
						touched.emplace_back(dst);
					}

			for (const int32_t dst : touched)
				seen[dst] = 0;

			std::sort(touched.begin(), touched.end());
			level.data.insert(level.data.end(), touched.begin(), touched.end());
			level.off[s + 1] = static_cast<uint32_t>(level.data.size());

			update_max(level.max_size, touched.size());
		}
	}
}
