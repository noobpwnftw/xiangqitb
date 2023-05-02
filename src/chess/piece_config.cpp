#include "piece_config.h"

#include "chess.h"
#include "position.h"

#include "util/defines.h"

#include <array>
#include <algorithm>
#include <map>

void Piece_Config::sort_pieces(Span<Piece> pieces)
{
	size_t score[COLOR_NB] = { 0, 0 };
	for (const Piece p : pieces)
		score[piece_color(p)] += PIECE_STRENGTH_FOR_SIDE_ORDER[p];

	if (score[BLACK] > score[WHITE])
		for (Piece& p : pieces)
			p = piece_opp_color(p);

	std::sort(
		pieces.begin(),
		pieces.end(),
		[](Piece a, Piece b) {
			return PIECE_ORDER[a] < PIECE_ORDER[b];
		}
	);
}

void Piece_Config::add_sub_configs_to(Unique_Piece_Configs& pss) const
{
	for (size_t i = 0; i < num_pieces(); ++i)
	{
		if (!can_remove_piece(i))
			continue;

		pss.add_unique(with_removed_piece(i));
	}
}

void Piece_Config::add_closure_in_dependency_order_to(Unique_Piece_Configs& pss, bool assume_contains_closures) const
{
	// The assumption of pss having full closers allows us to skip some work, 
	// because adding the closure again would not change anything.
	if (assume_contains_closures && pss.contains(*this))
		return;

	for (size_t i = 0; i < num_pieces(); ++i)
	{
		if (!can_remove_piece(i))
			continue;

		const auto ps = with_removed_piece(i);
		ps.add_closure_in_dependency_order_to(pss, assume_contains_closures);
		pss.add_unique(ps);
	}
	pss.add_unique(*this);
}

Unique_Piece_Configs Piece_Config::sub_configs() const
{
	Unique_Piece_Configs sub;
	add_sub_configs_to(sub);
	return sub;
}

std::map<Piece, Piece_Config> Piece_Config::sub_configs_by_capture() const
{
	std::map<Piece, Piece_Config> res;

	for (size_t i = 0; i < num_pieces(); ++i)
	{
		if (!can_remove_piece(i))
			continue;

		res.try_emplace(m_pieces[i], with_removed_piece(i));
	}

	return res;
}

Unique_Piece_Configs Piece_Config::closure() const
{
	Unique_Piece_Configs sub;
	add_closure_in_dependency_order_to(sub);
	return sub;
}
