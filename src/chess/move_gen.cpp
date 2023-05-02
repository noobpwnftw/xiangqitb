#include "position.h"
#include "move.h"
#include "attack.h"
#include "search.h"
#include "chess.h"
#include "bitboard.h"

#include "util/defines.h"

enum struct Gen_Type {
	CAPTURE, PRE_QUIET, QUIET, NORMAL
};

template <Gen_Type TypeV>
Move_List gen(const Position& pos)
{
	static_assert(
		   TypeV == Gen_Type::CAPTURE
		|| TypeV == Gen_Type::PRE_QUIET
		|| TypeV == Gen_Type::QUIET
		|| TypeV == Gen_Type::NORMAL
	);

	Move_List list;

	const Color me = TypeV == Gen_Type::PRE_QUIET ? color_opp(pos.turn()) : pos.turn();
	const Color opp = color_opp(me);

	// Pseudo legal generation is sometimes called when opponent is in check.
	// We must take care not to generate king captures in such case.
	const Bitboard target =
		  TypeV == Gen_Type::CAPTURE ? pos.occupied(opp) ^ pos.pieces_bb(opp, KING)
		: (TypeV == Gen_Type::PRE_QUIET || TypeV == Gen_Type::QUIET) ? ~pos.occupied() & Bitboard::make_board_mask()
		: /* TypeV == GenType::Normal */ (~pos.occupied(me) & Bitboard::make_board_mask()) ^ pos.pieces_bb(opp, KING);

	auto add_from_movebit = [&](const Square from, Bitboard_Half movebit, const Color side)
	{
		while (movebit)
		{
			const Square to = pop_first_square(movebit, side);
			list.add(Move(from, to));
		}
	};

	auto add_from_pawn_movebit = [&](const int offset, Bitboard_Half movebit, const Color side)
	{
		while (movebit)
		{
			const Square to = pop_first_square(movebit, side);
			list.add(Move(to + offset, to));
		}
	};

	{
		const Square from = pos.king_square(me);
		const Bitboard_Half movebit = king_attack_bb(from)[me] & target[me];
		add_from_movebit(from, movebit, me);
	}

	{
		Bitboard piecebb = pos.pieces_bb(me, ADVISOR);
		while (piecebb)
		{
			const Square from = piecebb.pop_first_square();
			const Bitboard_Half movebit = advisor_attack_bb(from)[me] & target[me];
			add_from_movebit(from, movebit, me);
		}
	}

	{
		Bitboard piecebb = pos.pieces_bb(me, BISHOP);
		while (piecebb)
		{
			const Square from = piecebb.pop_first_square();
			const Bitboard_Half movebit = bishop_attack_bb(from, pos.occupied())[me] & target[me];
			add_from_movebit(from, movebit, me);
		}
	}

	{
		Bitboard piecebb = pos.pieces_bb(me, KNIGHT);
		while (piecebb)
		{
			const Square from = piecebb.pop_first_square();
			const Bitboard movesbb = 
				TypeV == Gen_Type::PRE_QUIET
				? knight_attacked_bb(from, pos.occupied()) & target
				: knight_attack_bb(from, pos.occupied()) & target;

			for (const Color side : { WHITE, BLACK })
				add_from_movebit(from, movesbb[side], side);
		}
	}

	{
		Bitboard piecebb = pos.pieces_bb(me, ROOK);
		while (piecebb)
		{
			const Square from = piecebb.pop_first_square();
			const Bitboard movesbb = rook_attack_bb(from, pos.occupied()) & target;
			
			for (const Color side : { WHITE, BLACK })
				add_from_movebit(from, movesbb[side], side);
		}
	}

	{
		Bitboard piecebb = pos.pieces_bb(me, CANNON);
		while (piecebb)
		{
			const Square from = piecebb.pop_first_square();
			Bitboard movesbb = Bitboard::make_empty();
			if constexpr (TypeV == Gen_Type::CAPTURE || TypeV == Gen_Type::NORMAL)
				movesbb |= (cannon_attack_bb(from, pos.occupied()) & pos.occupied(opp));
			if constexpr (TypeV == Gen_Type::PRE_QUIET || TypeV == Gen_Type::QUIET || TypeV == Gen_Type::NORMAL)
				movesbb |= (rook_attack_bb(from, pos.occupied()) & ~pos.occupied());

			for (const Color side : { WHITE, BLACK })
				add_from_movebit(from, movesbb[side], side);
		}
	}

	if (pos.pieces_bb(me, PAWN))
	{
		const Bitboard piecebb = pos.pieces_bb(me, PAWN);
		const Color c = (TypeV == Gen_Type::PRE_QUIET ? BLACK : WHITE);
		Bitboard movesbb = 
			me == c
			? (piecebb << 9) & target
			: (piecebb >> 9) & target;

		if constexpr (TypeV == Gen_Type::PRE_QUIET)
			movesbb &= pawn_area_bb(me);

		for (const Color side : { WHITE, BLACK })
			add_from_pawn_movebit((me == c ? -9 : 9), movesbb[side], side);

		if (piecebb[opp])
		{
			{
				Bitboard_Half movebit = (piecebb[opp] << 1) & ~file_bb(FILE_A)[opp] & target[opp];
				add_from_pawn_movebit(-1, movebit, opp);
			}
			{
				Bitboard_Half movebit = (piecebb[opp] >> 1) & ~file_bb(FILE_I)[opp] & target[opp];
				add_from_pawn_movebit(1, movebit, opp);
			}
		}
	}

	return list;
}

Move_List Position::gen_pseudo_legal_captures() const
{
	return gen<Gen_Type::CAPTURE>(*this);
}

Move_List Position::gen_pseudo_legal_pre_quiets() const
{
	return gen<Gen_Type::PRE_QUIET>(*this);
}

Move_List Position::gen_pseudo_legal_quiets() const
{
	return gen<Gen_Type::QUIET>(*this);
}

Move_List Position::gen_all_pseudo_legal_moves() const
{
	return gen<Gen_Type::NORMAL>(*this);
}
