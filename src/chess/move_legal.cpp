#include "chess.h"
#include "attack.h"
#include "move.h"
#include "position.h"
#include "bitboard.h"

#include "util/defines.h"

bool Position::is_move_pseudo_legal(Move move) const
{
	const Square from = move.from();
	const Square to = move.to();

	if (!sq_is_ok(from) || !sq_is_ok(to) || from == to)
		return false;
	
	if (m_squares[from] == 0 || piece_color(m_squares[from]) != m_turn)
		return false;
	
	if (m_squares[to] && piece_color(m_squares[to]) != color_opp(m_turn))
		return false;

	if (m_squares[to] == WHITE_KING || m_squares[to] == BLACK_KING)
		return false;
	
	switch (piece_type_on(from))
	{
	case ROOK:
		return (sq_equal_rank(from, to) && !(rank_between_bb(from, to) & m_occupied))
			|| (sq_equal_file(from, to) && !(file_between_bb(from, to) & m_occupied));

	case CANNON:
	{
		const bool eq_rank = sq_equal_rank(from, to);
		const bool eq_file = sq_equal_file(from, to);
		if (!eq_rank && !eq_file)
			return false;

		const Bitboard pin_bb(
			eq_rank
			? rank_between_bb(from, to) & m_occupied
			: file_between_bb(from, to) & m_occupied
		);

		return 
			m_squares[to] == PIECE_NONE
			? pin_bb.empty()
			: pin_bb.has_only_one_set_bit();
	}

	case KNIGHT:
		return (knight_att_no_mask(from) & to)
			&& m_squares[knight_move_blocker(from, to)] == 0;

	case BISHOP:
		return is_bishop_pos(from, piece_color_on(from))
			&& (bishop_att_no_mask(from) & to)
			&& m_squares[sq_mid(from, to)] == 0;

	case ADVISOR:
		return is_advisor_pos(from, piece_color_on(from))
			&& (advisor_attack_bb(from) & to);

	case PAWN:
		return pawn_attack_bb(from, m_turn) & to;

	case KING:
		return is_king_pos(from, piece_color_on(from))
			&& (king_attack_bb(from) & to);

	default:
		return false;
	}
}

bool Position::is_pseudo_legal_move_legal(Move move) const
{
	const Square from = move.from();
	const Square to = move.to();
	const Color me = m_turn;
	const Color opp = color_opp(me);

	ASSERT(piece_color_on(from) == me);

	if (piece_type_on(from) == KING)
	{
		if (sq_equal_file(from, to))
		{
			const Bitboard_Half rank_mask = square_rank_bb(to)[me];
			const Bitboard rooks = piece_bb(opp, ROOK);
			if (   (rank_mask & rooks[me])
				&& (rook_rank_attack_bb(to, m_occupied)[me] & rooks[me]))
				return false;

			const Bitboard cannons = piece_bb(opp, CANNON);
			if (   (rank_mask & cannons[me])
				&& (cannon_rank_attack_bb(to, m_occupied)[me] & cannons[me]))
				return false;

			const Bitboard_Half knights = piece_bb(opp, KNIGHT)[me];
			if (   (knight_att_no_mask(to)[me] & knights)
				&& (knight_attacked_bb(to, m_occupied)[me] & knights))
				return false;

			const Bitboard_Half pawns = piece_bb(opp, PAWN)[me];
			if (   pawns
				&& (pawn_attacked_bb(to, opp)[me] & pawns))
				return false;

			if (m_squares[to] != PIECE_NONE)
			{
				const Bitboard file_mask = square_file_bb(to);
				const Bitboard kings = piece_bb(opp, KING);
				if (   (file_mask & (rooks | kings))
					&& (rook_file_attack_bb(to, m_occupied) & (rooks | kings)))
					return false;
				
				if (   (file_mask & cannons)
					&& (cannon_file_attack_bb(to, m_occupied ^ from) & cannons))
					return false;
			}
		}
		else
		{
			const Bitboard file_mask = square_file_bb(to);
			const Bitboard rooks = piece_bb(opp, ROOK);
			const Bitboard kings = piece_bb(opp, KING);
			if (   (file_mask & (rooks | kings))
				&& (rook_file_attack_bb(to, m_occupied) & (rooks | kings)))
				return false;
			
			const Bitboard cannons = piece_bb(opp, CANNON);
			if (   (file_mask & cannons)
				&& (cannon_file_attack_bb(to, m_occupied) & cannons))
				return false;
			
			const Bitboard_Half knights = piece_bb(opp, KNIGHT)[me];
			if (   (knight_att_no_mask(to)[me] & knights)
				&& (knight_attacked_bb(to, m_occupied)[me] & knights))
				return false;
			
			const Bitboard_Half pawns = piece_bb(opp, PAWN)[me];
			if (   pawns
				&& (pawn_attacked_bb(to, opp)[me] & pawns))
				return false;

			if (m_squares[to] != PIECE_NONE)
			{
				if (   (square_rank_bb(to)[me] & rooks[me])
					&& (rook_rank_attack_bb(to, m_occupied)[me] & rooks[me]))
					return false;
				
				if (   (square_rank_bb(to)[me] & cannons[me])
					&& (cannon_rank_attack_bb(to, m_occupied ^ from)[me] & cannons[me]))
					return false;
			}
		}
	}
	else
	{
		const Square king_pos = king_square(me);

		auto any_connect = [&](auto bits) {
			while (bits)
			{
				Square s;
				if constexpr (std::is_same_v<decltype(bits), Bitboard_Half>)
					s = pop_first_square(bits, me);
				else
					s = bits.pop_first_square();

				if (moves_connect(move, Move(s, king_pos), true))
					return true;
			}
			return false;
		};

		if (sq_equal_rank(from, king_pos))
		{
			if (sq_equal_rank(from, to))
			{
				if (m_squares[to] == PIECE_NONE)
					return true;

				const Bitboard_Half rank_mask = square_rank_bb(king_pos)[me];
				const Bitboard_Half cannons = piece_bb(opp, CANNON)[me];
				if (rank_mask & cannons)
				{
					const Bitboard_Half bits = cannon_rank_attack_bb(king_pos, m_occupied ^ from)[me]
						& cannons & ~(square_bb(to)[me]);

					if (any_connect(bits))
						return false;
				}

				const Bitboard_Half rooks = piece_bb(opp, ROOK)[me];
				if (piece_type_on(from) == CANNON && (rank_mask & rooks))
				{
					const Bitboard_Half bits = rook_rank_attack_bb(king_pos, m_occupied ^ from)[me]
						& rooks;

					if (any_connect(bits))
						return false;
				}
			}
			else
			{
				const Bitboard_Half rank_mask = square_rank_bb(king_pos)[me];
				const Bitboard_Half rooks = piece_bb(opp, ROOK)[me];
				if (rank_mask & rooks)
				{
					const Bitboard_Half bits = rook_rank_attack_bb(king_pos, m_occupied ^ from)[me]
						& rooks;

					if (any_connect(bits))
						return false;
				}

				const Bitboard_Half cannons = piece_bb(opp, CANNON)[me];
				if (rank_mask & cannons)
				{
					const Bitboard_Half bits = cannon_rank_attack_bb(king_pos, m_occupied ^ from)[me]
						& cannons;

					if (any_connect(bits))
						return false;
				}
			}
		}
		else if (sq_equal_file(from, king_pos))
		{
			if (sq_equal_file(from, to))
			{
				if (piece_on(to) == PIECE_NONE)
					return true;

				const Bitboard file_mask = square_file_bb(king_pos);
				const Bitboard cannons = piece_bb(opp, CANNON);
				if (file_mask & cannons)
				{
					const Bitboard bb = cannon_file_attack_bb(king_pos, m_occupied ^ from)
						& cannons & ~to;

					if (any_connect(bb))
						return false;
				}

				const Bitboard rooks = piece_bb(opp, ROOK);
				const Bitboard kings = piece_bb(opp, KING);
				if (piece_type_on(from) == CANNON && (file_mask & (rooks | kings)))
				{
					const Bitboard bb = rook_file_attack_bb(king_pos, m_occupied ^ from)
						& (rooks | kings);

					if (any_connect(bb))
						return false;
				}
			}
			else
			{
				const Bitboard file_mask = square_file_bb(king_pos);
				const Bitboard rooks = piece_bb(opp, ROOK);
				const Bitboard kings = piece_bb(opp, KING);
				if (file_mask & (rooks | kings))
				{
					const Bitboard bb = rook_file_attack_bb(king_pos, m_occupied ^ from)
						& (rooks | kings);

					if (any_connect(bb))
						return false;
				}

				const Bitboard cannons = piece_bb(opp, CANNON);
				if (file_mask & cannons)
				{
					const Bitboard bb = cannon_file_attack_bb(king_pos, m_occupied ^ from)
						& cannons;

					if (any_connect(bb))
						return false;
				}
			}
		}
		else if (may_block_knight_for_king(from, king_pos))
		{
			Bitboard_Half bits = knight_att_no_mask(king_pos)[me] & piece_bb(opp, KNIGHT)[me];
			while (bits)
			{
				const Square sq = pop_first_square(bits, me);
				if (from == knight_move_blocker(sq, king_pos) && to != sq)
					return false;
			}
		}

		if (piece_on(to) == PIECE_NONE)
		{
			if (sq_equal_rank(to, king_pos))
			{
				const Bitboard_Half bits = cannon_rank_attack_bb(king_pos, m_occupied ^ to)[me]
					& piece_bb(opp, CANNON)[me];

				if (any_connect(bits))
					return false;
			}
			else if (sq_equal_file(to, king_pos))
			{
				const Bitboard bb = cannon_file_attack_bb(king_pos, m_occupied ^ to) 
					& piece_bb(opp, CANNON);

				if (any_connect(bb))
					return false;
			}
		}
	}

	return true;
}

bool Position::is_pseudo_legal_move_legal_in_check(Move move) const
{
	ASSERT(is_in_check());

	const Square from = move.from();
	const Square to = move.to();
	const Color me = piece_color_on(from);
	const Color opp = color_opp(me);

	const bool is_king_move = piece_type_on(from) == KING;

	const Square k_pos = is_king_move ? to : king_square(me);

	const Bitboard remaining_mask =
		is_king_move || piece_on(to) == PIECE_NONE
		? Bitboard::make_board_mask()
		: ~to;

	const Bitboard block =
		is_king_move || piece_on(to) != PIECE_NONE
		? m_occupied ^ from
		: m_occupied ^ from ^ to;

	const Bitboard rank_mask = square_rank_bb(k_pos);
	const Bitboard file_mask = square_file_bb(k_pos);

	const Bitboard rooks = piece_bb(opp, ROOK);
	if (rooks)
	{
		if (   (rank_mask[me] & rooks[me])
			&& (rook_rank_attack_bb(k_pos, block)[me] & rooks[me] & remaining_mask[me]))
			return false;
				
		if (   (file_mask & rooks)
			&& (rook_file_attack_bb(k_pos, block) & rooks & remaining_mask))
			return false;
	}

	const Bitboard cannons = piece_bb(opp, CANNON);
	if (cannons)
	{
		if (   (rank_mask[me] & cannons[me])
			&& (cannon_rank_attack_bb(k_pos, block)[me] & cannons[me] & remaining_mask[me]))
			return false;

		if (   (file_mask & cannons)
			&& (cannon_file_attack_bb(k_pos, block) & cannons & remaining_mask))
			return false;
	}

	const Bitboard_Half knights = piece_bb(opp, KNIGHT)[me];
	if (   knights
		&& (knight_att_no_mask(k_pos)[me] & knights)
		&& (knight_attacked_bb(k_pos, block)[me] & knights & remaining_mask[me]))
		return false;

	const Bitboard_Half pawns = piece_bb(opp, PAWN)[me];
	if (   pawns
		&& (pawn_attacked_bb(k_pos, opp)[me] & pawns & remaining_mask[me]))
		return false;

	const Square opp_king_pos = king_square(opp);
	if (   sq_equal_file(k_pos, opp_king_pos)
		&& !(file_between_bb(k_pos, opp_king_pos) & block))
		return false;

	return true;
}
