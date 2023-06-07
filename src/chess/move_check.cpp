#include "position.h"
#include "move.h"
#include "attack.h"
#include "bitboard.h"
#include "chess.h"

#include "util/defines.h"
#include "util/param.h"
#include "util/math.h"
#include "util/lazy.h"

bool Position::is_in_check(Color me) const
{
	const Color opp = color_opp(me);
	const Square k_pos = king_square(me);

	// For attacks other than file attacks it's enough to check only one half of the board
	// because the general cannot leave his palace.

	if (   (square_rank_bb(k_pos)[me] & m_pieces[piece_make(opp, ROOK)][me])
		&& (rook_rank_attack_bb(k_pos, m_occupied)[me] & m_pieces[piece_make(opp, ROOK)][me]))
		return true;

	if (   (square_rank_bb(k_pos)[me] & m_pieces[piece_make(opp, CANNON)][me])
		&& (cannon_rank_attack_bb(k_pos, m_occupied)[me] & m_pieces[piece_make(opp, CANNON)][me]))
		return true;

	if (   (knight_att_no_mask(k_pos)[me] & m_pieces[piece_make(opp, KNIGHT)][me])
		&& (knight_attacked_bb(k_pos, m_occupied)[me] & m_pieces[piece_make(opp, KNIGHT)][me]))
		return true;

	if (pawn_attacked_bb(k_pos, opp)[me] & m_pieces[piece_make(opp, PAWN)][me])
		return true;

	if (   (square_file_bb(k_pos) & m_pieces[piece_make(opp, ROOK)])
		&& (rook_file_attack_bb(k_pos, m_occupied) & m_pieces[piece_make(opp, ROOK)]))
		return true;

	if (   (square_file_bb(k_pos) & m_pieces[piece_make(opp, CANNON)])
		&& (cannon_file_attack_bb(k_pos, m_occupied) & m_pieces[piece_make(opp, CANNON)]))
		return true;

	return false;
}

bool Position::is_legal() const
{
	const Square me_king = king_square(m_turn);
	const Square opp_king = king_square(color_opp(m_turn));
	auto any_doubled_pawn_half = [this]() {
		for (const Color color : { WHITE, BLACK })
		{
			Bitboard_Half piecebit = m_pieces[piece_make(color, PAWN)][color];
			while (piecebit)
			{
				const Square from = pop_first_square(piecebit, color);
				if (pawn_attack_bb(from, color)[color] & m_pieces[piece_make(color, PAWN)][color])
				{
					return true;
				}
			}
		}
		return false;
	};
	return    (!sq_equal_file(me_king, opp_king) || (file_between_bb(me_king, opp_king) & m_occupied))
		   && (!is_in_check(color_opp(m_turn))) && !any_doubled_pawn_half();
}

bool Position::is_mate(bool check, bool quiet) const
{
	ASSERT(is_legal());

	auto any_legal_move_half = [this, check](Square from, Bitboard_Half movebit, Color color) {
		while (movebit)
		{
			const Square to = pop_first_square(movebit, color);
			const Move move = Move(from, to);
			if (is_pseudo_legal_move_legal(move, check))
				return true;
		}

		return false;
	};

	auto any_legal_move_full = [&any_legal_move_half](Square from, Bitboard movesbb) {
		return any_legal_move_half(from, movesbb[WHITE], WHITE)
			|| any_legal_move_half(from, movesbb[BLACK], BLACK);
	};

	auto any_legal_move_half_pawn = [this, check](int from_offset, Bitboard_Half movebit, Color color) {
		while (movebit)
		{
			const Square to = pop_first_square(movebit, color);
			const Move move = Move(to + from_offset, to);
			if (is_pseudo_legal_move_legal(move, check))
				return true;
		}

		return false;
	};

	auto any_legal_move_full_pawn = [&any_legal_move_half_pawn](int from_offset, Bitboard movesbb) {
		return any_legal_move_half_pawn(from_offset, movesbb[WHITE], WHITE)
			|| any_legal_move_half_pawn(from_offset, movesbb[BLACK], BLACK);
	};

	const Color me = m_turn;
	const Color opp = color_opp(me);
	const Bitboard target = ~(quiet ? m_occupied : occupied(me)) & Bitboard::make_board_mask();

	{
		const Square from = king_square(me);
		const Bitboard_Half movebit = king_attack_bb(from)[me] & target[me];
		if (any_legal_move_half(from, movebit, me))
			return false;
	}

	Bitboard piecebb = m_pieces[piece_make(me, ADVISOR)];
	while (piecebb)
	{
		const Square from = piecebb.pop_first_square();
		const Bitboard_Half movebit = advisor_attack_bb(from)[me] & target[me];
		if (any_legal_move_half(from, movebit, me))
			return false;
	}

	piecebb = m_pieces[piece_make(me, BISHOP)];
	while (piecebb)
	{
		const Square from = piecebb.pop_first_square();
		const Bitboard_Half movebit = bishop_attack_bb(from, m_occupied)[me] & target[me];
		if (any_legal_move_half(from, movebit, me))
			return false;
	}

	piecebb = m_pieces[piece_make(me, KNIGHT)];
	while (piecebb)
	{
		const Square from = piecebb.pop_first_square();
		const Bitboard movesbb = knight_attack_bb(from, m_occupied) & target;
		if (any_legal_move_full(from, movesbb))
			return false;
	}
	piecebb = m_pieces[piece_make(me, ROOK)];
	while (piecebb)
	{
		const Square from = piecebb.pop_first_square();
		const Bitboard movesbb = rook_attack_bb(from, m_occupied) & target;
		if (any_legal_move_full(from, movesbb))
			return false;
	}
	piecebb = m_pieces[piece_make(me, CANNON)];
	while (piecebb)
	{
		const Square from = piecebb.pop_first_square();
		const Bitboard movesbb = 
			(rook_attack_bb(from, m_occupied) & ~m_occupied) 
			| (quiet ? Bitboard::make_empty() : (cannon_attack_bb(from, m_occupied) & occupied(opp)));
		if (any_legal_move_full(from, movesbb))
			return false;
	}

	if (m_pieces[piece_make(me, PAWN)])
	{
		const Bitboard piecebb = m_pieces[piece_make(me, PAWN)];
		if (me == WHITE)
		{
			const Bitboard movesbb = (piecebb << 9) & target;
			if (any_legal_move_full_pawn(-9, movesbb))
				return false;
		}
		else
		{
			const Bitboard movesbb = (piecebb >> 9) & target;
			if (any_legal_move_full_pawn(9, movesbb))
				return false;
		}

		if (piecebb[opp])
		{
			{
				const Bitboard_Half movebit = (piecebb[opp] << 1) & ~file_bb(FILE_A)[opp] & target[opp];
				if (any_legal_move_half_pawn(-1, movebit, opp))
					return false;
			}

			{
				const Bitboard_Half movebit = (piecebb[opp] >> 1) & ~file_bb(FILE_I)[opp] & target[opp];
				if (any_legal_move_half_pawn(1, movebit, opp))
					return false;
			}
		}
	}

	return true;
}

bool Position::is_quiet_mate(bool in_check) const
{
	return is_mate(in_check, true);
}

Move_List Position::gen_legal_capture_evasions()
{
	ASSERT(!is_in_check());

	do_null_move();

	Move_List evt_list;
	auto legal_quiet_list = Lazy_Cached_Value([this]() { 
		Move_List quiets = gen_pseudo_legal_quiets(); 
		quiets.remove_if([this](Move move) { return !is_pseudo_legal_move_legal(move); });
		return quiets;
	});

	for (const Move move : gen_pseudo_legal_captures())
	{
		if (!is_move_attack(move, Move_Legality_Lower_Bound::PSEUDO_LEGAL))
			continue;

		undo_null_move();

		add_evasion_moves(move, inout_param(evt_list), inout_param(*legal_quiet_list));

		do_null_move();
	}

	undo_null_move();

	return evt_list;
}

bool Position::is_move_evasion(Move evd_move, Optional_Out_Param<Bitboard> bb)
{
	if (bb)
		bb->clear();

	const Square evd_from = evd_move.from();

	ASSERT(m_squares[evd_from] != PIECE_NONE);
	ASSERT(piece_color(m_squares[evd_from]) == m_turn);

	do_null_move();

	bool result = false;
	bool added_evd_target = false;
	for (const Move move : gen_pseudo_legal_captures())
	{
		if (!is_move_attack(move, Move_Legality_Lower_Bound::PSEUDO_LEGAL))
			continue;

		if (move.to() == evd_from)
		{
			if (!added_evd_target && moves_chain(move, evd_move))
			{
				result = true;
				if (bb)
				{
					added_evd_target = true;
					*bb |= evd_move.to();
				}
			}
		}
		else
		{
			undo_null_move();
			do_quiet_move(evd_move);
			if (!is_move_attack(move, Move_Legality_Lower_Bound::NONE))
			{
				result = true;
				if (bb)
					*bb |= move.to();
			}
			undo_quiet_move(evd_move);
			do_null_move();
		}

		if (!bb && result)
			break;
	}

	undo_null_move();

	return result;
}

bool Position::is_move_evasion(Move evd_move)
{
	return is_move_evasion(evd_move, {});
}

bool Position::is_move_attack(Move move, Move_Legality_Lower_Bound legality)
{
	const Square from = move.from();
	const Square to = move.to();

	ASSERT(m_squares[from] != PIECE_NONE);

	const Piece_Type piece = piece_type(m_squares[from]);

	ASSERT(piece_color(m_squares[from]) == m_turn);

	if (piece == KING || piece == PAWN)
		return false;

	if (legality <= Move_Legality_Lower_Bound::NONE && !is_move_pseudo_legal(move))
		return false;

	if (legality <= Move_Legality_Lower_Bound::PSEUDO_LEGAL && !is_pseudo_legal_move_legal(move))
		return false;

/*
	Giving checks are not considered evasions of existing attacks, therefore if this is called when in check, we ignore the legality of NOT dealing with existing checks.

	if (legality <= Move_Legality_Lower_Bound::PSEUDO_LEGAL && (is_in_check() ? !is_pseudo_legal_move_legal_in_check(move) : !is_pseudo_legal_move_legal(move)))
		return false;
*/

	ASSERT(is_move_pseudo_legal(move));
	ASSERT(is_pseudo_legal_move_legal(move));

	const Piece cap = m_squares[to];
	ASSERT(cap != PIECE_NONE);
	ASSERT(cap != WHITE_KING);
	ASSERT(cap != BLACK_KING);
	ASSERT(piece_color(cap) == color_opp(m_turn));
	const Color color = color_opp(m_turn);

	auto is_square_attacked_after_move = [&]() {
		do_capture_move(move);
		const bool prt = is_square_attacked(to);
		undo_capture_move(move, cap);
		return prt;
	};

	auto would_be_legal_for_other_side = [&](Move move) {
		do_null_move();
		const bool legal = is_pseudo_legal_move_legal(move);
		undo_null_move();
		return legal;
	};

	switch (piece_type(cap))
	{
	case ROOK:
		if (piece == BISHOP || piece == ADVISOR)
			return !is_square_attacked_after_move();
		else if (piece == ROOK)
			return !would_be_legal_for_other_side(Move(to, from)) && !is_square_attacked_after_move();
		else
			return true;
	case KNIGHT:
		return (piece != KNIGHT || m_squares[knight_move_blocker(to, from)] || !would_be_legal_for_other_side(Move(to, from)))
			&& !is_square_attacked_after_move();
	case CANNON:
		return (piece != CANNON || !would_be_legal_for_other_side(Move(to, from)))
			&& !is_square_attacked_after_move();
	case PAWN:
		return (sq_color(to) != color) 
			&& !is_square_attacked_after_move();
	case ADVISOR:
	case BISHOP:
		return !is_square_attacked_after_move();
	default:
		ASSERT(false);
	}

	return false;
}

bool Position::is_square_attacked(Square to)
{
	const Color color = m_turn;

	auto is_legal = [this, check = is_in_check(color)](const Square from, const Square to) {
		return is_pseudo_legal_move_legal(Move(from, to), check);
	};

	if (   (square_bb(to)[color] & king_area_bb()[color])
		&& (king_attack_bb(to) & m_pieces[piece_make(color, KING)])
		&& is_legal(king_square(color), to))
		return true;

	if (   (m_pieces[piece_make(color, ADVISOR)])
		&& (square_bb(to)[color] & advisor_area_bb()[color]))
	{
		Bitboard piecesbb = advisor_attack_bb(to) & m_pieces[piece_make(color, ADVISOR)];
		while (piecesbb)
			if (is_legal(piecesbb.pop_first_square(), to))
				return true;
	}

	if (   (m_pieces[piece_make(color, BISHOP)])
		&& (square_bb(to)[color] & bishop_area_bb()[color]))
	{
		Bitboard piecesbb = bishop_attack_bb(to, m_occupied) & m_pieces[piece_make(color, BISHOP)];
		while (piecesbb)
			if (is_legal(piecesbb.pop_first_square(), to))
				return true;
	}

	if (m_pieces[piece_make(color, ROOK)])
	{
		Bitboard piecesbb = square_rank_bb(to) & m_pieces[piece_make(color, ROOK)];
		while (piecesbb)
		{
			const Square from = piecesbb.pop_first_square();
			if (   !(rank_between_bb(from, to) & m_occupied)
				&& is_legal(from, to))
				return true;
		}

		piecesbb = square_file_bb(to) & m_pieces[piece_make(color, ROOK)];
		while (piecesbb)
		{
			const Square from = piecesbb.pop_first_square();
			if (   !(file_between_bb(from, to) & m_occupied)
				&& is_legal(from, to))
				return true;
		}
	}

	if (m_pieces[piece_make(color, CANNON)])
	{
		Bitboard piecesbb = square_rank_bb(to) & m_pieces[piece_make(color, CANNON)];
		while (piecesbb)
		{
			const Square from = piecesbb.pop_first_square();
			if (   (rank_between_bb(from, to) & m_occupied).has_only_one_set_bit()
				&& is_legal(from, to))
				return true;
		}

		piecesbb = square_file_bb(to) & m_pieces[piece_make(color, CANNON)];
		while (piecesbb)
		{
			const Square from = piecesbb.pop_first_square();
			if (   (file_between_bb(from, to) & m_occupied).has_only_one_set_bit()
				&& is_legal(from, to))
				return true;
		}
	}

	if (m_pieces[piece_make(color, KNIGHT)])
	{
		Bitboard piecesbb = knight_attacked_bb(to, m_occupied) & m_pieces[piece_make(color, KNIGHT)];
		while (piecesbb)
			if (is_legal(piecesbb.pop_first_square(), to))
				return true;
	}

	if (m_pieces[piece_make(color, PAWN)])
	{
		Bitboard piecesbb = pawn_attacked_bb(to, color) & m_pieces[piece_make(color, PAWN)];
		while (piecesbb)
			if (is_legal(piecesbb.pop_first_square(), to))
				return true;
	}

	return false;
}

void Position::add_evasion_moves(Move att_move, In_Out_Param<Move_List> evt_list, In_Out_Param<Move_List> legal_quiet_list)
{
	ASSERT(!is_in_check());

	const Square att_from = att_move.from();
	const Square att_to = att_move.to();

	for (size_t i = 0; i < legal_quiet_list->size();)
	{
		const Move move = (*legal_quiet_list)[i];

		ASSERT(is_pseudo_legal_move_legal(move));

		const Square from = move.from();
		const Square to = move.to();

		bool is_evt = false;

		if (from == att_to)
		{
			if (   (sq_equal_file(att_from, att_to) && sq_equal_file(from, to))
				|| (sq_equal_rank(att_from, att_to) && sq_equal_rank(from, to)))
			{
				do_quiet_move(move);
				is_evt = !is_move_attack(Move(att_from, to), Move_Legality_Lower_Bound::NONE);
				undo_quiet_move(move);
			}
			else
				is_evt = true;
		}
		else
		{
			do_quiet_move(move);
			is_evt = !is_move_attack(att_move, Move_Legality_Lower_Bound::NONE);
			undo_quiet_move(move);
		}

		if (is_evt)
		{
			evt_list->add(move);
			legal_quiet_list->swap_with_last_and_pop(i);
		}
		else
		{
			++i;
		}
	}
}

bool Position::has_attack_after_quiet_move(Move move, const Bitboard& target, Out_Param<Bitboard> bb)
{
	ASSERT(piece_color(m_squares[move.from()]) == m_turn);
	*bb = attack_bb_after_quiet_move(move);
	return target & *bb;
}

bool Position::has_attack_after_quiet_move(Move move, const Bitboard& target)
{
	Bitboard bb = Bitboard::make_empty();
	return has_attack_after_quiet_move(move, target, out_param(bb));
}

bool Position::has_attack_after_quiet_move(Move move)
{
	Bitboard bb = Bitboard::make_empty();
	return has_attack_after_quiet_move(move, out_param(bb));
}

bool Position::has_attack_after_quiet_move(Move move, Out_Param<Bitboard> bb)
{
	return has_attack_after_quiet_move(move, Bitboard::make_board_mask(), bb);
}

bool Position::always_has_attack_after_quiet_moves(const Move_List& list, const Bitboard& target)
{
	Bitboard capbb = target;
	for (const Move move : list)
	{
		ASSERT(piece_color(m_squares[move.from()]) == m_turn);
		capbb &= attack_bb_after_quiet_move(move);
		if (!capbb)
			return false;
	}

	return true;
}

bool Position::always_has_attack_after_quiet_moves(const Move_List& list)
{
	return always_has_attack_after_quiet_moves(list, Bitboard::make_board_mask());
}

bool Position::moves_chain(Move pre_move, Move move) const
{
	const Square pre_from = pre_move.from();
	const Square pre_to = pre_move.to();
	
	const Square from = move.from();
	const Square to = move.to();

	ASSERT(pre_to == from);

	const bool eq_rank = sq_equal_rank(pre_from, pre_to);
	const bool eq_file = sq_equal_file(pre_from, pre_to);
	return (   (eq_rank && !sq_equal_rank(from, to))
			|| (eq_file && !sq_equal_file(from, to))
			|| (!eq_rank && !eq_file));
}

Bitboard Position::attack_bb_after_quiet_move(Move pre_move)
{
	Bitboard bb = Bitboard::make_empty();

	do_quiet_move(pre_move);
	do_null_move();

	for (const Move move : gen_pseudo_legal_captures())
	{
		if (!is_move_attack(move, Move_Legality_Lower_Bound::PSEUDO_LEGAL))
			continue;

		if (pre_move.to() == move.from())
		{
			if (moves_chain(pre_move, move))
				bb |= move.to();
		}
		else
		{
			if (moves_connect(pre_move, move))
				bb |= move.to();
			else
			{
				// 不同子移动
				undo_null_move();
				undo_quiet_move(pre_move);

				if (!is_move_attack(move, Move_Legality_Lower_Bound::NONE))
					bb |= move.to();

				do_quiet_move(pre_move);
				do_null_move();
			}
		}
	}

	undo_null_move();
	undo_quiet_move(pre_move);

	return bb;
}

bool Position::moves_connect(Move first, Move second, bool is_cap) const
{
	const Square a_from = first.from();
	const Square a_to = first.to();
	const Square b_from = second.from();
	const Square b_to = second.to();

	auto generic_check = [&](auto&& eq_func) {
		if (!eq_func(b_from, b_to))
			return false;

		if (eq_func(a_from, a_to))
			return eq_func(b_from, a_from) 
			    && is_mid(a_from, b_from, b_to) 
			    && (is_cap || !is_mid(a_to, b_from, b_to));
		else
			return (eq_func(b_from, a_from) && is_mid(a_from, b_from, b_to)) 
			    || (eq_func(b_from, a_to) && is_mid(a_to, b_from, b_to));
	};

	if (   generic_check([](Square a, Square b) { return sq_equal_file(a, b); }) 
		|| generic_check([](Square a, Square b) { return sq_equal_rank(a, b); }))
		return true;

	if (   (knight_att_no_mask(b_from) & b_to)
		&& knight_move_blocker(b_from, b_to) == a_from)
		return true;

	if (   (is_bishop_pos(b_from) && is_bishop_pos(b_to))
		&& (bishop_att_no_mask(b_from) & b_to)
		&& sq_mid(b_from, b_to) == a_from)
		return true;

	return false;
}

bool Position::is_move_check(Move move) const
{
	const Square from = move.from();
	const Square to = move.to();
	const Piece piece = m_squares[from];
	const Piece_Type pt = piece_type(piece);
	const Piece cap = m_squares[to];
	const Color me = piece_color_on(from);
	const Color opp = color_opp(me);
	const Square k_pos = king_square(opp);

	auto is_attacked_by = [&](auto&& attack_bb_func, Piece_Type piece_type, bool include_to = false) {
		Bitboard targets = piece_bb(me, piece_type);
		if (include_to)
			targets |= to;

		return attack_bb_func(k_pos, m_occupied ^ from) & targets;
	};

	auto is_attacked_by_cannon_on_file = [&](bool cannon_move = false) {
		return (cannon_move || (square_file_bb(k_pos) & piece_bb(me, CANNON)))
			&& is_attacked_by(cannon_file_attack_bb, CANNON, cannon_move);
	};

	auto is_attacked_by_cannon_on_rank = [&](bool cannon_move = false) {
		return (cannon_move || (square_rank_bb(k_pos) & piece_bb(me, CANNON)))
			&& is_attacked_by(cannon_rank_attack_bb, CANNON, cannon_move);
	};

	auto is_attacked_by_rook_on_file = [&]() {
		return (square_file_bb(k_pos) & piece_bb(me, ROOK))
			&& is_attacked_by(rook_file_attack_bb, ROOK, false);
	};

	auto is_attacked_by_rook_on_rank = [&]() {
		return (square_rank_bb(k_pos) & piece_bb(me, ROOK))
			&& is_attacked_by(rook_rank_attack_bb, ROOK, false);
	};

	auto is_attacked_by_pawn = [&]() {
		return pawn_attack_bb(to, me) & k_pos;
	};

	// King was a second blocker and therefore cannon can now jump.
	// Kings cannot oppose each other directly so this is the only possible "discovered" check.
	if (pt == KING)
		return   sq_equal_file(from, k_pos) 
			  && sq_equal_rank(from, to) 
			  && is_attacked_by_cannon_on_file();

	if (sq_equal_rank(from, k_pos))
	{
		if (sq_equal_rank(from, to))
		{
			if (pt == PAWN && is_attacked_by_pawn())
				return true;

			if (cap == PIECE_NONE)
				return false;

			if (pt == ROOK)
				return !(rank_between_bb(to, k_pos) & m_occupied);

			if (pt == PAWN)
				return is_attacked_by_cannon_on_rank();

			return (pt == CANNON)
				&& (   is_attacked_by_cannon_on_rank(true)
					|| is_attacked_by_rook_on_rank());
		}

		if (   is_attacked_by_cannon_on_rank()
			|| is_attacked_by_rook_on_rank())
			return true;
	}

	if (sq_equal_file(from, k_pos))
	{
		if (sq_equal_file(from, to))
		{
			if (pt == PAWN && is_attacked_by_pawn())
				return true;
				
			if (cap == PIECE_NONE)
				return false;
				
			if (pt == ROOK)
				return !(file_between_bb(to, k_pos) & m_occupied);

			if (pt == PAWN)
				return is_attacked_by_cannon_on_file();
			
			return (pt == CANNON)
				&& (   is_attacked_by_cannon_on_file(true)
					|| is_attacked_by_rook_on_file());
		}

		if (   is_attacked_by_cannon_on_file()
			|| is_attacked_by_rook_on_file())
			return true;
	}

	// 马腿离将
	if (   may_block_knight_for_king(from, k_pos)
		&& piece_bb(me, KNIGHT)[opp]
		&& (knight_att_no_mask(k_pos)[opp] & piece_bb(me, KNIGHT)[opp])
		&& is_attacked_by(knight_attacked_bb, KNIGHT))
		return true;

	if (sq_equal_rank(to, k_pos))
	{
		if (pt == ROOK)
			return !(rank_between_bb(to, k_pos) & m_occupied);

		if (pt == PAWN && is_attacked_by_pawn())
			return true;

		// Jumps over the blocker
		if (pt == CANNON && (rank_between_bb(to, k_pos) & m_occupied).has_only_one_set_bit())
			return true;

		// Added a blocker to jump over.
		return cap == 0 && is_attacked_by_cannon_on_rank();
	}

	if (sq_equal_file(to, k_pos))
	{
		if (pt == ROOK)
			return !(file_between_bb(to, k_pos) & m_occupied);

		if (pt == PAWN && is_attacked_by_pawn())
			return true;
		
		// Jumps over the blocker
		if (pt == CANNON && (file_between_bb(to, k_pos) & m_occupied).has_only_one_set_bit())
			return true;

		// Added a blocker to jump over.
		return cap == 0 && is_attacked_by_cannon_on_file();
	}

	return pt == KNIGHT
		&& (knight_att_no_mask(k_pos) & to)
		&& m_squares[knight_move_blocker(to, k_pos)] == 0;
}
