#pragma once

#include "chess.h"
#include "bitboard.h"
#include "move.h"

#include "util/param.h"

#include <cstring>

// Represents a 64-bit key of the position (pieces, their locations, and turn).
// It is not guaranteed to be unique for every possible position.
struct Position_Key
{
private:
	static const uint64_t RANDOM_64[SQUARE_NB * PIECE_NB];

public:
	Position_Key(Color turn) :
		m_value(turn == WHITE ? RANDOM_64[0] : 0)
	{
	}

	void add(Piece piece, Square sq)
	{
		m_value ^= RANDOM_64[piece * SQUARE_NB + sq];
	}

	NODISCARD uint64_t value() const
	{
		return m_value;
	}

	NODISCARD friend bool operator==(Position_Key lhs, Position_Key rhs) noexcept
	{
		return lhs.m_value == rhs.m_value;
	}

	NODISCARD friend bool operator!=(Position_Key lhs, Position_Key rhs) noexcept
	{
		return lhs.m_value != rhs.m_value;
	}

	NODISCARD friend bool operator<(Position_Key lhs, Position_Key rhs) noexcept
	{
		return lhs.m_value < rhs.m_value;
	}

	NODISCARD friend bool operator>(Position_Key lhs, Position_Key rhs) noexcept
	{
		return lhs.m_value > rhs.m_value;
	}

	NODISCARD friend bool operator<=(Position_Key lhs, Position_Key rhs) noexcept
	{
		return lhs.m_value <= rhs.m_value;
	}

	NODISCARD friend bool operator>=(Position_Key lhs, Position_Key rhs) noexcept
	{
		return lhs.m_value >= rhs.m_value;
	}

private:
	uint64_t m_value;
};

// Represents a xiangqi position.
// Not initialized on creation for performance reasons.
struct Position
{
	friend struct Piece_Config_For_Gen;

	// Clears the board.
	void clear()
	{
		static_assert(PIECE_NONE == 0);
		static_assert(WHITE == 0);
		std::memset(this, 0, sizeof(Position));
	}

	// Returns the square of the king of the given color.
	NODISCARD INLINE Square king_square(Color color) const
	{
		if (color == WHITE)
			return peek_first_square(m_pieces[WHITE_KING][WHITE], WHITE);
		else
			return peek_first_square(m_pieces[BLACK_KING][BLACK], BLACK);
	}

	// Returns the occupation bitboard for pieces of given color.
	NODISCARD INLINE const Bitboard& occupied(Color color) const
	{
		return m_pieces[piece_occupy(color)];
	}

	// Returns the occupation bitboard of all pieces.
	NODISCARD INLINE const Bitboard& occupied() const
	{
		return m_occupied;
	}

	// Returns the color of the piece on a given square.
	// The square must be occupied.
	NODISCARD INLINE Color piece_color_on(Square sq) const
	{
		ASSERT(m_squares[sq] != PIECE_NONE);
		return piece_color(m_squares[sq]);
	}

	// Returns the type of the piece on a given square.
	// The square must be occupied.
	NODISCARD INLINE Piece_Type piece_type_on(Square sq) const
	{
		ASSERT(m_squares[sq] != PIECE_NONE);
		return piece_type(m_squares[sq]);
	}

	// Returns the occupation bitboard for a given piece (by its color and type).
	NODISCARD INLINE const Bitboard& piece_bb(Color color, Piece_Type type) const
	{
		return m_pieces[piece_make(color, type)];
	}

	// Returns a key of this position, and a key of this position with files mirrored.
	NODISCARD std::pair<Position_Key, Position_Key> pos_keys() const
	{
		Position_Key key(m_turn);
		Position_Key mir(m_turn);

		for (Square i = SQ_START; i < SQ_END; ++i)
		{
			const Piece piece = m_squares[i];
			if (piece)
			{
				key.add(piece, i);
				mir.add(piece, sq_file_mirror(i));
			}
		}

		return { key, mir };
	}

	// Returns a key of this position.
	NODISCARD Position_Key pos_key() const
	{
		Position_Key key(m_turn);

		for (Square i = SQ_START; i < SQ_END; ++i)
		{
			const Piece piece = m_squares[i];
			if (piece)
				key.add(piece, i);
		}

		return key;
	}

	// Places a piece on a given square.
	void put_piece(Piece piece, Square sq)
	{
		const Color color = piece_color(piece);
		m_squares[sq] = piece;
		m_piece_counts[piece] += 1;
		m_pieces[piece] |= sq;
		m_pieces[piece_occupy(color)] |= sq;
		m_occupied |= sq;
	}

	// Returns whether the king of given color is under attack.
	NODISCARD bool is_in_check(Color color) const;

	// Returns whether a given square is attacked by the opponent.
	NODISCARD bool is_square_attacked(Square to);

	// Returns whether the move is an attack for the purpose of rule loops.
	NODISCARD bool is_move_attack(Move move, Move_Legality_Lower_Bound legality);

	// Returns whether the move is an evasion for the purpose of rule loops.
	NODISCARD bool is_move_evasion(Move move);

	// Returns whether the move is an evasion for the purpose of rule loops.
	// 
	NODISCARD bool is_move_evasion(Move evd_move, Optional_Out_Param<Bitboard> bb);

	// Returns moves that evade a capture. For the purpose of rule loops.
	NODISCARD Move_List gen_legal_capture_evasions();

	// Helper for gen_legal_capture_evasions. All moves in legal_quiet_list must be legal.
	// IMPORTANT: Moves that are moved to evt_list are removed from legal_quiet_list.
	//            The order of moves in legal_quiet_list is not preserved.
	void add_evasion_moves(Move move, In_Out_Param<Move_List> evt_list, In_Out_Param<Move_List> legal_quiet_list);

	// TODO: figure out and describe. For the purpose of rule loops.
	NODISCARD bool has_attack_after_quiet_move(Move move);
	NODISCARD bool has_attack_after_quiet_move(Move move, Out_Param<Bitboard> bb);
	NODISCARD bool has_attack_after_quiet_move(Move move, const Bitboard& target);
	NODISCARD bool has_attack_after_quiet_move(Move move, const Bitboard& target, Out_Param<Bitboard> bb);

	// TODO: figure out and describe. For the purpose of rule loops.
	NODISCARD bool always_has_attack_after_quiet_moves(const Move_List& list);
	NODISCARD bool always_has_attack_after_quiet_moves(const Move_List& list, const Bitboard& target);

	// TODO: figure out and describe. For the purpose of rule loops.
	NODISCARD bool moves_connect(Move first, Move second, bool is_cap = false) const;
	NODISCARD bool moves_chain(Move pre_move, Move move) const;

	// TODO: figure out and describe. For the purpose of rule loops.
	NODISCARD Bitboard attack_bb_after_quiet_move(Move pre_move);

	// Returns whether the side to move is in check.
	NODISCARD INLINE bool is_in_check() const
	{
		return is_in_check(m_turn);
	}

	// Returns whether the side to move is checkmated.
	// in_check must specify whether the side to move is in check.
	// if quiet == true then only quiet moves are considered as possible.
	NODISCARD bool is_mate(bool in_check, bool quiet = false) const;

	// Returns whether the position is legal.
	// The position is illegal if the kings oppose each other 
	// or the opponent's king is in check.
	NODISCARD bool is_legal() const;

	// Returns whether side to move is checkmated if only quiet moves are considered.
	NODISCARD bool is_quiet_mate(bool in_check) const;

	//position.cpp
	void to_fen(Span<char> fen) const;
	void from_fen(Const_Span<char> fen);
	void display() const;

	void set_turn(Color color);

	// Returns whether the positions is 100% a draw.
	// This is not an exhaustive check and should be used merely as an early exit.
	NODISCARD bool is_draw() const
	{
		return is_draw(m_piece_counts);
	}

	//move_legal.cpp

	// Returns whether a move is pseudo legal. A pseudo legal move is
	// a move that respects piece's movement restrictrictions but may leave
	// the king in check.
	NODISCARD bool is_move_pseudo_legal(Move move) const;

	// Returns whether a pseudo legal move is legal in this position.
	NODISCARD bool is_pseudo_legal_move_legal(Move move) const;

	// Returns whether a pseudo legal move is legal in this position. Assumes the king is in check.
	NODISCARD bool is_pseudo_legal_move_legal_in_check(Move move) const;

	// Returns whether the move checks the opponent's king in this position.
	NODISCARD bool is_move_check(Move move) const;

	// Returns whether a pseudo legal move is legal in this position.
	// Uses the in_check parameter to dispatch to a faster method.
	NODISCARD bool is_pseudo_legal_move_legal(Move move, bool in_check) const
	{
		return
			in_check
			? is_pseudo_legal_move_legal_in_check(move)
			: is_pseudo_legal_move_legal(move);
	}

	//move_gen.cpp

	// Generates all pseudo legal moves in this position that capture a piece.
	NODISCARD Move_List gen_pseudo_legal_captures() const;

	// Generates all pseudo legal moves in this position that are quiet (not captures).
	NODISCARD Move_List gen_pseudo_legal_quiets() const;

	// Generates all pseudo legal moves in this position.
	NODISCARD Move_List gen_all_pseudo_legal_moves() const;

	// Generates all pseudo legal reverse moves in this position that are quiet (not captures).
	NODISCARD Move_List gen_pseudo_legal_pre_quiets() const;

	// Executes a capture move. Switches turn. Returns the captured piece.
	Piece do_capture_move(Move move);

	// Undoes a capture move. Switches turn.
	void undo_capture_move(Move move, Piece cap);

	// Executes a quiet move. Switches turn.
	void do_quiet_move(Move move);

	// Undoes a quiet move. Switches turn.
	// Functionally identical to do_quiet_move, but does different assertions.
	void undo_quiet_move(Move move);

	// Executes a null move. Effectively only switches turn.
	void do_null_move()
	{
		m_turn = color_opp(m_turn);
	}

	// Undoes a null move. Effectively only switches turn.
	void undo_null_move()
	{
		m_turn = color_opp(m_turn);
	}

	NODISCARD Color turn() const
	{
		return m_turn;
	}

	// Returns the number of instances of the given piece `p` on the board.
	// Must be a valid piece, not WHITE_OCCUPY, BLACK_OCCUPY, PIECE_NONE.
	NODISCARD int8_t piece_count(Piece p) const
	{
		ASSERT(piece_is_ok(p));
		return m_piece_counts[p];
	}

	// Returns the number of instances of a piece of type `pt` and color `c` on the board.
	// Must be a valid piece, not WHITE_OCCUPY, BLACK_OCCUPY, PIECE_NONE.
	NODISCARD int8_t piece_count(Color c, Piece_Type pt) const
	{
		ASSERT(color_is_ok(c));
		ASSERT(piece_type_is_ok(pt));
		return m_piece_counts[piece_make(c, pt)];
	}

	// Returns the piece on the given square.
	// Returns PIECE_NONE if the square is empty.
	NODISCARD Piece piece_on(Square sq) const
	{
		ASSERT(sq_is_ok(sq));
		return m_squares[sq];
	}

	NODISCARD bool is_empty(Square sq) const
	{
		ASSERT(sq_is_ok(sq));
		return m_squares[sq] == PIECE_NONE;
	}	

	NODISCARD const Bitboard& pieces_bb(Piece p) const
	{
		return m_pieces[p];
	}

	NODISCARD const Bitboard& pieces_bb(Color c, Piece_Type pt) const
	{
		return m_pieces[piece_make(c, pt)];
	}

private:
	// Every piece on the board is included in this bitboard.
	Bitboard m_occupied;

	// Bitboards for specific pieces. Additionally:
	// m_pieces[WHITE_OCCUPY] is a bitboard of all white pieces.
	// m_pieces[BLACK_OCCUPY] is a bitboard of all black pieces.
	// There is no bitboard for empty squares, as it would collide 
	// with WHITE_OCCUPY, and can be easily computed as ~m_occupied.
	Bitboard m_pieces[PIECE_NB];

	// A count of each piece on the board.
	// Only valid for actual pieces. Counts for all white/black pieces is not tracked,
	// so m_piece_counts[WHITE_OCCUPY] and m_piece_counts[BLACK_OCCUPY] is undefined.
	int8_t m_piece_counts[PIECE_NB];

	// A piece on each square. PIECE_NONE if the square is empty. Otherwise a valid piece.
	Piece m_squares[SQUARE_NB];

	// The current side to move.
	Color m_turn;

	NODISCARD static constexpr bool is_draw(const int8_t piece_counts[PIECE_NB])
	{
		if ((piece_counts[WHITE_ROOK] + piece_counts[WHITE_KNIGHT] + piece_counts[WHITE_CANNON] + piece_counts[WHITE_PAWN] +
			piece_counts[BLACK_ROOK] + piece_counts[BLACK_KNIGHT] + piece_counts[BLACK_CANNON] + piece_counts[BLACK_PAWN]) == 0)
			return true;
		return false;
	}
};
