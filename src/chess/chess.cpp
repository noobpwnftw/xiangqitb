#include "chess.h"

#include "bitboard.h"

#include "util/defines.h"

#include <cstring>

int8_t PIECE_POSSIBLE_SQUARE_NB[PIECE_NB];

Square PIECE_POSSIBLE_SQUARE[PIECE_NB][SQUARE_NB];
int8_t PIECE_POSSIBLE_SQUARE_INDEX[SQUARE_NB][PIECE_NB];

int8_t KING_POS_INDEX[SQUARE_NB];
int8_t ADVISOR_POS_INDEX[SQUARE_NB];
int8_t BISHOP_POS_INDEX[SQUARE_NB];
	
void init_possible()
{
	// NOTE: The order of possible squares (PIECE_POSSIBLE_SQUARE) MATTERS because it 
	//       determines the order of square lists in the piece groups. 
	//       The placement of these square lists determines the position index.
	//       
	//       The order of possible square indices (PIECE_POSSIBLE_SQUARE_INDEX) DOESN'T MATTER, 
	//       because it's only used as an indirection into an array that translates
	//       square lists into square configuration indices; and the same indexing is
	//       used on creation as is used on lookup. It is only required that they are
	//       unique, start from 0, and don't have gaps.

	std::memset(PIECE_POSSIBLE_SQUARE, static_cast<int8_t>(-1), sizeof(PIECE_POSSIBLE_SQUARE));
	std::memset(PIECE_POSSIBLE_SQUARE_INDEX, static_cast<int8_t>(-1), sizeof(PIECE_POSSIBLE_SQUARE_INDEX));
	std::memset(KING_POS_INDEX, static_cast<int8_t>(-1), sizeof(KING_POS_INDEX));
	std::memset(ADVISOR_POS_INDEX, static_cast<int8_t>(-1), sizeof(ADVISOR_POS_INDEX));
	std::memset(BISHOP_POS_INDEX, static_cast<int8_t>(-1), sizeof(BISHOP_POS_INDEX));

	size_t k[COLOR_NB]{ 0, 0 };
	size_t kk = 0;
	size_t a[COLOR_NB]{ 0, 0 };
	size_t aa = 0;
	size_t b[COLOR_NB]{ 0, 0 };
	size_t bb = 0;
	size_t p[COLOR_NB]{ 0, 0 };

	for (Square i = SQ_START; i < SQ_END; ++i)
	{
		PIECE_POSSIBLE_SQUARE[WHITE_ROOK][i] = i;
		PIECE_POSSIBLE_SQUARE[BLACK_ROOK][i] = i;
		PIECE_POSSIBLE_SQUARE[WHITE_KNIGHT][i] = i;
		PIECE_POSSIBLE_SQUARE[BLACK_KNIGHT][i] = i;
		PIECE_POSSIBLE_SQUARE[WHITE_CANNON][i] = i;
		PIECE_POSSIBLE_SQUARE[BLACK_CANNON][i] = i;

		PIECE_POSSIBLE_SQUARE_INDEX[i][WHITE_ROOK] = i;
		PIECE_POSSIBLE_SQUARE_INDEX[i][BLACK_ROOK] = i;
		PIECE_POSSIBLE_SQUARE_INDEX[i][WHITE_KNIGHT] = i;
		PIECE_POSSIBLE_SQUARE_INDEX[i][BLACK_KNIGHT] = i;
		PIECE_POSSIBLE_SQUARE_INDEX[i][WHITE_CANNON] = i;
		PIECE_POSSIBLE_SQUARE_INDEX[i][BLACK_CANNON] = i;

		const Bitboard sq_2_bbs = square_bb(i);
		for (const Color color : { WHITE, BLACK })
		{
			if (king_area_bb()[color] & sq_2_bbs[color])
			{
				PIECE_POSSIBLE_SQUARE[piece_make(color, KING)][k[color]] = i;
				PIECE_POSSIBLE_SQUARE_INDEX[i][piece_make(color, KING)] = narrowing_static_cast<int8_t>(k[color]);
				k[color] += 1;
			}
			
			if (advisor_area_bb()[color] & sq_2_bbs[color])
			{
				PIECE_POSSIBLE_SQUARE[piece_make(color, ADVISOR)][a[color]] = i;
				PIECE_POSSIBLE_SQUARE_INDEX[i][piece_make(color, ADVISOR)] = narrowing_static_cast<int8_t>(a[color]);
				a[color] += 1;
			}
			
			if (bishop_area_bb()[color] & sq_2_bbs[color])
			{
				PIECE_POSSIBLE_SQUARE[piece_make(color, BISHOP)][b[color]] = i;
				PIECE_POSSIBLE_SQUARE_INDEX[i][piece_make(color, BISHOP)] = narrowing_static_cast<int8_t>(b[color]);
				b[color] += 1;
			}
			
			if (pawn_area_bb(color) & sq_2_bbs)
			{
				PIECE_POSSIBLE_SQUARE[piece_make(color, PAWN)][p[color]] = i;
				PIECE_POSSIBLE_SQUARE_INDEX[i][piece_make(color, PAWN)] = narrowing_static_cast<int8_t>(p[color]);
				p[color] += 1;
			}
		}

		if (king_area_bb() & sq_2_bbs)
			KING_POS_INDEX[i] = narrowing_static_cast<int8_t>(kk++);

		if (advisor_area_bb() & sq_2_bbs)
			ADVISOR_POS_INDEX[i] = narrowing_static_cast<int8_t>(aa++);

		if (bishop_area_bb() & sq_2_bbs)
			BISHOP_POS_INDEX[i] = narrowing_static_cast<int8_t>(bb++);
	}

	ASSERT(k[WHITE] == KING_SQUARE_NB);
	ASSERT(k[BLACK] == KING_SQUARE_NB);
	ASSERT(a[WHITE] == ADVISOR_SQUARE_NB);
	ASSERT(a[BLACK] == ADVISOR_SQUARE_NB);
	ASSERT(b[WHITE] == BISHOP_SQUARE_NB);
	ASSERT(b[BLACK] == BISHOP_SQUARE_NB);
	ASSERT(p[WHITE] == PAWN_SQUARE_NB);
	ASSERT(p[BLACK] == PAWN_SQUARE_NB);
	ASSERT(kk == KING_SQUARE_NB * 2);
	ASSERT(aa == ADVISOR_SQUARE_NB * 2);
	ASSERT(bb == BISHOP_SQUARE_NB * 2);

	PIECE_POSSIBLE_SQUARE_NB[WHITE_ROOK] = SQUARE_NB;
	PIECE_POSSIBLE_SQUARE_NB[BLACK_ROOK] = SQUARE_NB;
	PIECE_POSSIBLE_SQUARE_NB[WHITE_KNIGHT] = SQUARE_NB;
	PIECE_POSSIBLE_SQUARE_NB[BLACK_KNIGHT] = SQUARE_NB;
	PIECE_POSSIBLE_SQUARE_NB[WHITE_CANNON] = SQUARE_NB;
	PIECE_POSSIBLE_SQUARE_NB[BLACK_CANNON] = SQUARE_NB;
	PIECE_POSSIBLE_SQUARE_NB[WHITE_KING] = narrowing_static_cast<int8_t>(k[WHITE]);
	PIECE_POSSIBLE_SQUARE_NB[BLACK_KING] = narrowing_static_cast<int8_t>(k[BLACK]);
	PIECE_POSSIBLE_SQUARE_NB[WHITE_ADVISOR] = narrowing_static_cast<int8_t>(a[WHITE]);
	PIECE_POSSIBLE_SQUARE_NB[BLACK_ADVISOR] = narrowing_static_cast<int8_t>(a[BLACK]);
	PIECE_POSSIBLE_SQUARE_NB[WHITE_BISHOP] = narrowing_static_cast<int8_t>(b[WHITE]);
	PIECE_POSSIBLE_SQUARE_NB[BLACK_BISHOP] = narrowing_static_cast<int8_t>(b[BLACK]);
	PIECE_POSSIBLE_SQUARE_NB[WHITE_PAWN] = narrowing_static_cast<int8_t>(p[WHITE]);
	PIECE_POSSIBLE_SQUARE_NB[BLACK_PAWN] = narrowing_static_cast<int8_t>(p[BLACK]);
}
