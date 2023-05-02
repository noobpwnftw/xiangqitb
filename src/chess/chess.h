#pragma once

#include <array>

#include "util/defines.h"
#include "util/enum.h"
#include "util/span.h"

constexpr size_t MAX_MAN = 32;

enum Color : int8_t {
	WHITE, BLACK, COLOR_NB = 2
};

enum Piece_Type : int8_t {
	PIECE_TYPE_NONE = 0, KING = 1, ROOK, KNIGHT, CANNON, ADVISOR, BISHOP, PAWN, PIECE_TYPE_NB = 8
};

// WHITE_OCCUPY and BLACK_OCCUPY are used white and black occupation bitboards in Position.
// They are unused anywhere else. This is the best way to reduce memory footprint of Position,
// while allowing PIECE_NONE to be 0.
enum Piece : int8_t {
	WHITE_OCCUPY, WHITE_KING, WHITE_ROOK, WHITE_KNIGHT, WHITE_CANNON, WHITE_ADVISOR, WHITE_BISHOP, WHITE_PAWN,
	BLACK_OCCUPY, BLACK_KING, BLACK_ROOK, BLACK_KNIGHT, BLACK_CANNON, BLACK_ADVISOR, BLACK_BISHOP, BLACK_PAWN,
	PIECE_NONE = 0, PIECE_NB = 16
};

constexpr inline Piece piece_occupy(Color color)
{
	static_assert(WHITE_OCCUPY == (WHITE << 3));
	static_assert(BLACK_OCCUPY == (BLACK << 3));
	return static_cast<Piece>(color << 3);
}

constexpr Piece ALL_PIECES[] = {
	WHITE_KING, WHITE_ROOK, WHITE_KNIGHT, WHITE_CANNON, WHITE_ADVISOR, WHITE_BISHOP, WHITE_PAWN,
	BLACK_KING, BLACK_ROOK, BLACK_KNIGHT, BLACK_CANNON, BLACK_ADVISOR, BLACK_BISHOP, BLACK_PAWN
};

// Pieces (other than king) that can attack opponent's king.
constexpr Piece ALL_FREE_ATTACKING_PIECES[] = {
	WHITE_ROOK, WHITE_KNIGHT, WHITE_CANNON, WHITE_PAWN,
	BLACK_ROOK, BLACK_KNIGHT, BLACK_CANNON, BLACK_PAWN
};

// Squares are ordered by file first, then by rank.
// First 45 squares on the white's side of the board, last 45 on black's.
enum Square : int8_t {
	SQ_A0, SQ_B0, SQ_C0, SQ_D0, SQ_E0, SQ_F0, SQ_G0, SQ_H0, SQ_I0,
	SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1, SQ_I1,
	SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2, SQ_I2,
	SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3, SQ_I3,
	SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4, SQ_I4,
	SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5, SQ_I5,
	SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6, SQ_I6,
	SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7, SQ_I7,
	SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8, SQ_I8,
	SQ_A9, SQ_B9, SQ_C9, SQ_D9, SQ_E9, SQ_F9, SQ_G9, SQ_H9, SQ_I9,
	SQ_END, SQ_START = 0, SQUARE_NB = 90, BISHOP_SQUARE_NB = 7,
	KING_SQUARE_NB = 9, ADVISOR_SQUARE_NB = 5, PAWN_SQUARE_NB = 55
};

constexpr Square FIRST_BLACK_SQUARE = SQ_A5;

ENUM_ENABLE_OPERATOR_INC(Square);
ENUM_ENABLE_OPERATOR_DEC(Square);
ENUM_ENABLE_OPERATOR_ADD(Square);
ENUM_ENABLE_OPERATOR_SUB(Square);
ENUM_ENABLE_OPERATOR_ADD_EQ(Square);
ENUM_ENABLE_OPERATOR_SUB_EQ(Square);
ENUM_ENABLE_OPERATOR_DIFF(Square);

enum File : int8_t {
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I, FILE_END, FILE_START = 0, FILE_NB = 9
};

ENUM_ENABLE_OPERATOR_INC(File);
ENUM_ENABLE_OPERATOR_DEC(File);
ENUM_ENABLE_OPERATOR_ADD(File);
ENUM_ENABLE_OPERATOR_SUB(File);
ENUM_ENABLE_OPERATOR_ADD_EQ(File);
ENUM_ENABLE_OPERATOR_SUB_EQ(File);
ENUM_ENABLE_OPERATOR_DIFF(File);

enum Rank : int8_t {
	RANK_0, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9, RANK_END, RANK_START = 0, RANK_NB = 10
};

ENUM_ENABLE_OPERATOR_INC(Rank);
ENUM_ENABLE_OPERATOR_DEC(Rank);
ENUM_ENABLE_OPERATOR_ADD(Rank);
ENUM_ENABLE_OPERATOR_SUB(Rank);
ENUM_ENABLE_OPERATOR_ADD_EQ(Rank);
ENUM_ENABLE_OPERATOR_SUB_EQ(Rank);
ENUM_ENABLE_OPERATOR_DIFF(Rank);

constexpr Rank SQ_RANK[SQUARE_NB] =
{
	RANK_0, RANK_0, RANK_0, RANK_0, RANK_0, RANK_0, RANK_0, RANK_0, RANK_0,
	RANK_1, RANK_1, RANK_1, RANK_1, RANK_1, RANK_1, RANK_1, RANK_1, RANK_1,
	RANK_2, RANK_2, RANK_2, RANK_2, RANK_2, RANK_2, RANK_2, RANK_2, RANK_2,
	RANK_3, RANK_3, RANK_3, RANK_3, RANK_3, RANK_3, RANK_3, RANK_3, RANK_3,
	RANK_4, RANK_4, RANK_4, RANK_4, RANK_4, RANK_4, RANK_4, RANK_4, RANK_4,
	RANK_5, RANK_5, RANK_5, RANK_5, RANK_5, RANK_5, RANK_5, RANK_5, RANK_5,
	RANK_6, RANK_6, RANK_6, RANK_6, RANK_6, RANK_6, RANK_6, RANK_6, RANK_6,
	RANK_7, RANK_7, RANK_7, RANK_7, RANK_7, RANK_7, RANK_7, RANK_7, RANK_7,
	RANK_8, RANK_8, RANK_8, RANK_8, RANK_8, RANK_8, RANK_8, RANK_8, RANK_8,
	RANK_9, RANK_9, RANK_9, RANK_9, RANK_9, RANK_9, RANK_9, RANK_9, RANK_9,
};

constexpr File SQ_FILE[SQUARE_NB] =
{
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
	FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I,
};

constexpr Color SQ_COLOR[SQUARE_NB] =
{
	WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,
	WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,
	WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,
	WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,
	WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE,
	BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK,
	BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK,
	BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK,
	BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK,
	BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK,
};

constexpr Square SQ_FILE_MIRROR[SQUARE_NB] = {
	SQ_I0, SQ_H0, SQ_G0, SQ_F0, SQ_E0, SQ_D0, SQ_C0, SQ_B0, SQ_A0,
	SQ_I1, SQ_H1, SQ_G1, SQ_F1, SQ_E1, SQ_D1, SQ_C1, SQ_B1, SQ_A1,
	SQ_I2, SQ_H2, SQ_G2, SQ_F2, SQ_E2, SQ_D2, SQ_C2, SQ_B2, SQ_A2,
	SQ_I3, SQ_H3, SQ_G3, SQ_F3, SQ_E3, SQ_D3, SQ_C3, SQ_B3, SQ_A3,
	SQ_I4, SQ_H4, SQ_G4, SQ_F4, SQ_E4, SQ_D4, SQ_C4, SQ_B4, SQ_A4,
	SQ_I5, SQ_H5, SQ_G5, SQ_F5, SQ_E5, SQ_D5, SQ_C5, SQ_B5, SQ_A5,
	SQ_I6, SQ_H6, SQ_G6, SQ_F6, SQ_E6, SQ_D6, SQ_C6, SQ_B6, SQ_A6,
	SQ_I7, SQ_H7, SQ_G7, SQ_F7, SQ_E7, SQ_D7, SQ_C7, SQ_B7, SQ_A7,
	SQ_I8, SQ_H8, SQ_G8, SQ_F8, SQ_E8, SQ_D8, SQ_C8, SQ_B8, SQ_A8,
	SQ_I9, SQ_H9, SQ_G9, SQ_F9, SQ_E9, SQ_D9, SQ_C9, SQ_B9, SQ_A9,
};

constexpr Square SQ_RANK_MIRROR[SQUARE_NB] = {
	SQ_A9, SQ_B9, SQ_C9, SQ_D9, SQ_E9, SQ_F9, SQ_G9, SQ_H9, SQ_I9,
	SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8, SQ_I8,
	SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7, SQ_I7,
	SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6, SQ_I6,
	SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5, SQ_I5,
	SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4, SQ_I4,
	SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3, SQ_I3,
	SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2, SQ_I2,
	SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1, SQ_I1,
	SQ_A0, SQ_B0, SQ_C0, SQ_D0, SQ_E0, SQ_F0, SQ_G0, SQ_H0, SQ_I0,
};

constexpr int8_t PAWN_MOVE_INC[COLOR_NB] = { 9, -9 };

constexpr char START_FEN[] = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w";

constexpr size_t MAX_FEN_LENGTH = 120;

// A unique 32-bit key of the pieces present on the board.
// It assumes that the set of pieces is the subset
// of pieces in the starting position.
// Two kings (one white, one black) are assumed to be present.
struct Material_Key
{
private:
	// The actual constants used are depended upon in a few places,
	// for example in encoding EGTB material, so they must not change.
	// They are chosen such that any valid multiplicity of a piece does
	// not overflow to ambiguity with other piece.
	static constexpr uint32_t MAT_KEY[PIECE_NB] = {
		0, 0, 708588, 78732, 8748, 972, 108, 6,
		0, 0, 236196, 26244, 2916, 324,  36, 1,
	};

public:
	constexpr Material_Key() :
		key(0)
	{
	}

	explicit constexpr Material_Key(uint32_t k) :
		key(k)
	{
	}

	constexpr void add_piece(Piece pc)
	{
		key += MAT_KEY[pc];
	}

	NODISCARD constexpr uint32_t value() const
	{
		return key;
	}

	NODISCARD constexpr friend bool operator==(Material_Key lhs, Material_Key rhs) noexcept
	{
		return lhs.key == rhs.key;
	}

	NODISCARD constexpr friend bool operator!=(Material_Key lhs, Material_Key rhs) noexcept
	{
		return lhs.key != rhs.key;
	}

	NODISCARD constexpr friend bool operator<(Material_Key lhs, Material_Key rhs) noexcept
	{
		return lhs.key < rhs.key;
	}

	NODISCARD constexpr friend bool operator>(Material_Key lhs, Material_Key rhs) noexcept
	{
		return lhs.key > rhs.key;
	}

	NODISCARD constexpr friend bool operator<=(Material_Key lhs, Material_Key rhs) noexcept
	{
		return lhs.key <= rhs.key;
	}

	NODISCARD constexpr friend bool operator>=(Material_Key lhs, Material_Key rhs) noexcept
	{
		return lhs.key >= rhs.key;
	}

private:
	uint32_t key;
};

// Number of squares that can be occupied by a given piece.
// Initialized by init_possible.
extern int8_t PIECE_POSSIBLE_SQUARE_NB[PIECE_NB];

// A list of squares that can be occupied by a given piece.
// Each list has only PIECE_POSSIBLE_SQUARE_NB[p] squares defined.
// Initialized by init_possible.
extern Square PIECE_POSSIBLE_SQUARE[PIECE_NB][SQUARE_NB];

// A reverse mapping of PIECE_POSSIBLE_SQUARE. That is, 
// PIECE_POSSIBLE_SQUARE[p][PIECE_POSSIBLE_SQUARE_INDEX[sq][p]] == sq
// Squares that cannot be occupied by a given piece has an index of -1.
// NOTE: The array rank order is reversed with respect to PIECE_POSSIBLE_SQUARE.
//       This is an optimization, it results in slightly better cache usage during EGTB generation.
// Initialized by init_possible.
extern int8_t PIECE_POSSIBLE_SQUARE_INDEX[SQUARE_NB][PIECE_NB];

// Same as PIECE_POSSIBLE_SQUARE_INDEX but considers pieces of
// either color at the same time (union).
// Used in attack generation.
// Initialized by init_possible.
extern int8_t KING_POS_INDEX[SQUARE_NB];
extern int8_t ADVISOR_POS_INDEX[SQUARE_NB];
extern int8_t BISHOP_POS_INDEX[SQUARE_NB];

// Initializes the following globals:
//   - PIECE_POSSIBLE_SQUARE_NB
//   - PIECE_POSSIBLE_SQUARE
//   - PIECE_POSSIBLE_SQUARE_INDEX
//   - KING_POS_INDEX
//   - ADVISOR_POS_INDEX
//   - BISHOP_POS_INDEX
void init_possible();

NODISCARD INLINE auto possible_sq_nb(Piece piece)
{
	return PIECE_POSSIBLE_SQUARE_NB[piece];
}

NODISCARD INLINE Square possible_sq(Piece piece, size_t index)
{
	return PIECE_POSSIBLE_SQUARE[piece][index];
}

NODISCARD INLINE auto possible_sq_index(Piece piece, Square sq)
{
	return PIECE_POSSIBLE_SQUARE_INDEX[sq][piece];
}

NODISCARD INLINE constexpr auto pawn_move_inc(Color color)
{
	return PAWN_MOVE_INC[color];
}

NODISCARD INLINE constexpr bool color_is_ok(Color color)
{
	return (color == WHITE || color == BLACK);
}

NODISCARD INLINE constexpr Color color_opp(Color color)
{
	ASSERT(color_is_ok(color));
	return static_cast<Color>(color ^ 1);
}

// Returns the opposite color if opp == true. Returns color unchanged otherwise.
NODISCARD INLINE constexpr Color color_maybe_opp(Color color, bool opp)
{
	ASSERT(color_is_ok(color));
	return static_cast<Color>(static_cast<int>(color) ^ static_cast<int>(opp));
}

NODISCARD INLINE constexpr bool piece_is_ok(Piece piece)
{
	return ((piece >= WHITE_KING && piece <= WHITE_PAWN) || (piece >= BLACK_KING && piece <= BLACK_PAWN));
}

NODISCARD INLINE constexpr bool piece_type_is_ok(Piece_Type piece)
{
	return piece >= KING && piece <= PAWN;
}

NODISCARD INLINE constexpr Piece_Type piece_type(Piece piece)
{
	return static_cast<Piece_Type>(piece & 7);
}

NODISCARD INLINE constexpr Color piece_color(Piece piece)
{
	ASSERT(piece != WHITE_OCCUPY && piece != BLACK_OCCUPY);
	return static_cast<Color>(piece >> 3);
}

NODISCARD INLINE constexpr Piece piece_make(Color color, Piece_Type type)
{
	ASSERT(color_is_ok(color));
	ASSERT(piece_type_is_ok(type));
	return static_cast<Piece>((color << 3) + type);
}

NODISCARD INLINE constexpr Piece piece_opp_color(Piece piece)
{
	return piece_make(color_opp(piece_color(piece)), piece_type(piece));
}

NODISCARD constexpr bool sq_is_ok(Square sq)
{
	return sq >= SQ_A0 && sq < SQ_END;
}

NODISCARD constexpr bool file_is_ok(File file)
{
	return file >= FILE_A && file < FILE_END;
}

NODISCARD constexpr bool rank_is_ok(Rank rank)
{
	return rank >= RANK_0 && rank < RANK_END;
}

NODISCARD INLINE constexpr File sq_file(Square sq)
{
	ASSERT(sq_is_ok(sq));
	return SQ_FILE[sq];
}

NODISCARD INLINE constexpr Rank sq_rank(Square sq)
{
	ASSERT(sq_is_ok(sq));
	return SQ_RANK[sq];
}

NODISCARD INLINE constexpr Square sq_mid(Square a, Square b)
{
	ASSERT(sq_is_ok(a) && sq_is_ok(b));
	return static_cast<Square>((static_cast<int>(a) + static_cast<int>(b)) >> 1);
}

NODISCARD INLINE constexpr Square sq_make(Rank rank, File file)
{
	// rank * 9 == rank * 8 + rank
	return static_cast<Square>(((rank) << 3) + rank + file);
}

NODISCARD INLINE constexpr bool sq_equal_rank(Square sq1, Square sq2)
{
	return SQ_RANK[sq1] == SQ_RANK[sq2];
}

NODISCARD INLINE constexpr bool sq_equal_file(Square sq1, Square sq2)
{
	return SQ_FILE[sq1] == SQ_FILE[sq2];
}

NODISCARD INLINE constexpr Color sq_color(Square sq)
{
	ASSERT(sq >= 0 && sq < SQUARE_NB);
	return SQ_COLOR[sq];
}

NODISCARD INLINE constexpr Square sq_file_mirror(Square sq)
{
	return SQ_FILE_MIRROR[sq];
}

NODISCARD INLINE constexpr Square sq_rank_mirror(Square sq)
{
	return SQ_RANK_MIRROR[sq];
}

NODISCARD INLINE auto king_pos_index(Square sq)
{
	return KING_POS_INDEX[sq];
}

NODISCARD INLINE auto advisor_pos_index(Square sq)
{
	return ADVISOR_POS_INDEX[sq];
}

NODISCARD INLINE auto bishop_pos_index(Square sq)
{
	return BISHOP_POS_INDEX[sq];
}

NODISCARD INLINE bool is_king_pos(Square sq)
{
	return KING_POS_INDEX[sq] >= 0;
}

NODISCARD INLINE bool is_advisor_pos(Square sq)
{
	return ADVISOR_POS_INDEX[sq] >= 0;
}

NODISCARD INLINE bool is_bishop_pos(Square sq)
{
	return BISHOP_POS_INDEX[sq] >= 0;
}

NODISCARD INLINE bool is_king_pos(Square sq, Color color)
{
	return KING_POS_INDEX[sq] >= 0 && sq_color(sq) == color;
}

NODISCARD INLINE bool is_advisor_pos(Square sq, Color color)
{
	return ADVISOR_POS_INDEX[sq] >= 0 && sq_color(sq) == color;
}

NODISCARD INLINE bool is_bishop_pos(Square sq, Color color)
{
	return BISHOP_POS_INDEX[sq] >= 0 && sq_color(sq) == color;
}

NODISCARD INLINE constexpr bool is_piece_free_attacker(Piece piece)
{
	const Piece_Type type = piece_type(piece);
	return type == ROOK
		|| type == KNIGHT
		|| type == CANNON
		|| type == PAWN;
}

constexpr char PIECE_STRING[20] = "?KRNCABP?krncabp";
constexpr std::array<Piece, 128> PIECE_FROM_CHAR = [](){
	std::array<Piece, 128> arr{};
	arr['K'] = WHITE_KING;
	arr['R'] = WHITE_ROOK;
	arr['N'] = WHITE_KNIGHT;
	arr['C'] = WHITE_CANNON;
	arr['A'] = WHITE_ADVISOR;
	arr['B'] = WHITE_BISHOP;
	arr['P'] = WHITE_PAWN;
	arr['k'] = BLACK_KING;
	arr['r'] = BLACK_ROOK;
	arr['n'] = BLACK_KNIGHT;
	arr['c'] = BLACK_CANNON;
	arr['a'] = BLACK_ADVISOR;
	arr['b'] = BLACK_BISHOP;
	arr['p'] = BLACK_PAWN;
	return arr;
}();

NODISCARD INLINE constexpr File file_from_char(char c)
{
	return FILE_I - (c - 'a');
}

NODISCARD INLINE constexpr Rank rank_from_char(char c)
{
	return RANK_0 + (c - '0');
}

NODISCARD INLINE constexpr char file_to_char(File file)
{
	return static_cast<char>('i' - (file - FILE_A));
}

NODISCARD INLINE constexpr char rank_to_char(Rank rank)
{
	return static_cast<char>('0' + (rank - RANK_0));
}

NODISCARD INLINE constexpr Piece piece_from_char(char c)
{
	if (c >= 0)
		return PIECE_FROM_CHAR[c];
	return PIECE_NONE;
}

NODISCARD INLINE constexpr char piece_to_char(Piece p)
{
	ASSERT(p >= 0 && p < PIECE_NB);
	return PIECE_STRING[p];
}

NODISCARD INLINE constexpr char piece_type_to_char(Piece_Type p)
{
	ASSERT(p >= 0 && p < PIECE_TYPE_NB);
	return PIECE_STRING[p];
}

INLINE constexpr void square_to_string(Square sq, char string[])
{
	ASSERT(sq_is_ok(sq));
	string[0] = file_to_char(sq_file(sq));
	string[1] = rank_to_char(sq_rank(sq));
	string[2] = '\0';
}

NODISCARD INLINE constexpr Square square_from_string(const char string[])
{
	if (string[0] < 'a' || string[0] > 'i')
		return SQ_END;
	if (string[1] < '0' || string[1] > '9')
		return SQ_END;
	if (string[2] != '\0')
		return SQ_END;

	const File file = file_from_char(string[0]);
	const Rank rank = rank_from_char(string[1]);

	return sq_make(rank, file);
}
