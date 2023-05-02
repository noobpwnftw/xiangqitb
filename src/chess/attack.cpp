#include "attack.h"
#include "bitboard.h"
#include "chess.h"

#include "util/defines.h"

Bitboard FILE_BLOCK_MASK[FILE_NB];
Bitboard RANK_BLOCK_MASK[RANK_NB];
Bitboard BISHOP_BLOCK_MASK[BISHOP_SQUARE_NB * 2];
Bitboard KNIGHT_BLOCK_MASK[SQUARE_NB];
Bitboard KNIGHT_ATTACKED_MASK[SQUARE_NB];
Bitboard BISHOP_ATTACK_BB[BISHOP_SQUARE_NB * 2][pow_2(BISHOP_ATTACK_NB)];
Bitboard KNIGHT_ATTACK_BB[SQUARE_NB][pow_2(KNIGHT_ATTACK_NB / 2)];
Bitboard KNIGHT_ATTACKED_BB[SQUARE_NB][pow_2(KNIGHT_ATTACK_NB / 2)];
Bitboard ROOK_RANK_ATTACK_TABLE[SQUARE_NB][pow_2(FILE_NB - 2)];
Bitboard ROOK_FILE_ATTACK_TABLE[SQUARE_NB][pow_2(RANK_NB - 2)];
Bitboard CANNON_RANK_ATTACK_TABLE[SQUARE_NB][pow_2(FILE_NB - 2)];
Bitboard CANNON_FILE_ATTACK_TABLE[SQUARE_NB][pow_2(RANK_NB - 2)];

Bitboard KING_ATTACK[KING_SQUARE_NB * 2];
Bitboard ADVISOR_ATTACK[ADVISOR_SQUARE_NB * 2];
Bitboard PAWN_ATTACK[SQUARE_NB][COLOR_NB];
Bitboard PAWN_ATTACKED[SQUARE_NB][COLOR_NB];

Bitboard BISHOP_ATTACK_NO_MASK[BISHOP_SQUARE_NB * 2];
Bitboard KNIGHT_ATTACK_NO_MASK[SQUARE_NB];

Bitboard RANK_BETWEEN_MASK[FILE_NB][ceil_to_power_of_2(static_cast<size_t>(FILE_NB))];
Bitboard FILE_BETWEEN_MASK[RANK_NB][ceil_to_power_of_2(static_cast<size_t>(RANK_NB))];

constexpr size_t DIR_NB = 2;

constexpr int8_t PAWN_RANK_INC[COLOR_NB] = { 9, -9 };
constexpr int8_t RANK_INC[DIR_NB] = { -9, 9 };
constexpr int8_t FILE_INC[DIR_NB] = { -1, 1 };
constexpr int8_t KNIGHT_INC[KNIGHT_ATTACK_NB] = { -19, -17, -11, -7, 7, 11, 17, 19 };
constexpr int8_t KNIGHT_LEG_INC[KNIGHT_ATTACK_NB] = { -9, -9, -1, 1, -1, 1, 9, 9 };
constexpr int8_t KNIGHT_LEGED_INC[KNIGHT_ATTACK_NB] = { -10, -8, -10, -8, 8, 10, 8, 10 };
constexpr int8_t BISHOP_INC[BISHOP_ATTACK_NB] = { -20, -16, 16, 20 };
constexpr int8_t ADVISOR_INC[ADVISOR_ATTACK_NB] = { -10, -8, 8, 10 };

static ptrdiff_t sq_distance(Square sq1, Square sq2);
static Bitboard ix_2_bb(size_t ix, const Bitboard& mask);

static Bitboard rank_block_mask(Square sq);
static Bitboard file_block_mask(Square sq);
static Bitboard knight_block_mask(Square sq);
static Bitboard bishop_block_mask(Square sq);
static Bitboard knight_attd_mask(Square sq);

static Bitboard rook_rank_att(Square sq, const Bitboard& block);
static Bitboard cannon_rank_att(Square sq, const Bitboard& block);
static Bitboard rook_file_att(Square sq, const Bitboard& block);
static Bitboard cannon_file_att(Square sq, const Bitboard& block);

static Bitboard knight_att(Square sq, const Bitboard& block);
static Bitboard bishop_att(Square sq, const Bitboard& block);
static Bitboard knight_attd(Square sq, const Bitboard& block);

static Bitboard pawn_att(Square sq, Color color);
static Bitboard pawn_attd(Square sq, Color color);
static Bitboard advisor_att(Square sq);
static Bitboard king_att(Square sq);

static void magic_attack_init();
static void base_attack_init();

void attack_init()
{
	base_attack_init();
	magic_attack_init();
}

static void base_attack_init()
{
	for (Rank r1 = RANK_START; r1 < RANK_END; ++r1)
	{
		for (Rank r2 = RANK_START; r2 < RANK_END; ++r2)
		{
			Bitboard bb = Bitboard::make_empty();
			for (Rank i = std::min(r1, r2) + 1; i < std::max(r1, r2); ++i)
				bb |= rank_bb(i);
			FILE_BETWEEN_MASK[r1][r2] = bb;
		}
	}

	for (File f1 = FILE_START; f1 < FILE_END; ++f1)
	{
		for (File f2 = FILE_START; f2 < FILE_END; ++f2)
		{
			Bitboard bb = Bitboard::make_empty();
			for (File i = std::min(f1, f2) + 1; i < std::max(f1, f2); ++i)
				bb |= file_bb(i);
			RANK_BETWEEN_MASK[f1][f2] = bb;
		}
	}

	for (Square sq = SQ_START; sq < SQ_END; ++sq)
	{
		if (is_king_pos(sq))
			KING_ATTACK[king_pos_index(sq)] = king_att(sq);

		if (is_advisor_pos(sq))
			ADVISOR_ATTACK[advisor_pos_index(sq)] = advisor_att(sq);

		if (is_bishop_pos(sq))
			BISHOP_ATTACK_NO_MASK[bishop_pos_index(sq)] = bishop_att(sq, Bitboard::make_empty());

		PAWN_ATTACK[sq][WHITE] = pawn_att(sq, WHITE);
		PAWN_ATTACK[sq][BLACK] = pawn_att(sq, BLACK);
		PAWN_ATTACKED[sq][WHITE] = pawn_attd(sq, WHITE);
		PAWN_ATTACKED[sq][BLACK] = pawn_attd(sq, BLACK);
		KNIGHT_ATTACK_NO_MASK[sq] = knight_att(sq, Bitboard::make_empty());
	}
}

void magic_attack_init()
{
	for (File f = FILE_START; f < FILE_END; ++f)
	{
		FILE_BLOCK_MASK[f] = file_block_mask(sq_make(RANK_0, f));
		const size_t bits = FILE_BLOCK_MASK[f].num_set_bits();
		for (Rank r = RANK_START; r < RANK_END; ++r)
		{
			const Square sq = sq_make(r, f);
			for (size_t i = 0; i < nth_bit(bits); ++i)
			{
				const Bitboard mask = ix_2_bb(i, FILE_BLOCK_MASK[f]);
				const size_t index = apply_magic(mask, FILE_MAGIC[f]) >> FILE_SHIFT;
				ROOK_FILE_ATTACK_TABLE[sq][index] = rook_file_att(sq, mask);
				CANNON_FILE_ATTACK_TABLE[sq][index] = cannon_file_att(sq, mask);
			}
		}
	}

	for (Rank r = RANK_START; r < RANK_END; ++r)
	{
		RANK_BLOCK_MASK[r] = rank_block_mask(sq_make(r, FILE_A));
		const size_t bits = RANK_BLOCK_MASK[r].num_set_bits();
		for (File f = FILE_START; f < FILE_END; ++f)
		{
			const Square sq = sq_make(r, f);
			for (size_t i = 0; i < nth_bit(bits); ++i)
			{
				const Bitboard mask = ix_2_bb(i, RANK_BLOCK_MASK[r]);
				const size_t index = (mask[sq_color(sq)] >> RANK_SHIFT_RIGHT[r]) % pow_2(FILE_NB - 2);
				ROOK_RANK_ATTACK_TABLE[sq][index] = rook_rank_att(sq, mask);
				CANNON_RANK_ATTACK_TABLE[sq][index] = cannon_rank_att(sq, mask);
			}
		}
	}

	for (Square sq = SQ_START; sq < SQ_END; ++sq)
	{
		{
			KNIGHT_BLOCK_MASK[sq] = knight_block_mask(sq);
			const size_t bits = KNIGHT_BLOCK_MASK[sq].num_set_bits();
			for (size_t i = 0; i < nth_bit(bits); ++i)
			{
				const Bitboard mask = ix_2_bb(i, KNIGHT_BLOCK_MASK[sq]);
				const Bitboard attack = knight_att(sq, mask);
				const size_t index = apply_magic(mask, KNIGHT_MAGIC[sq]) >> KNIGHT_SHIFT;
				KNIGHT_ATTACK_BB[sq][index] = attack;
			}
		}

		{
			KNIGHT_ATTACKED_MASK[sq] = knight_attd_mask(sq);
			const size_t bits = KNIGHT_ATTACKED_MASK[sq].num_set_bits();
			for (size_t i = 0; i < nth_bit(bits); ++i)
			{
				const Bitboard mask = ix_2_bb(i, KNIGHT_ATTACKED_MASK[sq]);
				const Bitboard attack = knight_attd(sq, mask);
				const size_t index = apply_magic(mask, BY_KNIGHT_MAGIC[sq]) >> KNIGHT_SHIFT;
				KNIGHT_ATTACKED_BB[sq][index] = attack;
			}
		}

		if (is_bishop_pos(sq))
		{
			const size_t ix = bishop_pos_index(sq);
			BISHOP_BLOCK_MASK[ix] = bishop_block_mask(sq);
			const size_t bits = BISHOP_BLOCK_MASK[ix].num_set_bits();
			for (size_t i = 0; i < nth_bit(bits); ++i)
			{
				const Bitboard mask = ix_2_bb(i, BISHOP_BLOCK_MASK[ix]);
				const Bitboard attack = bishop_att(sq, mask);
				const size_t index = apply_magic(mask, BISHOP_MAGIC[ix]) >> BISHOP_SHIFT;
				BISHOP_ATTACK_BB[ix][index] = attack;
			}
		}
	}
}

NODISCARD static ptrdiff_t sq_distance(Square sq1, Square sq2)
{
	return std::max(std::abs(sq_rank(sq1) - sq_rank(sq2)), std::abs(sq_file(sq1) - sq_file(sq2)));
}

NODISCARD static Bitboard ix_2_bb(size_t index, const Bitboard& block)
{
	Bitboard rlt = Bitboard::make_empty();
	Bitboard mask = block;

	for (size_t i = 0; mask; ++i)
	{
		const size_t j = mask.pop_1st_bit();
		if (index & nth_bit(i))
			rlt.set_bit(j);
	}
	return rlt;
}

NODISCARD static Bitboard rook_file_att(Square sq, const Bitboard& block)
{
	Bitboard attack = Bitboard::make_empty();
	for (size_t i = 0; i < DIR_NB; ++i)
	{
		for (Square tmp = sq + RANK_INC[i]; sq_is_ok(tmp); tmp += RANK_INC[i])
		{
			attack |= tmp;
			if (block.has_square(tmp))
				break;
		}
	}
	return attack;
}

NODISCARD static Bitboard rook_rank_att(Square sq, const Bitboard& block)
{
	Bitboard attack = Bitboard::make_empty();
	for (size_t i = 0; i < DIR_NB; ++i)
	{
		for (Square tmp = sq + FILE_INC[i]; sq_is_ok(tmp) && sq_equal_rank(tmp, sq); tmp += FILE_INC[i])
		{
			attack |= tmp;
			if (block.has_square(tmp))
				break;
		}
	}
	return attack;
}

NODISCARD static Bitboard cannon_file_att(Square sq, const Bitboard& block)
{
	Bitboard attack = Bitboard::make_empty();
	for (size_t i = 0; i < DIR_NB; ++i)
	{
		bool find = false;
		for (Square tmp = sq + RANK_INC[i]; sq_is_ok(tmp); tmp += RANK_INC[i])
		{
			if (find)
			{
				attack |= tmp;
				if (block.has_square(tmp))
					break;
			}
			else
			{
				find = block.has_square(tmp);
			}
		}
	}
	return attack;
}

NODISCARD static Bitboard cannon_rank_att(Square sq, const Bitboard& block)
{
	Bitboard attack = Bitboard::make_empty();
	for (size_t i = 0; i < DIR_NB; ++i)
	{
		bool find = false;
		for (Square tmp = sq + FILE_INC[i]; sq_is_ok(tmp) && sq_equal_rank(tmp, sq); tmp += FILE_INC[i])
		{
			if (find)
			{
				attack |= tmp;
				if (block.has_square(tmp))
					break;
			}
			else
			{
				find = block.has_square(tmp);
			}
		}
	}
	return attack;
}

NODISCARD static Bitboard knight_att(Square sq, const Bitboard& block)
{
	Bitboard attack = Bitboard::make_empty();
	for (size_t i = 0; i < KNIGHT_ATTACK_NB; ++i)
	{
		const Square tmp = sq + KNIGHT_INC[i];
		if (sq_is_ok(tmp) && sq_distance(sq, tmp) <= 2)
		{
			const Square t = sq + KNIGHT_LEG_INC[i];
			if (sq_is_ok(t) && !block.has_square(t))
				attack |= tmp;
		}
	}
	return attack;
}

NODISCARD static Bitboard bishop_att(Square sq, const Bitboard& block)
{
	Bitboard attack = Bitboard::make_empty();

	if (!is_bishop_pos(sq))
		return attack;

	for (size_t i = 0; i < BISHOP_ATTACK_NB; ++i)
	{
		const Square tmp = sq + BISHOP_INC[i];
		if (sq_is_ok(tmp)
			&& is_bishop_pos(tmp)
			&& !block.has_square(static_cast<Square>((static_cast<int>(sq) + static_cast<int>(tmp)) / 2)))
			attack |= tmp;
	}
	return attack;
}

NODISCARD static Bitboard knight_attd(Square sq, const Bitboard& block)
{
	Bitboard attack = Bitboard::make_empty();
	for (size_t i = 0; i < KNIGHT_ATTACK_NB; ++i)
	{
		const Square tmp = sq + KNIGHT_INC[i];
		if (sq_is_ok(tmp) && sq_distance(sq, tmp) <= 2)
		{
			const Square t = sq + KNIGHT_LEGED_INC[i];
			if (sq_is_ok(t) && !block.has_square(t))
				attack |= tmp;
		}
	}
	return attack;
}

enum struct Pawn_Move_Dir
{
	Forward,
	Backward
};

template <Pawn_Move_Dir DIR>
NODISCARD static Bitboard pawn_att_generic(Square sq, Color color)
{
	Bitboard attack = Bitboard::make_empty();

	const Square fwd = sq + PAWN_RANK_INC[color_maybe_opp(color, DIR == Pawn_Move_Dir::Backward)];
	if (sq_is_ok(fwd))
		attack |= fwd;

	if (color == WHITE ? sq >= FIRST_BLACK_SQUARE : sq < FIRST_BLACK_SQUARE)
	{
		const Square right = sq + 1;
		if (sq_is_ok(right) && sq_rank(right) == sq_rank(sq))
			attack |= right;

		const Square left = sq - 1;
		if (sq_is_ok(left) && sq_rank(left) == sq_rank(sq))
			attack |= left;
	}

	return attack;
}

NODISCARD static Bitboard pawn_att(Square sq, Color color)
{
	return pawn_att_generic<Pawn_Move_Dir::Forward>(sq, color);
}

NODISCARD static Bitboard pawn_attd(Square sq, Color color)
{
	return pawn_att_generic<Pawn_Move_Dir::Backward>(sq, color);
}

NODISCARD static Bitboard advisor_att(Square sq)
{
	Bitboard attack = Bitboard::make_empty();

	if (!is_advisor_pos(sq))
		return attack;

	for (size_t i = 0; i < ADVISOR_ATTACK_NB; ++i)
	{
		const Square tmp = sq + ADVISOR_INC[i];
		if (sq_is_ok(tmp) && is_advisor_pos(tmp))
			attack |= tmp;
	}

	return attack;
}

NODISCARD static Bitboard king_att(Square sq)
{
	Bitboard attack = Bitboard::make_empty();

	if (!is_king_pos(sq))
		return attack;

	for (size_t i = 0; i < DIR_NB; ++i)
	{
		const Square tmp_rank = sq + RANK_INC[i];
		if (sq_is_ok(tmp_rank) && is_king_pos(tmp_rank))
			attack |= tmp_rank;

		const Square tmp_file = sq + FILE_INC[i];
		if (sq_is_ok(tmp_file) && is_king_pos(tmp_file))
			attack |= tmp_file;
	}

	return attack;
}

NODISCARD static Bitboard rank_block_mask(Square sq)
{
	Bitboard mask = Bitboard::make_empty();

	for (Square tmp = sq + FILE_INC[0]; tmp >= SQ_START && sq_rank(tmp) == sq_rank(sq) && sq_file(tmp) > FILE_A; tmp += FILE_INC[0])
		mask |= tmp;

	for (Square tmp = sq + FILE_INC[1]; tmp < SQ_END && sq_rank(tmp) == sq_rank(sq) && sq_file(tmp) < FILE_I; tmp += FILE_INC[1])
		mask |= tmp;

	return mask;
}

NODISCARD static Bitboard file_block_mask(Square sq)
{
	Bitboard mask = Bitboard::make_empty();

	for (Square tmp = sq + RANK_INC[0]; tmp >= SQ_START && sq_rank(tmp) > RANK_0; tmp += RANK_INC[0])
		mask |= tmp;

	for (Square tmp = sq + RANK_INC[1]; tmp < SQ_END && sq_rank(tmp) < RANK_9; tmp += RANK_INC[1])
		mask |= tmp;

	return mask;
}

NODISCARD static Bitboard knight_mask(Square sq, const int8_t inc[KNIGHT_ATTACK_NB])
{
	Bitboard mask = Bitboard::make_empty();

	for (size_t i = 0; i < KNIGHT_ATTACK_NB; ++i)
	{
		const Square tmp = sq + KNIGHT_INC[i];
		if (sq_is_ok(tmp) && sq_distance(sq, tmp) <= 2)
			mask |= sq + inc[i];
	}

	return mask;
}

NODISCARD static Bitboard knight_block_mask(Square sq)
{
	return knight_mask(sq, KNIGHT_LEG_INC);
}

NODISCARD static Bitboard knight_attd_mask(Square sq)
{
	return knight_mask(sq, KNIGHT_LEGED_INC);
}

NODISCARD static Bitboard bishop_block_mask(Square sq)
{
	Bitboard mask = Bitboard::make_empty();

	if (!is_bishop_pos(sq))
		return mask;

	for (size_t i = 0; i < BISHOP_ATTACK_NB; ++i)
	{
		const Square tmp = sq + BISHOP_INC[i];
		if (sq_is_ok(tmp) && is_bishop_pos(tmp))
			mask |= static_cast<Square>((static_cast<int>(sq) + static_cast<int>(tmp)) / 2);
	}

	return mask;
}
