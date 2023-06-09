#pragma once

#include "bitboard.h"
#include "chess.h"

#include "util/defines.h"
#include "util/math.h"

constexpr int8_t FILE_SHIFT = 56;
constexpr int8_t KNIGHT_SHIFT = 60;
constexpr int8_t BISHOP_SHIFT = 60;

constexpr uint8_t RANK_SHIFT_RIGHT[RANK_NB] = {
	20, 29, 38, 47, 56, 1, 10, 19, 28, 37
};

constexpr uint8_t RANK_SHIFT_LEFT[RANK_NB] = {
	0, 9, 18, 27, 36, 45, 54, 63, 72, 81
};

constexpr int8_t KNIGHT_BLOCKER_OFFSET[40] = {
	0,  -9,   0,  -9,   0,   0,   0,   0,   0,  -1,
	0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,   0,   0,   0,  -1,   0,   0,
	0,   1,   0,   0,   0,   0,   0,   9,   0,   9,
};

enum Attack_Magic : uint64_t {};

#define M(v) Attack_Magic(v)

constexpr Attack_Magic FILE_MAGIC[FILE_NB] = {
	M(0x081024080088100cULL), M(0x1006284400404801ULL), M(0x0810807200301001ULL),
	M(0x0088010004980810ULL), M(0x0281001101080088ULL), M(0x2420808041040308ULL),
	M(0x0040050120008085ULL), M(0x0420278308008022ULL), M(0x2911110114008021ULL),
};

constexpr Attack_Magic RANK_MAGIC[RANK_NB] = {
	M(0x000564600003a041ULL), M(0x0081822102000000ULL), M(0x0001000000080000ULL),
	M(0x0000030011800400ULL), M(0x0000010000000002ULL), M(0xc100001181113000ULL),
	M(0x48008000001000c1ULL), M(0x1801220400009810ULL), M(0x048130406002008aULL),
	M(0x0000000021020200ULL),
};

constexpr Attack_Magic BISHOP_MAGIC[BISHOP_SQUARE_NB * 2] = {
	M(0x00020028a2000142ULL), M(0x0001800208101404ULL), M(0x0000000420005000ULL),
	M(0x00c8004040001880ULL), M(0x0000030008040400ULL), M(0x00000004000c2000ULL),
	M(0x0020003000001089ULL), M(0x0038088a00001090ULL), M(0x0000a00004c24060ULL),
	M(0x10200a0403488008ULL), M(0x1021010082010504ULL), M(0x200910002002010cULL),
	M(0x0000d05201000010ULL), M(0x0880020084010000ULL),
};

constexpr Attack_Magic KNIGHT_MAGIC[SQUARE_NB] = {
	M(0x1000420410000021ULL), M(0x10200c4200002000ULL), M(0x23000b8100008000ULL),
	M(0x3420044020000000ULL), M(0x2050113920010080ULL), M(0x0202410811018044ULL),
	M(0x0960008912000001ULL), M(0x0800a01d08104310ULL), M(0x8102001001010400ULL),
	M(0x0000004207402080ULL), M(0x8020080041400000ULL), M(0x1000002a49240000ULL),
	M(0x6080a04042810900ULL), M(0x00000024204000a7ULL), M(0x009080001008002fULL),
	M(0x0000002089040001ULL), M(0x0c20008014060008ULL), M(0x2000000104808080ULL),
	M(0x6000900428804040ULL), M(0x0080100400502800ULL), M(0x29084450d0804004ULL),
	M(0x24010404804008c0ULL), M(0x0440400040a01408ULL), M(0x00240004421218c2ULL),
	M(0x0110202058040808ULL), M(0x000000334228040dULL), M(0x0000001004894200ULL),
	M(0x0040084001388024ULL), M(0x100a000802012810ULL), M(0x84b1100000864308ULL),
	M(0x42002001a2247060ULL), M(0x2000001000400c08ULL), M(0x420000008021044aULL),
	M(0x2108000880040604ULL), M(0x0204000001015402ULL), M(0x001100000500810dULL),
	M(0x20040400050a2014ULL), M(0x4528080000022260ULL), M(0x2000084221802011ULL),
	M(0x08a0882028030890ULL), M(0x04200a00c0844408ULL), M(0x00c0002160000834ULL),
	M(0x8088020002000801ULL), M(0x0105054000400102ULL), M(0x8020000000109051ULL),
	M(0x21290a1000001040ULL), M(0x08140000c0000018ULL), M(0x9028400800100148ULL),
	M(0x0402081000099060ULL), M(0x2482008112002810ULL), M(0x2d01800090002088ULL),
	M(0x0f00604092001012ULL), M(0x0208b05034102081ULL), M(0x00c0080120000201ULL),
	M(0x2004140050000080ULL), M(0x1010832002000204ULL), M(0x1008058200000820ULL),
	M(0x8844008100840022ULL), M(0x0402084020084500ULL), M(0xa100810105004141ULL),
	M(0x0180408010000000ULL), M(0x01108010000c4000ULL), M(0x2080101102541000ULL),
	M(0x004c8c0424181010ULL), M(0x0028420100001024ULL), M(0x10c2040220040300ULL),
	M(0x4802010602500200ULL), M(0x0013844880022000ULL), M(0x5002834080800000ULL),
	M(0x0090b04060020202ULL), M(0x000080a231000480ULL), M(0x1400442402100010ULL),
	M(0x4020110101812020ULL), M(0x0b00050200000290ULL), M(0x0010814880010010ULL),
	M(0x98001112c0128010ULL), M(0x40500a0844141008ULL), M(0x0102802040020201ULL),
	M(0x06c6154584100000ULL), M(0x0101201108000160ULL), M(0x0000002209000000ULL),
	M(0x1202000804000020ULL), M(0x404000a201400000ULL), M(0x10030000810d1c00ULL),
	M(0x00100000804080c0ULL), M(0x4809000280401000ULL), M(0x0020000020c40000ULL),
	M(0x0000008210180d00ULL), M(0x0120020008300000ULL), M(0x6328000102078100ULL),
};

constexpr Attack_Magic BY_KNIGHT_MAGIC[SQUARE_NB] = {
	M(0x8020020501200000ULL), M(0x1070008302022040ULL), M(0x4141081900080800ULL),
	M(0x0000000220000004ULL), M(0x4100022920063000ULL), M(0x0824042030180028ULL),
	M(0x04ba004288020c41ULL), M(0x290000040c002020ULL), M(0x06c8004004000400ULL),
	M(0x14408a4000800010ULL), M(0x0512888004402808ULL), M(0x0240850200808302ULL),
	M(0x0402011030a00808ULL), M(0xb420804006200000ULL), M(0x200400c000080020ULL),
	M(0x0010282080042000ULL), M(0x00a0124444023001ULL), M(0x0404401401041003ULL),
	M(0x0240008244804022ULL), M(0x0100c40104078003ULL), M(0x8030200891004080ULL),
	M(0x0000088080001050ULL), M(0x040100c0c0810800ULL), M(0x00002042b1206824ULL),
	M(0x5000080090080880ULL), M(0x0c41060048400500ULL), M(0x100000020d008480ULL),
	M(0x00020040420084a5ULL), M(0x1000040d00800140ULL), M(0x0080000281428220ULL),
	M(0x0380000010200030ULL), M(0x8648021400108008ULL), M(0x0200008020093004ULL),
	M(0x0128038800620001ULL), M(0x0400000608521001ULL), M(0x4002004000012852ULL),
	M(0x2000002000003001ULL), M(0xa000000000014400ULL), M(0x086080010002c000ULL),
	M(0x1400200090082006ULL), M(0x3400080080804840ULL), M(0x0100000228400840ULL),
	M(0x008081020f800c44ULL), M(0x4444010003082200ULL), M(0x822018a000905166ULL),
	M(0x0808000000004080ULL), M(0x0088414060810040ULL), M(0x0004004402000120ULL),
	M(0x04c400003a008018ULL), M(0x0008880124000004ULL), M(0x008080100824022cULL),
	M(0x00a8824101000041ULL), M(0x0082200880400421ULL), M(0x0000200000000a05ULL),
	M(0x082008000000c924ULL), M(0x60080c0001000006ULL), M(0x110102061c027c00ULL),
	M(0x0400220110200008ULL), M(0x1100048008804800ULL), M(0x02000311a8008110ULL),
	M(0x010008a000002810ULL), M(0x0480801001008400ULL), M(0x0840910800040800ULL),
	M(0x101040020a000020ULL), M(0x4508000c0000000bULL), M(0x00080003000080c0ULL),
	M(0x3182030100189000ULL), M(0x0802000242000000ULL), M(0x0882200240000201ULL),
	M(0x0288490020200100ULL), M(0x0840208010000002ULL), M(0x0500202020010844ULL),
	M(0x4100021503044001ULL), M(0xa100044202300400ULL), M(0x2240020001000880ULL),
	M(0x8040020000c00180ULL), M(0x0000010080201000ULL), M(0x0000028280100020ULL),
	M(0x0000c48a08104082ULL), M(0x8040201000480004ULL), M(0x040100a004040000ULL),
	M(0x0804000400409a00ULL), M(0x40005050c0084000ULL), M(0x0302040100020000ULL),
	M(0x8c06204240108000ULL), M(0x0828c00882080804ULL), M(0x00110000c4000003ULL),
	M(0x1082050248121420ULL), M(0x0008014922000000ULL), M(0x013000000420080cULL),
};

#undef M

constexpr size_t KNIGHT_ATTACK_NB = 8;
constexpr size_t BISHOP_ATTACK_NB = 4;
constexpr size_t ADVISOR_ATTACK_NB = 4;

extern Bitboard FILE_BLOCK_MASK[FILE_NB];
extern Bitboard RANK_BLOCK_MASK[RANK_NB];
extern Bitboard BISHOP_BLOCK_MASK[BISHOP_SQUARE_NB * 2];
extern Bitboard KNIGHT_BLOCK_MASK[SQUARE_NB];
extern Bitboard KNIGHT_ATTACKED_MASK[SQUARE_NB];
extern Bitboard BISHOP_ATTACK_BB[BISHOP_SQUARE_NB * 2][pow_2(BISHOP_ATTACK_NB)];
extern Bitboard KNIGHT_ATTACK_BB[SQUARE_NB][pow_2(KNIGHT_ATTACK_NB / 2)]; // one blocker blocks 2 knight moves
extern Bitboard KNIGHT_ATTACKED_BB[SQUARE_NB][pow_2(KNIGHT_ATTACK_NB / 2)];
extern Bitboard ROOK_RANK_ATTACK_TABLE[SQUARE_NB][pow_2(FILE_NB - 2)];
extern Bitboard ROOK_FILE_ATTACK_TABLE[SQUARE_NB][pow_2(RANK_NB - 2)];
extern Bitboard CANNON_RANK_ATTACK_TABLE[SQUARE_NB][pow_2(FILE_NB - 2)];
extern Bitboard CANNON_FILE_ATTACK_TABLE[SQUARE_NB][pow_2(RANK_NB - 2)];

extern Bitboard KING_ATTACK[KING_SQUARE_NB * 2];
extern Bitboard ADVISOR_ATTACK[ADVISOR_SQUARE_NB * 2];
extern Bitboard PAWN_ATTACK[SQUARE_NB][COLOR_NB];
extern Bitboard PAWN_ATTACKED[SQUARE_NB][COLOR_NB];
extern Bitboard BISHOP_ATTACK_NO_MASK[BISHOP_SQUARE_NB * 2];
extern Bitboard KNIGHT_ATTACK_NO_MASK[SQUARE_NB];

extern Bitboard RANK_BETWEEN_MASK[FILE_NB][ceil_to_power_of_2(static_cast<size_t>(FILE_NB))];
extern Bitboard FILE_BETWEEN_MASK[RANK_NB][ceil_to_power_of_2(static_cast<size_t>(RANK_NB))];

NODISCARD INLINE constexpr uint64_t apply_magic(const Bitboard& a, const Attack_Magic b)
{
	return (a[WHITE] ^ a[BLACK]) * b;
}

NODISCARD INLINE Bitboard file_between_bb(Square sq1, Square sq2)
{
	ASSERT(sq_equal_file(sq1, sq2));
	return (FILE_BETWEEN_MASK[sq_rank(sq1)][sq_rank(sq2)] & square_file_bb(sq1));
}

NODISCARD INLINE Bitboard rank_between_bb(Square sq1, Square sq2)
{
	ASSERT(sq_equal_rank(sq1, sq2));
	return (RANK_BETWEEN_MASK[sq_file(sq1)][sq_file(sq2)] & square_rank_bb(sq1));
}

NODISCARD INLINE Bitboard sq_between_bb(Square sq1, Square sq2)
{
	ASSERT(sq_equal_file(sq1, sq2) || sq_equal_rank(sq1, sq2));

	return 
		sq_equal_rank(sq1, sq2)
		? rank_between_bb(sq1, sq2)
		: file_between_bb(sq1, sq2);
}

NODISCARD INLINE Square knight_move_blocker(Square from, Square to)
{
	return from + KNIGHT_BLOCKER_OFFSET[static_cast<int>(to) - static_cast<int>(from) + 20];
}

NODISCARD INLINE bool may_block_knight_for_king(Square leg, Square k_pos)
{
	ASSERT(king_area_bb().has_square(k_pos));
	const auto dlt = std::abs(leg - k_pos);
	return (dlt == 8 || dlt == 10); // diagonal directions from k_pos
}

NODISCARD INLINE const Bitboard& rook_rank_attack_bb(Square sq, const Bitboard& block)
{
	const size_t ix = (block[sq_color(sq)] >> RANK_SHIFT_RIGHT[sq_rank(sq)]) % pow_2(FILE_NB - 2);
	return ROOK_RANK_ATTACK_TABLE[sq][ix];
}

NODISCARD INLINE const Bitboard& cannon_rank_attack_bb(Square sq, const Bitboard& block)
{
	const size_t ix = (block[sq_color(sq)] >> RANK_SHIFT_RIGHT[sq_rank(sq)]) % pow_2(FILE_NB - 2);
	return CANNON_RANK_ATTACK_TABLE[sq][ix];
}

NODISCARD INLINE const Bitboard& rook_file_attack_bb(Square sq, const Bitboard& block)
{
	const size_t ix = apply_magic(block & FILE_BLOCK_MASK[sq_file(sq)], FILE_MAGIC[sq_file(sq)]) >> FILE_SHIFT;
	return ROOK_FILE_ATTACK_TABLE[sq][ix];
}

NODISCARD INLINE const Bitboard& cannon_file_attack_bb(Square sq, const Bitboard& block)
{
	const size_t ix = apply_magic(block & FILE_BLOCK_MASK[sq_file(sq)], FILE_MAGIC[sq_file(sq)]) >> FILE_SHIFT;
	return CANNON_FILE_ATTACK_TABLE[sq][ix];
}

NODISCARD INLINE const Bitboard& knight_attack_bb(Square sq, const Bitboard& block)
{
	const size_t ix = apply_magic(block & KNIGHT_BLOCK_MASK[sq], KNIGHT_MAGIC[sq]) >> KNIGHT_SHIFT;
	return KNIGHT_ATTACK_BB[sq][ix];
}

NODISCARD INLINE const Bitboard& bishop_attack_bb(Square sq, const Bitboard& block)
{
	ASSERT(is_bishop_pos(sq));
	const size_t ix = apply_magic(block & BISHOP_BLOCK_MASK[bishop_pos_index(sq)], BISHOP_MAGIC[bishop_pos_index(sq)]) >> BISHOP_SHIFT;
	return BISHOP_ATTACK_BB[bishop_pos_index(sq)][ix];
}

NODISCARD INLINE const Bitboard& knight_attacked_bb(Square sq, const Bitboard& block)
{
	const size_t ix = apply_magic(block & KNIGHT_ATTACKED_MASK[sq], BY_KNIGHT_MAGIC[sq]) >> KNIGHT_SHIFT;
	return KNIGHT_ATTACKED_BB[sq][ix];
}

NODISCARD INLINE const Bitboard& pawn_attack_bb(Square sq, Color color)
{
	return PAWN_ATTACK[sq][color];
}

NODISCARD INLINE const Bitboard& pawn_attacked_bb(Square sq, Color color)
{
	return PAWN_ATTACKED[sq][color];
}

NODISCARD INLINE const Bitboard& advisor_attack_bb(Square sq)
{
	return ADVISOR_ATTACK[advisor_pos_index(sq)];
}

NODISCARD INLINE const Bitboard& king_attack_bb(Square sq)
{
	return KING_ATTACK[king_pos_index(sq)];
}

NODISCARD INLINE const Bitboard& bishop_att_no_mask(Square sq)
{
	ASSERT(is_bishop_pos(sq));
	return BISHOP_ATTACK_NO_MASK[bishop_pos_index(sq)];
}

NODISCARD INLINE const Bitboard& knight_att_no_mask(Square sq)
{
	return KNIGHT_ATTACK_NO_MASK[sq];
}

NODISCARD INLINE Bitboard rook_attack_bb(Square sq, const Bitboard& block)
{
	return (rook_rank_attack_bb(sq, block) | rook_file_attack_bb(sq, block));
}

NODISCARD INLINE Bitboard cannon_attack_bb(Square sq, const Bitboard& block)
{
	return (cannon_rank_attack_bb(sq, block) | cannon_file_attack_bb(sq, block));
}

extern void attack_init();
