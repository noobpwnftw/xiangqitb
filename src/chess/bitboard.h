#pragma once

#include "chess.h"

#include "util/intrin.h"
#include "util/defines.h"
#include "util/enum.h"
#include "util/init.h"

#include <cstdint>

// The bitboard is 128 bits but there are only 90 squares. Offsets are used
// for each bitboard half to ensure a contiguous range of bits is used for
// the squares while having each board half correspond to a bitboard half.
constexpr int8_t SQ_LSB_INC[COLOR_NB] = { FIRST_BLACK_SQUARE - 64, FIRST_BLACK_SQUARE };

enum Bitboard_Half : uint64_t { 
	EMPTY_BITBOARD_HALF = 0,
	WHITE_BITBOARD_HALF = 0xfffffffffff80000ull,
	BLACK_BITBOARD_HALF = 0x00001fffffffffffull,
};

ENUM_ENABLE_OPERATOR_OR(Bitboard_Half);
ENUM_ENABLE_OPERATOR_OR_EQ(Bitboard_Half);
ENUM_ENABLE_OPERATOR_XOR(Bitboard_Half);
ENUM_ENABLE_OPERATOR_XOR_EQ(Bitboard_Half);
ENUM_ENABLE_OPERATOR_AND(Bitboard_Half);
ENUM_ENABLE_OPERATOR_AND_EQ(Bitboard_Half);
ENUM_ENABLE_OPERATOR_RSHIFT(Bitboard_Half);
ENUM_ENABLE_OPERATOR_RSHIFT_EQ(Bitboard_Half);
ENUM_ENABLE_OPERATOR_LSHIFT(Bitboard_Half);
ENUM_ENABLE_OPERATOR_LSHIFT_EQ(Bitboard_Half);
ENUM_ENABLE_OPERATOR_INV(Bitboard_Half);

NODISCARD INLINE Square peek_first_square(const Bitboard_Half& b, Color color)
{
	return static_cast<Square>(lsb(b) + SQ_LSB_INC[color]);
}

NODISCARD INLINE Square pop_first_square(Bitboard_Half& b, Color color)
{
	const Square r = static_cast<Square>(lsb(b) + SQ_LSB_INC[color]);
	b &= static_cast<Bitboard_Half>(b - 1);
	return r;
}

// Represents a set of squares on a xiangqi board.
struct Bitboard
{
	INLINE static constexpr Bitboard make_board_mask()
	{
		return Bitboard(WHITE_BITBOARD_HALF, BLACK_BITBOARD_HALF);
	}

	INLINE static constexpr Bitboard make_empty()
	{
		return Bitboard(EMPTY_BITBOARD_HALF, EMPTY_BITBOARD_HALF);
	}

	// NOTE: Bitboards are not initialized by default for performance reasons.
	//       Use Bitboard::make_empty() if you want an empty bitboard.
	INLINE Bitboard() = default;

	INLINE constexpr Bitboard(Const_Span<Square> squares) :
		m_halves{ EMPTY_BITBOARD_HALF, EMPTY_BITBOARD_HALF }
	{
		for (const Square sq : squares)
			*this |= sq;
	}

	INLINE constexpr Bitboard(const Bitboard& bb) = default;
	INLINE constexpr Bitboard(Bitboard&& bb) noexcept = default;

	INLINE constexpr Bitboard(const Bitboard_Half& a, const Bitboard_Half& b) :
		m_halves{ a, b }
	{
	}

	INLINE constexpr Bitboard& operator=(const Bitboard& bb) = default;
	INLINE constexpr Bitboard& operator=(Bitboard&& bb) noexcept = default;

	INLINE constexpr Bitboard& operator&= (const Bitboard& bb)
	{
		m_halves[0] &= bb.m_halves[0];
		m_halves[1] &= bb.m_halves[1];
		return *this;
	}

	INLINE constexpr Bitboard& operator|= (const Bitboard& bb)
	{
		m_halves[0] |= bb.m_halves[0];
		m_halves[1] |= bb.m_halves[1];
		return *this;
	}

	INLINE constexpr Bitboard& operator^= (const Bitboard& bb)
	{
		m_halves[0] ^= bb.m_halves[0];
		m_halves[1] ^= bb.m_halves[1];
		return *this;
	}

	NODISCARD INLINE friend constexpr Bitboard operator&(const Bitboard& a, const Bitboard& b)
	{
		return Bitboard(a.m_halves[0] & b.m_halves[0], a.m_halves[1] & b.m_halves[1]);
	}

	NODISCARD INLINE friend constexpr Bitboard operator|(const Bitboard& a, const Bitboard& b)
	{
		return Bitboard(a.m_halves[0] | b.m_halves[0], a.m_halves[1] | b.m_halves[1]);
	}

	NODISCARD INLINE friend constexpr Bitboard operator^(const Bitboard& a, const Bitboard& b)
	{
		return Bitboard(a.m_halves[0] ^ b.m_halves[0], a.m_halves[1] ^ b.m_halves[1]);
	}

	NODISCARD INLINE friend constexpr Bitboard operator~(const Bitboard& a)
	{
		return Bitboard(~a.m_halves[0], ~a.m_halves[1]);
	}

	INLINE constexpr Bitboard& operator|= (Square sq);
	INLINE constexpr Bitboard& operator&= (Square sq);
	INLINE constexpr Bitboard& operator^= (Square sq);

	INLINE constexpr Bitboard& operator|= (Rank sq);
	INLINE constexpr Bitboard& operator&= (Rank sq);
	INLINE constexpr Bitboard& operator^= (Rank sq);

	INLINE constexpr Bitboard& operator|= (File sq);
	INLINE constexpr Bitboard& operator&= (File sq);
	INLINE constexpr Bitboard& operator^= (File sq);

	template <typename IntT>
	NODISCARD INLINE friend Bitboard operator>>(const Bitboard& bb, IntT bit)
	{
		ASSUME(bit >= 0 && bit <= 128);
		return
			bit >= 64
			? Bitboard(bb[BLACK] >> (bit - 64), EMPTY_BITBOARD_HALF)
			: Bitboard(static_cast<Bitboard_Half>(shiftright128(bb[WHITE], bb[BLACK], static_cast<uint8_t>(bit))), bb[BLACK] >> bit);
	}

	template <typename IntT>
	NODISCARD INLINE friend Bitboard operator<<(const Bitboard& bb, IntT bit)
	{
		ASSUME(bit >= 0 && bit <= 128);
		return
			bit >= 64
			? Bitboard(EMPTY_BITBOARD_HALF, (bb[WHITE] << (bit - 64)))
			: Bitboard((bb[WHITE] << bit), static_cast<Bitboard_Half>(shiftleft128(bb[WHITE], bb[BLACK], static_cast<uint8_t>(bit))));
	}

	NODISCARD INLINE friend constexpr bool operator==(const Bitboard& a, const Bitboard& b) noexcept
	{
		return (a.m_halves[0] == b.m_halves[0] && a.m_halves[1] == b.m_halves[1]);
	}

	NODISCARD INLINE friend constexpr bool operator!=(const Bitboard& a, const Bitboard& b) noexcept
	{
		return !(a == b);
	}

	NODISCARD INLINE friend constexpr bool operator<(const Bitboard& a, const Bitboard& b) noexcept
	{
		if (a.m_halves[1] != b.m_halves[1])
			return a.m_halves[1] < b.m_halves[1];
		return a.m_halves[0] < b.m_halves[0];
	}

	NODISCARD INLINE friend constexpr bool operator>(const Bitboard& a, const Bitboard& b) noexcept
	{
		if (a.m_halves[1] != b.m_halves[1])
			return a.m_halves[1] > b.m_halves[1];
		return a.m_halves[0] > b.m_halves[0];
	}

	// Returns a half of the bitboard corresponding to the given side.
	NODISCARD INLINE constexpr Bitboard_Half operator[](Color index) const
	{
		ASSUME(index == WHITE || index == BLACK);
		return m_halves[index];
	}

	NODISCARD INLINE constexpr operator bool() const
	{
		return (m_halves[0] | m_halves[1]) != 0;
	}

	NODISCARD INLINE constexpr bool empty() const
	{
		return (m_halves[0] | m_halves[1]) == 0;

	}

	NODISCARD INLINE size_t peek_1st_bit() const
	{
		return 
			m_halves[0]
			? lsb(m_halves[0])
			: lsb(m_halves[1]) + 64;
	}

	// Returns a copy of this bitboard with squares in each file mirrored.
	NODISCARD Bitboard mirror_files() const;

	// Returns a copy of this bitboard with squares in each file mirrored
	// if mirr == true. Otherwise returns a copy of this bitboard.
	NODISCARD Bitboard maybe_mirror_files(bool mirr) const
	{
		return mirr ? mirror_files() : *this;
	}

	// Returns and pops the first lowest set bit. Used for attack generation.
	INLINE size_t pop_1st_bit()
	{
		size_t sq;
		if (m_halves[0])
		{
			sq = lsb(m_halves[0]);
			m_halves[0] &= static_cast<Bitboard_Half>(m_halves[0] - 1);
		}
		else
		{
			sq = lsb(m_halves[1]) + 64;
			m_halves[1] &= static_cast<Bitboard_Half>(m_halves[1] - 1);
		}
		return sq;
	}

	NODISCARD INLINE size_t num_set_bits() const
	{
		return popcnt(m_halves[0]) + popcnt(m_halves[1]);
	}

	NODISCARD INLINE bool has_only_one_set_bit() const
	{
		return num_set_bits() == 1;
	}

	// Sets a raw bit. Used for attack generation.
	INLINE Bitboard& set_bit(size_t idx)
	{
		if (idx < 64)
			m_halves[0] |= static_cast<Bitboard_Half>(1ULL << idx);
		else
			m_halves[1] |= static_cast<Bitboard_Half>(1ULL << (idx - 64));
		return *this;
	}

	NODISCARD INLINE constexpr bool has_square(Square sq) const
	{
		ASSERT(sq >= SQ_START && sq < SQ_END);
		if (sq < FIRST_BLACK_SQUARE)
			return (m_halves[0] & static_cast<Bitboard_Half>(1ULL << (sq - SQ_LSB_INC[WHITE]))) != 0;
		else
			return (m_halves[1] & static_cast<Bitboard_Half>(1ULL << (sq - SQ_LSB_INC[BLACK]))) != 0;
	}

	// Returns and pops the lowest present square.
	INLINE Square pop_first_square()
	{
		Square sq;
		if (m_halves[0])
		{
			sq = static_cast<Square>(lsb(m_halves[0]) + SQ_LSB_INC[WHITE]);
			m_halves[0] &= static_cast<Bitboard_Half>(m_halves[0] - 1);
		}
		else
		{
			sq = static_cast<Square>(lsb(m_halves[1]) + SQ_LSB_INC[BLACK]);
			m_halves[1] &= static_cast<Bitboard_Half>(m_halves[1] - 1);
		}
		return sq;
	}

	// Returns the lowest present square.
	NODISCARD INLINE Square peek_first_square() const
	{
		return 
			m_halves[0]
			? static_cast<Square>(lsb(m_halves[0]) + SQ_LSB_INC[WHITE])
			: static_cast<Square>(lsb(m_halves[1]) + SQ_LSB_INC[BLACK]);
	}

	// Returns and pops the highest present square.
	INLINE Square pop_last_square()
	{
		Square sq;
		if (m_halves[1])
		{
			sq = static_cast<Square>(msb(m_halves[1]));
			m_halves[1] ^= static_cast<Bitboard_Half>(1ULL << sq);
			return sq + SQ_LSB_INC[BLACK];
		}
		else
		{
			sq = static_cast<Square>(msb(m_halves[0]));
			m_halves[0] ^= static_cast<Bitboard_Half>(1ULL << sq);
			return sq + SQ_LSB_INC[WHITE];
		}
	}

	// Returns the highest present square.
	NODISCARD INLINE Square peek_last_square() const
	{
		return 
			m_halves[1]
			? static_cast<Square>(msb(m_halves[1]) + SQ_LSB_INC[BLACK])
			: static_cast<Square>(msb(m_halves[0]) + SQ_LSB_INC[WHITE]);
	}

	// Clears all bits in this bitboard.
	INLINE constexpr void clear()
	{
		m_halves[0] = m_halves[1] = EMPTY_BITBOARD_HALF;
	}

	void display() const;

private:
	Bitboard_Half m_halves[2];
}; 

constexpr std::array<Bitboard, SQUARE_NB> SQ_BB_MASK = []() {
	auto res = make_filled_array<SQUARE_NB, Bitboard>(Bitboard::make_empty());
	for (Square sq = SQ_A0; sq < SQUARE_NB; ++sq)
	{
		if (sq < FIRST_BLACK_SQUARE)
			res[sq] = Bitboard(static_cast<Bitboard_Half>(1ULL << (sq - SQ_LSB_INC[WHITE])), EMPTY_BITBOARD_HALF);
		else
			res[sq] = Bitboard(EMPTY_BITBOARD_HALF, static_cast<Bitboard_Half>(1ULL << (sq - SQ_LSB_INC[BLACK])));
	}
	return res;
}();

NODISCARD INLINE constexpr const Bitboard& square_bb(Square sq)
{
	return SQ_BB_MASK[sq];
}

NODISCARD INLINE constexpr Bitboard operator&(const Bitboard& a, Square sq)
{
	return a & square_bb(sq);
}

NODISCARD INLINE constexpr Bitboard operator|(const Bitboard& a, Square sq)
{
	return a | square_bb(sq);
}

NODISCARD INLINE constexpr Bitboard operator^(const Bitboard& a, Square sq)
{
	return a ^ square_bb(sq);
}

constexpr Bitboard& Bitboard::operator|= (Square sq)
{
	return *this = *this | sq;
}

constexpr Bitboard& Bitboard::operator&= (Square sq)
{
	return *this = *this & sq;
}

constexpr Bitboard& Bitboard::operator^= (Square sq)
{
	return *this = *this ^ sq;
}

constexpr std::array<Bitboard, RANK_NB> RANK_BB_MASK = []() {
	auto res = make_filled_array<RANK_NB, Bitboard>(Bitboard::make_empty());
	for (Square sq = SQ_A0; sq < SQUARE_NB; ++sq)
		res[sq_rank(sq)] |= sq;
	return res;
}();

NODISCARD INLINE constexpr const Bitboard& rank_bb(Rank r)
{
	return RANK_BB_MASK[r];
}

NODISCARD INLINE const Bitboard& square_rank_bb(Square sq)
{
	return rank_bb(sq_rank(sq));
}

NODISCARD INLINE constexpr Bitboard operator&(const Bitboard& a, Rank r)
{
	return a & rank_bb(r);
}

NODISCARD INLINE constexpr Bitboard operator|(const Bitboard& a, Rank r)
{
	return a | rank_bb(r);
}

NODISCARD INLINE constexpr Bitboard operator^(const Bitboard& a, Rank r)
{
	return a ^ rank_bb(r);
}

constexpr std::array<Bitboard, FILE_NB> FILE_BB_MASK = []() {
	auto res = make_filled_array<FILE_NB, Bitboard>(Bitboard::make_empty());
	for (Square sq = SQ_A0; sq < SQUARE_NB; ++sq)
		res[sq_file(sq)] |= sq;
	return res;
}();

NODISCARD INLINE constexpr const Bitboard& file_bb(File f)
{
	return FILE_BB_MASK[f];
}

NODISCARD INLINE const Bitboard& square_file_bb(Square sq)
{
	return file_bb(sq_file(sq));
}

NODISCARD INLINE constexpr Bitboard operator&(const Bitboard& a, File f)
{
	return a & file_bb(f);
}

NODISCARD INLINE constexpr Bitboard operator|(const Bitboard& a, File f)
{
	return a | file_bb(f);
}

NODISCARD INLINE constexpr Bitboard operator^(const Bitboard& a, File f)
{
	return a ^ file_bb(f);
}

constexpr Bitboard& Bitboard::operator|= (Rank r)
{
	return *this = *this | r;
}

constexpr Bitboard& Bitboard::operator&= (Rank r)
{
	return *this = *this & r;
}

constexpr Bitboard& Bitboard::operator^= (Rank r)
{
	return *this = *this ^ r;
}

constexpr Bitboard& Bitboard::operator|= (File f)
{
	return *this = *this | f;
}

constexpr Bitboard& Bitboard::operator&= (File f)
{
	return *this = *this & f;
}

constexpr Bitboard& Bitboard::operator^= (File f)
{
	return *this = *this ^ f;
}

INLINE constexpr Bitboard operator|(Square lhs, Square rhs)
{
	return square_bb(lhs) | square_bb(rhs);
}

INLINE constexpr Bitboard operator&(Square lhs, Square rhs)
{
	return square_bb(lhs) & square_bb(rhs);
}

INLINE constexpr Bitboard operator^(Square lhs, Square rhs)
{
	return square_bb(lhs) ^ square_bb(rhs);
}

INLINE constexpr Bitboard operator~(Square sq)
{
	return ~square_bb(sq);
}

INLINE constexpr Bitboard operator|(Rank lhs, Rank rhs)
{
	return rank_bb(lhs) | rank_bb(rhs);
}

INLINE constexpr Bitboard operator&(Rank lhs, Rank rhs)
{
	return rank_bb(lhs) & rank_bb(rhs);
}

INLINE constexpr Bitboard operator^(Rank lhs, Rank rhs)
{
	return rank_bb(lhs) ^ rank_bb(rhs);
}

INLINE constexpr Bitboard operator~(Rank sq)
{
	return ~rank_bb(sq);
}

INLINE constexpr Bitboard operator|(File lhs, File rhs)
{
	return file_bb(lhs) | file_bb(rhs);
}

INLINE constexpr Bitboard operator&(File lhs, File rhs)
{
	return file_bb(lhs) & file_bb(rhs);
}

INLINE constexpr Bitboard operator^(File lhs, File rhs)
{
	return file_bb(lhs) ^ file_bb(rhs);
}

INLINE constexpr Bitboard operator~(File sq)
{
	return ~file_bb(sq);
}

constexpr Bitboard KING_AREA_BB =
	(RANK_0 | RANK_1 | RANK_2 | RANK_7 | RANK_8 | RANK_9) & (FILE_D | FILE_E | FILE_F);

NODISCARD INLINE constexpr const Bitboard& king_area_bb()
{
	return KING_AREA_BB;
}

constexpr Bitboard ADVISOR_AREA_BB =
	(  ((RANK_0 | RANK_2 | RANK_7 | RANK_9) & (FILE_D | FILE_F))
	 | ((RANK_1 | RANK_8) & FILE_E));

NODISCARD INLINE constexpr const Bitboard& advisor_area_bb()
{
	return ADVISOR_AREA_BB;
}

constexpr Bitboard BISHOP_AREA_BB =
	(  ((RANK_0 | RANK_4 | RANK_5 | RANK_9) & (FILE_C | FILE_G))
	 | ((RANK_2 | RANK_7) & (FILE_A | FILE_E | FILE_I)));

NODISCARD INLINE constexpr const Bitboard& bishop_area_bb()
{
	return BISHOP_AREA_BB;
}

constexpr Bitboard PAWN_AREA_BB[COLOR_NB] = {
	  (RANK_5 | RANK_6 | RANK_7 | RANK_8 | RANK_9)
	| ((RANK_3 | RANK_4) & (FILE_A | FILE_C | FILE_E | FILE_G | FILE_I)),
	  (RANK_0 | RANK_1 | RANK_2 | RANK_3 | RANK_4)
	| ((RANK_5 | RANK_6) & (FILE_A | FILE_C | FILE_E | FILE_G | FILE_I))
};

NODISCARD INLINE constexpr const Bitboard& pawn_area_bb(Color color)
{
	return PAWN_AREA_BB[color];
}
