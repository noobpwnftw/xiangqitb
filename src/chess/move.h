#pragma once

#include "chess.h"

#include "util/defines.h"

#include <algorithm>

struct Move
{
	NODISCARD static Move make_null()
	{
		return Move(SQ_A0, SQ_A0);
	}

	NODISCARD static Move make_from_string(const char string[]);

	Move() = default;

	Move(Square from, Square to) :
		m_packed(static_cast<uint16_t>(((from) << 8) | (to)))
	{
	}

	NODISCARD bool is_ok() const;

	NODISCARD bool is_null() const
	{
		return m_packed == 0;
	}

	NODISCARD Square from() const
	{
		return static_cast<Square>(m_packed >> 8);
	}

	NODISCARD Square to() const
	{
		return static_cast<Square>(m_packed & 0xff);
	}

	void to_string(char string[]) const;

private:
	// We pack squares on each byte manually, because
	// we can't rely on the compilers to merge the load/stores.
	uint16_t m_packed;
};

// Stores a list of moves in the most efficient way possible.
// No heap allocation, takes 4 cache lines.
struct Move_List
{
	static constexpr size_t CAPACITY = 127;

	Move_List() :
		m_size(0)
	{
	}

	INLINE void clear()
	{
		m_size = 0;
	}

	INLINE void add(Move move)
	{
		ASSERT(m_size < CAPACITY);
		m_moves[m_size++] = move;
	}

	INLINE void pop_last()
	{
		ASSERT(m_size > 0);
		m_size -= 1;
	}

	INLINE void swap_with_last_and_pop(size_t idx)
	{
		ASSERT(idx < m_size);
		std::swap(m_moves[idx], m_moves[--m_size]);
	}

	NODISCARD INLINE size_t size() const
	{
		return m_size;
	}

	NODISCARD INLINE bool empty() const
	{
		return m_size == 0;
	}

	NODISCARD INLINE Move operator[](size_t pos) const
	{
		return m_moves[pos];
	}

	NODISCARD INLINE const Move* begin() const
	{
		return m_moves;
	}

	NODISCARD INLINE Move* begin()
	{
		return m_moves;
	}

	NODISCARD INLINE const Move* cbegin() const
	{
		return m_moves;
	}

	NODISCARD INLINE const Move* end() const
	{
		return m_moves + m_size;
	}

	NODISCARD INLINE Move* end()
	{
		return m_moves + m_size;
	}

	NODISCARD INLINE const Move* cend() const
	{
		return m_moves + m_size;
	}

	template <typename F>
	INLINE void remove_if(F&& func)
	{
		const auto new_end = std::remove_if(begin(), end(), std::forward<F>(func));
		m_size = static_cast<uint16_t>(std::distance(begin(), new_end));
	}

private:
	uint16_t m_size;
	Move m_moves[CAPACITY];
};
static_assert(sizeof(Move_List) == 256);

enum struct Move_Legality_Lower_Bound
{
	NONE,
	PSEUDO_LEGAL,
	LEGAL
};

extern void move_display(Move move);
