#include "bitboard.h"

#include <cstdio>

void Bitboard::display() const
{
	for (size_t i = 0; i < 19; ++i)
	{
		printf(m_halves[0] & (1ULL << i) ? "1 " : "0 ");
		if (i % 9 == 0)
			printf("\n");
	}
	printf("******************\n");
	for (size_t i = 19; i < 64; ++i)
	{
		if (i != 19 && (i - 19) % 9 == 0)
			printf("\n");

		printf(m_halves[0] & (1ULL << i) ? "1 " : "0 ");
	}
	printf("\n==================");
	for (size_t i = 0; i < 45; ++i)
	{
		if (i % 9 == 0)
			printf("\n");

		printf(m_halves[1] & (1ULL << i) ? "1 " : "0 ");
	}
	printf("\n******************");
	for (size_t i = 45; i < 64; ++i)
	{
		if (i % 9 == 0)
			printf("\n");

		printf(m_halves[1] & (1ULL << i) ? "1 " : "0 ");
	}
	printf("\n############################\n");
}


Bitboard Bitboard::mirror_files() const
{
	Bitboard mirrored = Bitboard::make_empty();

	// TODO: a better solution
	Bitboard cpy = *this;
	while (cpy)
		mirrored |= sq_file_mirror(cpy.pop_first_square());

	return mirrored;
}
