#include "move.h"

#include "chess.h"

#include <cstring>
#include <cstdio>

bool Move::is_ok() const
{
	return sq_is_ok(from()) && sq_is_ok(to()) && from() != to();
}

Move Move::make_from_string(const char string[])
{
	char tmp_string[3];

	tmp_string[0] = string[0];
	tmp_string[1] = string[1];
	tmp_string[2] = '\0';

	const Square from = square_from_string(tmp_string);
	if (from < 0)
		return Move::make_null();

	tmp_string[0] = string[2];
	tmp_string[1] = string[3];
	tmp_string[2] = '\0';

	const Square to = square_from_string(tmp_string);
	if (to < 0)
		return Move::make_null();

	return Move(from, to);
}

void Move::to_string(char string[]) const
{
	if (is_null())
	{
		strcpy(string, "NULL");
		return;
	}

	square_to_string(from(), &string[0]);
	square_to_string(to(), &string[2]);
}

void move_display(Move move)
{
	char t_string[16];
	move.to_string(t_string);
	printf("%s ", t_string);
}