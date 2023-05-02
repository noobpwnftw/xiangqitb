#pragma once

#include "chess.h"
#include "position.h"

#include "util/defines.h"
#include "util/enum.h"
#include "util/span.h"
#include "util/param.h"

#include <array>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <utility>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <map>
#include <set>
#include <memory>
#include <optional>

enum Piece_Type_Class : int8_t {
	DEFENDERS, ROOKS, KNIGHTS, CANNONS, PAWNS
};

// Pieces on the board are divided into smaller buckets, by their "class",
// to essentially add one layer to the board index computation. This makes
// it feasible to optimize the list of valid piece placements and allows
// easy reduction of redundant configurations (when two same pieces can exchange squares).
// Defender class includes a king + any number of advisors and bishops.
enum Piece_Class : int8_t {
	WHITE_DEFENDERS, WHITE_ROOKS, WHITE_KNIGHTS, WHITE_CANNONS, WHITE_PAWNS,
	BLACK_DEFENDERS, BLACK_ROOKS, BLACK_KNIGHTS, BLACK_CANNONS, BLACK_PAWNS,
	PIECE_CLASS_START = 0, PIECE_CLASS_END = 10, PIECE_CLASS_NONE = -1, PIECE_CLASS_NB = 10
};

constexpr Piece_Class& operator++(Piece_Class& p_class)
{
	p_class = static_cast<Piece_Class>(static_cast<int>(p_class) + 1);
	return p_class;
}

NODISCARD constexpr Piece_Class make_piece_class(Color color, Piece_Type_Class pt_class)
{
	ASSERT(pt_class < BLACK_DEFENDERS);
	return static_cast<Piece_Class>(pt_class + BLACK_DEFENDERS * color);
}

NODISCARD constexpr Piece_Class opp_piece_class(Piece_Class set)
{
	return static_cast<Piece_Class>(set < BLACK_DEFENDERS ? set + BLACK_DEFENDERS : set - BLACK_DEFENDERS);
}

NODISCARD constexpr Piece_Class maybe_opp_piece_class(Piece_Class set, bool mirror)
{
	return mirror ? opp_piece_class(set) : set;
}

NODISCARD constexpr Color piece_class_color(Piece_Class set)
{
	return set >= BLACK_DEFENDERS ? BLACK : WHITE;
}

// The map of a piece to its corresponding class.
constexpr std::array<Piece_Class, PIECE_NB> PIECE_TO_PIECE_CLASS = []() {
	std::array<Piece_Class, PIECE_NB> arr{};
	arr[WHITE_OCCUPY] = PIECE_CLASS_NONE;
	arr[WHITE_KING] = WHITE_DEFENDERS;
	arr[WHITE_ADVISOR] = WHITE_DEFENDERS;
	arr[WHITE_BISHOP] = WHITE_DEFENDERS;
	arr[WHITE_ROOK] = WHITE_ROOKS;
	arr[WHITE_KNIGHT] = WHITE_KNIGHTS;
	arr[WHITE_CANNON] = WHITE_CANNONS;
	arr[WHITE_PAWN] = WHITE_PAWNS;
	arr[BLACK_OCCUPY] = PIECE_CLASS_NONE;
	arr[BLACK_KING] = BLACK_DEFENDERS;
	arr[BLACK_ADVISOR] = BLACK_DEFENDERS;
	arr[BLACK_BISHOP] = BLACK_DEFENDERS;
	arr[BLACK_ROOK] = BLACK_ROOKS;
	arr[BLACK_KNIGHT] = BLACK_KNIGHTS;
	arr[BLACK_CANNON] = BLACK_CANNONS;
	arr[BLACK_PAWN] = BLACK_PAWNS;
	return arr;
}();

constexpr Piece_Class piece_class(Piece p)
{
	return PIECE_TO_PIECE_CLASS[p];
}

struct Unique_Piece_Configs;

// Represents a set of pieces on the board that is a valid configuration
// (a subset of the starting position). The set of pieces is stored in
// a cardinal order, including the order of sides based on piece strength.
struct Piece_Config
{
	static constexpr char VALID_PIECES[] = "KABCNPR";

	static constexpr std::array<int16_t, PIECE_NB> PIECE_STRENGTH_FOR_SIDE_ORDER = []() {
		std::array<int16_t, PIECE_NB> arr{}; 

		// These are either not present or present with a fixed amount, 
		// so the exact value doesn't matter.
		arr[WHITE_OCCUPY] = arr[BLACK_OCCUPY] = 0;
		arr[WHITE_KING] = arr[BLACK_KING] = 0;

		// All free attackers have scores higher than any combination of
		// non-free attackers (so they are a primary order). 
		// This means that if there's only one free attacker on the 
		// board it will always be on the white's side.
		arr[WHITE_ROOK] = arr[BLACK_ROOK] = 4000;
		arr[WHITE_KNIGHT] = arr[BLACK_KNIGHT] = 600;
		arr[WHITE_CANNON] = arr[BLACK_CANNON] = 603;
		arr[WHITE_PAWN] = arr[BLACK_PAWN] = 80;

		// Non-free attackers. Secondary in order.
		arr[WHITE_ADVISOR] = arr[BLACK_ADVISOR] = 11;
		arr[WHITE_BISHOP] = arr[BLACK_BISHOP] = 10;

		return arr;
	}();

	static constexpr std::array<int8_t, PIECE_NB> PIECE_ORDER = []() {
		std::array<int8_t, PIECE_NB> ret{};
		ret[WHITE_OCCUPY] = 0;
		ret[BLACK_OCCUPY] = 0;
		int8_t i = 1;
		constexpr Color AllColors[] = { WHITE, BLACK };
		for (const Color color : AllColors)
		{
			ret[piece_make(color, KING)] = i++;
			ret[piece_make(color, ADVISOR)] = i++;
			ret[piece_make(color, BISHOP)] = i++;
			ret[piece_make(color, ROOK)] = i++;
			ret[piece_make(color, KNIGHT)] = i++;
			ret[piece_make(color, CANNON)] = i++;
			ret[piece_make(color, PAWN)] = i++;
		}
		return ret;
	}();

	static void sort_pieces(Span<Piece> pieces);

	NODISCARD static bool is_constructible_from(const std::string& name)
	{
		if (name.empty() || name.size() > MAX_MAN)
			return false;

		if (name[0] != 'K')
			return false;

		if (std::count(name.begin(), name.end(), 'K') != 2)
			return false;

		if (name.find_first_not_of(VALID_PIECES) != std::string::npos)
			return false;

		return true;
	}

	NODISCARD static bool is_constructible_from(Const_Span<Piece> pieces)
	{
		if (pieces.size() < 2 || pieces.size() > MAX_MAN)
			return false;

		if (std::count(pieces.begin(), pieces.end(), WHITE_KING) != 1)
			return false;

		if (std::count(pieces.begin(), pieces.end(), BLACK_KING) != 1)
			return false;

		return true;
	}

	Piece_Config(const std::string& s) :
		m_num_pieces(0)
	{
		if (!is_constructible_from(s))
			throw std::runtime_error("Invalid PieceConfig: " + s);

		bool is_black = false;
		for (const char c : s)
		{
			const Piece_Type pt = piece_type(piece_from_char(c));
			if (m_num_pieces > 0 && pt == KING)
				is_black = true;

			const Piece p = piece_make(is_black ? BLACK : WHITE, pt);
			m_pieces[m_num_pieces++] = p;
		}

		sort_pieces(Span(m_pieces, m_num_pieces));

		// This needs to happen after the pieces are sorted, because sorting
		// can swap the piece colors.
		for (const Piece p : Const_Span(m_pieces, m_pieces + m_num_pieces))
		{
			m_base_mat_key.add_piece(p);
			m_mirr_mat_key.add_piece(piece_opp_color(p));
		}
	}

	Piece_Config(Const_Span<Piece> pcs) :
		m_num_pieces(0)
	{
		if (!is_constructible_from(pcs))
			throw std::runtime_error("Invalid PieceConfig.");

		std::memcpy(m_pieces, pcs.data(), pcs.size() * sizeof(Piece));

		sort_pieces(Span(m_pieces, pcs.size()));

		for (const Piece p : Const_Span(m_pieces, m_pieces + pcs.size()))
		{
			m_base_mat_key.add_piece(p);
			m_mirr_mat_key.add_piece(piece_opp_color(p));
		}
		m_num_pieces = pcs.size();
	}

	NODISCARD auto pieces() const
	{
		return Const_Span(m_pieces, m_pieces + m_num_pieces);
	}

	// Returns the uppercase name of the piece configuration.
	NODISCARD std::string name() const
	{
		std::string s;
		for (const Piece p : pieces())
			s += piece_type_to_char(piece_type(p));
		return s;
	}

	NODISCARD const std::array<size_t, PIECE_NB> piece_counts() const
	{
		std::array<size_t, PIECE_NB> counts;
		std::fill(counts.begin(), counts.end(), 0);
		for (const Piece piece : pieces())
			counts[piece] += 1;
		return counts;
	}

	NODISCARD bool operator==(const Piece_Config& other) const
	{
		return m_num_pieces == other.m_num_pieces
			&& std::equal(m_pieces, m_pieces + m_num_pieces, other.m_pieces);
	}

	// Returns whether the piece at index idx can be removed.
	// Returns false if king or out of range.
	NODISCARD bool can_remove_piece(size_t idx) const
	{
		return idx < m_num_pieces && piece_type(m_pieces[idx]) != KING;
	}

	NODISCARD size_t num_pieces() const
	{
		return m_num_pieces;
	}

	NODISCARD bool has_any_free_attackers(Color color) const
	{
		for (const Piece p : pieces())
			if (piece_color(p) == color && is_piece_free_attacker(p))
				return true;
		return false;
	}

	NODISCARD bool has_any_free_attackers() const
	{
		// With the way we order sides based on pieces they have
		// we have that if there are free attackers then at least white has some.
		return has_any_free_attackers(WHITE);
	}

	// Returns a copy of this configuration but with a piece at idx removed.
	// Throws an exception if the piece cannot be removed.
	NODISCARD Piece_Config with_removed_piece(size_t idx) const
	{
		if (!can_remove_piece(idx))
			throw std::runtime_error("Trying to remove a piece from PieceConfig.");

		Piece pcs_cpy[MAX_MAN];
		std::memcpy(pcs_cpy, m_pieces, idx * sizeof(Piece));
		std::memcpy(pcs_cpy + idx, m_pieces + idx + 1, (m_num_pieces - idx - 1) * sizeof(Piece));
		return Piece_Config(Span(pcs_cpy, m_num_pieces - 1));
	}

	// Returns whether removing a given piece (or constructing a Piece_Config
	// without said piece) would result in the sides to switch. 
	// This happens when the capture changes the relative strength of the sides.
	NODISCARD bool needs_mirror_after_capture(Piece cap_piece) const
	{
		// Since black's strength is always lower or equal to white's strength
		// it cannot become higher than white's strength by removing a piece.
		if (piece_color(cap_piece) == BLACK)
			return false;

		size_t score[COLOR_NB] = { 0, 0 };
		for (const Piece p : pieces())
			score[piece_color(p)] += PIECE_STRENGTH_FOR_SIDE_ORDER[p];

		ASSERT(score[WHITE] >= score[BLACK]);
		ASSERT(piece_color(cap_piece) == WHITE);

		return score[BLACK] > score[WHITE] - PIECE_STRENGTH_FOR_SIDE_ORDER[cap_piece];
	}

	// Returns all unique valid piece configurations that can be created
	// from this one by removing exaclty 1 piece.
	NODISCARD Unique_Piece_Configs sub_configs() const;

	// Same as sub_configs, but as a mapping of the removed piece.
	NODISCARD std::map<Piece, Piece_Config> sub_configs_by_capture() const;

	// Returns all unique valid piece configurations that can be created
	// from this one by removing any number of pieces. Includes this configuration.
	NODISCARD Unique_Piece_Configs closure() const;

	// Same as sub_configs but appends to an existing list instead.
	void add_sub_configs_to(Unique_Piece_Configs& pss) const;

	// Appends all unique valid piece configurations that can be created
	// from this one by removing any number of pieces. Includes this configuration.
	// The configurations are added in dependency order, so when a configuration
	// is to be added then all sub-configurations that it depends on have already been added.
	// If assume_contains_closures is true then all elements A that are already in pss
	// must also have all elements that are in A.closure() present. 
	// The implementaion skips doing redundant work in this case.
	void add_closure_in_dependency_order_to(Unique_Piece_Configs& pss, bool assume_contains_closures = false) const;

	// Returns the material key of this piece configuration.
	NODISCARD Material_Key base_material_key() const
	{
		return m_base_mat_key;
	}

	// Returns the material key of this piece configuration
	// and the material key of a this configuration with sides reversed (white <=> black).
	NODISCARD std::pair<Material_Key, Material_Key> material_keys() const
	{
		return { m_base_mat_key, m_mirr_mat_key };
	}

	// Returns the minimal of normal and mirrored material keys.
	NODISCARD Material_Key min_material_key() const
	{
		return std::min(m_base_mat_key, m_mirr_mat_key);
	}

private:
	Piece m_pieces[MAX_MAN];
	size_t m_num_pieces;
	Material_Key m_base_mat_key;
	Material_Key m_mirr_mat_key;
};

// A container that preserves stores piece configurations.
// It preserves the insertion order and ensures that all contained
// piece configurations are unique.
struct Unique_Piece_Configs
{
	using Container_Type = std::vector<Piece_Config>;
	using iterator = typename Container_Type::iterator;
	using const_iterator = typename Container_Type::const_iterator;
	using reverse_iterator = typename Container_Type::reverse_iterator;
	using const_reverse_iterator = typename Container_Type::const_reverse_iterator;

	NODISCARD const Piece_Config& operator[](size_t idx) const
	{
		return m_piece_sets[idx];
	}

	void clear()
	{
		m_mat_keys.clear();
		m_piece_sets.clear();
	}

	NODISCARD bool contains(const Piece_Config& ps) const
	{
		return m_mat_keys.find(ps.base_material_key()) != m_mat_keys.end();
	}

	void add_unique(Piece_Config ps)
	{
		if (!contains(ps))
		{
			m_mat_keys.insert(ps.base_material_key());
			m_piece_sets.emplace_back(std::move(ps));
		}
	}

	void add_unique(const Unique_Piece_Configs& pss)
	{
		for (const auto& ps : pss)
			add_unique(ps);
	}

	// Tries to remove a given piece configuration. Does nothing if it does not exist.
	void remove(const Piece_Config& ps)
	{
		auto iter = std::find(m_piece_sets.begin(), m_piece_sets.end(), ps);
		if (iter != m_piece_sets.end())
		{
			m_mat_keys.erase(ps.base_material_key());
			m_piece_sets.erase(iter);
		}
	}

	// Removes all piece configurations for which the predicate is true.
	template <typename FuncT>
	void remove_if(FuncT&& f)
	{
		auto new_end = std::remove_if(m_piece_sets.begin(), m_piece_sets.end(), std::forward<FuncT>(f));
		for (auto it = new_end; it != m_piece_sets.end(); ++it)
			m_mat_keys.erase(it->get_material_key());
		m_piece_sets.erase(new_end, m_piece_sets.end());
	}

	NODISCARD size_t size() const
	{
		return m_piece_sets.size();
	}

	NODISCARD bool empty() const
	{
		return m_piece_sets.empty();
	}

	NODISCARD const_iterator begin() const { return m_piece_sets.begin(); }
	NODISCARD const_iterator cbegin() const { return m_piece_sets.cbegin(); }
	NODISCARD const_iterator end() const { return m_piece_sets.end(); }
	NODISCARD const_iterator cend() const { return m_piece_sets.cend(); }

	NODISCARD const_reverse_iterator rbegin() const { return m_piece_sets.rbegin(); }
	NODISCARD const_reverse_iterator crbegin() const { return m_piece_sets.crbegin(); }
	NODISCARD const_reverse_iterator rend() const { return m_piece_sets.rend(); }
	NODISCARD const_reverse_iterator crend() const { return m_piece_sets.crend(); }

private:
	Container_Type m_piece_sets;
	std::set<Material_Key> m_mat_keys; // optimization for presence lookup
};
