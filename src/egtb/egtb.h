#pragma once

#include "chess/piece_config.h"

#include "util/defines.h"
#include "util/span.h"
#include "util/fixed_vector.h"
#include "util/enum.h"
#include "util/filesystem.h"
#include "util/utility.h"

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <mutex>
#include <immintrin.h>
#include <emmintrin.h>
#include <variant>

// If USE_LARGE_INDEX is defined then Piece_Group::Placement_Index 
// will use 32-bit indices instead of 16-bit indices.
// This allows more piece groups to be representible, for example
// PPPP, PPPPP. Note that these may take a VERY long time to initialize
// and have larger memory footprint.
// #define USE_LARGE_INDEX

// Magic values used for marking the EGTB files.
enum struct EGTB_Magic : uint64_t
{
	WDL_MAGIC = 0x7550918f,
	DTC_MAGIC = 0xb19122de,
	DTM_MAGIC = 0xc7b382a6,
};

// Each EGTB has at most two tables, one per color.
// This function converts the number of tables to the list of colors of these tables.
NODISCARD inline Fixed_Vector<Color, 2> egtb_table_colors(size_t table_num)
{
	ASSERT(table_num <= COLOR_NB);
	Fixed_Vector<Color, 2> table_colors;
	table_colors.emplace_back(WHITE);
	if (table_num == 2)
		table_colors.emplace_back(BLACK);
	return table_colors;
}

// Represents a group of pieces (subset of piece configuration) of specific class.
// See Piece_Class declaration for more information on what can form a piece group.
// The piece configuration is split into piece groups. Each group contains a listing
// of unique legal placements of the pieces from the group. These placements (and their indices)
// are used to form full board indices.
struct Piece_Group
{
	// Maximum number of pieces in a group is actually 5 (PPPPP or KAABB), 
	// but we use 7 so that things are better aligned in memory.
	static constexpr size_t MAX_PIECE_GROUP_SIZE = 7;

#if defined USE_LARGE_INDEX

	enum Placement_Index : uint32_t {
		ZERO_INDEX = 0, MAX_INDEX = 0xffffffffu, INDEX_MASK = 0xffffffffu
	};

#else

	// Represents an index into the list of unique legal placements of the pieces in this group.
	// 16-bit indices are enough for all important tablebases.
	enum Placement_Index : uint16_t {
		ZERO_INDEX = 0, MAX_INDEX = 0xffffu, INDEX_MASK = 0xffffu
	};

#endif

	// Holds two placement indices. One that corresponds to the placement directly,
	// and one that corresponds to the mirrored placement of the pieces.
	// Here, mirroring means mirroring each file (in other words, left-right mirroring).
	struct Full_Placement_Index
	{
		static_assert(sizeof(Placement_Index) == 2 || sizeof(Placement_Index) == 4);

		// We pack the two indices into a single value. This was the initial implementation
		// and it may have positive effects on register allocation. It remains to be seen
		// if a simplification could be made.
		using Packed_Type = std::conditional_t<sizeof(Placement_Index) == 2, uint32_t, uint64_t>;
		static constexpr Packed_Type HALF_MASK = static_cast<Packed_Type>(Placement_Index::INDEX_MASK);
		static constexpr Packed_Type HALF_SHIFT = 8 * sizeof(Placement_Index);

		Full_Placement_Index() :
			m_packed(0)
		{
		}

		Full_Placement_Index(Placement_Index normal) :
			m_packed(normal)
		{
		}

		Full_Placement_Index(Placement_Index normal, Placement_Index mirrored) :
			m_packed(normal | (static_cast<Packed_Type>(mirrored) << HALF_SHIFT))
		{
		}

		void set_base(Placement_Index ix)
		{
			m_packed = (m_packed & ~HALF_MASK) | ix;
		}

		void set_mirr(Placement_Index ix)
		{
			m_packed = (m_packed & HALF_MASK) | (static_cast<Packed_Type>(ix) << HALF_SHIFT);
		}

		NODISCARD Placement_Index base() const
		{
			return static_cast<Placement_Index>(m_packed & HALF_MASK);
		}

		NODISCARD Placement_Index mirr() const
		{
			return static_cast<Placement_Index>(m_packed >> HALF_SHIFT);
		}

		NODISCARD bool is_mirrored_same() const
		{
			return base() == mirr();
		}

	private:
		Packed_Type m_packed;
	};

	// Holds a list of squares. It is later used to represent 
	// a possible unique legal placement of pieces in the group.
	struct alignas(8) Placement
	{
		using iterator = Square*;

		// No need to initialize the squares, they won't be accessed.
		Placement() :
			m_size(0)
		{
		}

		INLINE void clear()
		{
			m_size = 0;
		}

		void mirror_files()
		{
			for (size_t i = 0; i < m_size; ++i)
				m_squares[i] = sq_file_mirror(m_squares[i]);
		}

		NODISCARD INLINE Placement with_mirrored_files() const
		{
			Placement dst = *this;
			dst.mirror_files();
			return dst;
		}

		INLINE void mirror_ranks()
		{
			for (size_t i = 0; i < m_size; ++i)
				m_squares[i] = sq_rank_mirror(m_squares[i]);
		}

		NODISCARD INLINE Placement with_mirrored_ranks() const
		{
			Placement dst = *this;
			dst.mirror_ranks();
			return dst;
		}

		NODISCARD bool are_all_squares_unique() const
		{
			bool arr[SQUARE_NB] = { 0 };
			for (size_t i = 0; i < m_size; ++i)
			{
				const Square sq = m_squares[i];
				if (arr[sq])
					return false;
				arr[sq] = true;
			}
			return true;
		}

		// Returns a copy of this placement with the square `from` replaced with the square `to`.
		NODISCARD INLINE Placement with_moved_square(Square from, Square to) const
		{
			// This is a commonly used function and we do gain slightly from a branchless implementation.

			static_assert(sizeof(Placement) - offsetof(Placement, m_squares) >= 8);

			Placement list;
			const __m128i from_x16 = _mm_set1_epi8(from);
			const __m128i to_x16 = _mm_set1_epi8(to);
			// Can't load full 16 bytes, so we load one half and rely on the compiler optimizing it.
			const __m128i squares = _mm_set_epi64x(0, *reinterpret_cast<const int64_t*>(m_squares));
			// We blend the broadcasted `to` squares in wherever the square was equal to `from`.
			// This can in some cases also alter the size, since we load full 8 bytes into the registers.
			// For this reason we assign size last.
			const __m128i replaced = _mm_blendv_epi8(squares, to_x16, _mm_cmpeq_epi8(squares, from_x16));
			*reinterpret_cast<int64_t*>(list.m_squares) = _mm_cvtsi128_si64(replaced);
			list.m_size = m_size;
			return list;
		}

		// Returns a copy of this placement with the given square removed.
		// All other squares are shifted towards the beginning by 1 place.
		NODISCARD INLINE Placement with_removed_square(Square to_remove) const
		{
			Placement list;
			for (size_t i = 0; i < m_size; ++i)
				if (m_squares[i] != to_remove)
					list.add(m_squares[i]);
			return list;
		}

		// Appends a square to the end of the list.
		INLINE void add(Square s)
		{
			ASSERT(m_size < MAX_PIECE_GROUP_SIZE);
			m_squares[m_size++] = s;
		}

		NODISCARD INLINE Square& operator[](size_t index)
		{
			ASSERT(index < m_size);
			return m_squares[index];
		}

		NODISCARD INLINE Square operator[](size_t index) const
		{
			ASSERT(index < m_size);
			return m_squares[index];
		}

		NODISCARD INLINE size_t size() const
		{
			return m_size;
		}

		NODISCARD INLINE const Square* begin() const
		{
			return m_squares;
		}

		NODISCARD INLINE Square* begin()
		{
			return m_squares;
		}

		NODISCARD INLINE const Square* cbegin() const
		{
			return m_squares;
		}

		NODISCARD INLINE const Square* end() const
		{
			return m_squares + m_size;
		}

		NODISCARD INLINE Square* end()
		{
			return m_squares + m_size;
		}

		NODISCARD INLINE const Square* cend() const
		{
			return m_squares + m_size;
		}

	private:
		Square m_squares[MAX_PIECE_GROUP_SIZE];
		int8_t m_size;
	};
	static_assert(sizeof(Placement) == 8);

	// Constructs a piece group from a list of pieces.
	// The list of pieces is NOT validated. It must contain pieces of only one class.
	Piece_Group(const std::vector<Piece>& pcs);

	// Returns the compound index (base and mirrored) corresponding to a given placement.
	NODISCARD INLINE Full_Placement_Index compound_index(const Placement& sq_list) const
	{
		const size_t index = non_unique_placement_index(sq_list);
		ASSERT(index < m_unique_placement_indices.size());
		return m_unique_placement_indices[index];
	}

	// Returns the placements (squares) with the given placement index.
	NODISCARD INLINE const Placement& squares(Placement_Index pos) const
	{
		ASSERT(pos < m_placements.size());
		return m_placements[pos];
	}

	// Returns the number of pieces in this group.
	NODISCARD INLINE size_t size() const
	{
		return m_num_pieces;
	}

	// Returns the ith piece in this group.
	NODISCARD INLINE Piece piece(size_t i) const
	{
		ASSERT(i < m_num_pieces);
		return m_pieces[i];
	}

	// Returns the ratio of the number of unique placements adjusted for 
	// the the symmetry under left-right mirror, 
	// and the total number of unique placements.
	// It is always >=0.5
	NODISCARD INLINE double compress_ratio() const
	{
		return static_cast<double>(m_compress_size) / m_table_size;
	}

	// Returns the number of unique placements adjusted for the symmetry
	// under left-right mirror.
	NODISCARD INLINE size_t compress_size() const
	{
		return m_compress_size;
	}

	// Returns the total number of unique placements.
	NODISCARD INLINE size_t table_size() const
	{
		return m_table_size;
	}

	// Adds a link to the same piece group but for the opposite color.
	// Should only be used during initialization phase.
	INLINE void link_to_opp_piece_group(const Piece_Group& other)
	{
		m_opp_piece_group = &other;
	}

	// Returns the placement index of the left-right mirrored position
	// corresponding to the given placement index.
	NODISCARD INLINE Placement_Index mirr_index(Placement_Index idx) const
	{
		return m_mirr_placement_index[idx];
	}

	NODISCARD INLINE const Piece_Group* opp_piece_group() const
	{
		ASSUME(m_opp_piece_group);
		return m_opp_piece_group;
	}

	// Returns the compound placement index (base and mirrored) after a given quiet move.
	NODISCARD INLINE Full_Placement_Index compound_index_after_quiet_move(const Placement_Index& current_idx, Move move) const
	{
		const Piece_Group::Placement& list = squares(current_idx);
		const size_t idx_before = non_unique_index(current_idx);
		const size_t idx_after = idx_before + non_unique_index_diff_on_move(list, move);
		return m_unique_placement_indices[idx_after];
	}

private:
	size_t m_num_pieces;
	Piece m_pieces[MAX_PIECE_GROUP_SIZE];

	// The total number of unique placements in this group.
	size_t m_table_size;

	// The number of unique placements in this group adjusted 
	// for the symmetry under left-right mirror.
	// (in other words, excluding already present by symmetry)
	size_t m_compress_size;

	// The unique placements in this group.
	// The placements are sorted by whether they are primary or not.
	// Secondary placements are the ones that already have had
	// their mirrored placement added during initialization.
	// All secondary placements have indices >=m_compress_size.
	// All primary placements have indices <m_compress_size.
	std::vector<Placement> m_placements;

	// A lookup table that converts the placement index to the placement
	// index of a mirrored placement.
	std::vector<Placement_Index> m_mirr_placement_index;

	// Index weights of each piece in this group.
	size_t m_weights[MAX_PIECE_GROUP_SIZE];

	// A lookup table that converts raw placement indices
	// (as computed by non_unique_placement_index)
	// to unique placement indices.
	std::vector<Full_Placement_Index> m_unique_placement_indices;

	// A lookup table that converts a unique placement index
	// to *some* non-unique (raw) placement index
	// (as computed by non_unique_placement_index)
	std::vector<uint32_t> m_unique_to_non_unique;

	// A lookup table that gives the non-unique (raw) placement index
	// difference after a quiet move.
	// m_diff_on_move[i][from][to] gives a difference for when a piece
	// at index `i` is moved from the square `from` to the square `to`.
	// Sizes of dimensions by square index are expanded to a power of 2 to speed up lookup.
	int32_t m_diff_on_move[MAX_PIECE_GROUP_SIZE][ceil_to_power_of_2(static_cast<size_t>(SQUARE_NB))][ceil_to_power_of_2(static_cast<size_t>(SQUARE_NB))];

	// The piece group for the same group but with pieces of the opposite color.
	const Piece_Group* m_opp_piece_group;

	// Computes a raw index of the given placement. All distinct placements have a different index, 
	// regardless of whether they are distinguishible on the board, or even legal.
	NODISCARD INLINE size_t non_unique_placement_index(const Placement& list) const
	{
		ASSUME(list.size() == m_num_pieces);
		size_t index = 0;
		for (size_t i = 0; i < list.size(); ++i)
		{
			ASSUME(m_weights[i]);
			index += m_weights[i] * static_cast<size_t>(possible_sq_index(m_pieces[i], list[i]));
		}
		return index;
	}

	// Converts a unique placement index into a non-unique (raw) one.
	// Since the raw -> unique mapping is not a bijection this returns only
	// one such corresponding index out of many possible.
	NODISCARD INLINE size_t non_unique_index(Placement_Index pos) const
	{
		ASSERT(pos < m_placements.size());
		return m_unique_to_non_unique[pos];
	}

	// Returns a difference on the raw (non-unique) placement index made by
	// a given quiet move.
	NODISCARD INLINE int32_t non_unique_index_diff_on_move(const Placement& list, Move move) const
	{
		// We can just go through all of the squares, even unpopulated one, because
		// the contract is that we will find something.
		static_assert(MAX_PIECE_GROUP_SIZE == 7);
		const Square from = move.from();
		if (list[0] == from) return m_diff_on_move[0][from][move.to()];
		if (list[1] == from) return m_diff_on_move[1][from][move.to()];
		if (list[2] == from) return m_diff_on_move[2][from][move.to()];
		if (list[3] == from) return m_diff_on_move[3][from][move.to()];
		if (list[4] == from) return m_diff_on_move[4][from][move.to()];
		if (list[5] == from) return m_diff_on_move[5][from][move.to()];
		if (list[6] == from) return m_diff_on_move[6][from][move.to()];

		ASSUME(false);
		return 0;
	}
};

ENUM_ENABLE_OPERATOR_INC(Piece_Group::Placement_Index);

// Returns a Piece_Group object corresponding to the given piece class 
// and piece counts within that class (key), for the given color.
// The DEFENDERS class has an implied single king, and the key is (num_advisors, num_bishops).
// For other classes the key is the number of the pieces in the class (since they are mono-piece classes).
// Piece_Group objects are lazily initialized and cached globally.
// This function is not thread-safe.
template <Piece_Type_Class PIECE_CLASS>
NODISCARD const Piece_Group* piece_group(
	std::conditional_t<PIECE_CLASS == DEFENDERS, 
		const std::pair<size_t, size_t>&, 
		size_t
	> key, 
	Color color
)
{
	static std::mutex s_mutex;

	static std::tuple<
		std::map<std::pair<size_t, size_t>, std::unique_ptr<Piece_Group>>,
		std::map<size_t, std::unique_ptr<Piece_Group>>,
		std::map<size_t, std::unique_ptr<Piece_Group>>,
		std::map<size_t, std::unique_ptr<Piece_Group>>,
		std::map<size_t, std::unique_ptr<Piece_Group>>
	> s_groups[COLOR_NB];

	std::unique_lock lock(s_mutex);

	auto& groups = std::get<PIECE_CLASS>(s_groups[color]);
	auto iter = groups.find(key);
	if (iter == groups.end())
	{
		std::vector<Piece> pieces;
		if constexpr (PIECE_CLASS == DEFENDERS)
		{
			pieces.emplace_back(piece_make(color, KING));
			for (size_t i = 0; i < key.first; ++i)
				pieces.emplace_back(piece_make(color, ADVISOR));
			for (size_t i = 0; i < key.second; ++i)
				pieces.emplace_back(piece_make(color, BISHOP));
		}
		else if constexpr (PIECE_CLASS == ROOKS)
		{
			for (size_t i = 0; i < key; ++i)
				pieces.emplace_back(piece_make(color, ROOK));
		}
		else if constexpr (PIECE_CLASS == KNIGHTS)
		{
			for (size_t i = 0; i < key; ++i)
				pieces.emplace_back(piece_make(color, KNIGHT));
		}
		else if constexpr (PIECE_CLASS == CANNONS)
		{
			for (size_t i = 0; i < key; ++i)
				pieces.emplace_back(piece_make(color, CANNON));
		}
		else if constexpr (PIECE_CLASS == PAWNS)
		{
			for (size_t i = 0; i < key; ++i)
				pieces.emplace_back(piece_make(color, PAWN));
		}
		else
			ASSUME(false);

		if (pieces.empty())
			return nullptr;

		auto [it, inserted] = groups.try_emplace(key, std::make_unique<Piece_Group>(pieces));
		if (inserted)
		{
			for (auto& p : pieces)
				p = piece_opp_color(p);
			auto& opp_groups = std::get<PIECE_CLASS>(s_groups[color_opp(color)]);
			auto [it_opp, inserted_opp] = opp_groups.try_emplace(key, std::make_unique<Piece_Group>(pieces));
			ASSERT(inserted_opp);
			it_opp->second->link_to_opp_piece_group(*(it->second));
			it->second->link_to_opp_piece_group(*(it_opp->second));
		}

		return &*(it->second);
	}

	return &*(iter->second);
}

// Fills the references to Piece_Group objects for each class
// based on the number of pieces of each type.
INLINE void fill_set_ids_from_piece_counts(Out_Param<const Piece_Group* [PIECE_CLASS_NB]> p_class, const std::array<size_t, PIECE_NB>& p_count)
{
	for (const Color color : { WHITE, BLACK })
	{
		p_class[make_piece_class(color, DEFENDERS)] = piece_group<DEFENDERS>(std::make_pair(p_count[piece_make(color, ADVISOR)], p_count[piece_make(color, BISHOP)]), color);
		p_class[make_piece_class(color, ROOKS)] = piece_group<ROOKS>(p_count[piece_make(color, ROOK)], color);
		p_class[make_piece_class(color, KNIGHTS)] = piece_group<KNIGHTS>(p_count[piece_make(color, KNIGHT)], color);
		p_class[make_piece_class(color, CANNONS)] = piece_group<CANNONS>(p_count[piece_make(color, CANNON)], color);
		p_class[make_piece_class(color, PAWNS)] = piece_group<PAWNS>(p_count[piece_make(color, PAWN)], color);
	}
}

// Represents a board index. It is used to index positions within an EGTB.
// It is formed by combining Piece_Group-local indices for all relevant groups.
enum Board_Index : size_t
{
	BOARD_INDEX_ZERO = 0,
	BOARD_INDEX_NONE = std::numeric_limits<size_t>::max()
};

ENUM_ENABLE_OPERATOR_INC(Board_Index);
ENUM_ENABLE_OPERATOR_DEC(Board_Index);
ENUM_ENABLE_OPERATOR_ADD(Board_Index);
ENUM_ENABLE_OPERATOR_SUB(Board_Index);
ENUM_ENABLE_OPERATOR_DIFF(Board_Index);
ENUM_ENABLE_OPERATOR_ADD_EQ(Board_Index);
ENUM_ENABLE_OPERATOR_SUB_EQ(Board_Index);

// Piece_Group-local index for each piece class.
using Decomposed_Board_Index = std::array<Piece_Group::Placement_Index, PIECE_CLASS_NB>;

// A helper that provides paths to existing and new EGTB artifacts.
struct EGTB_Paths
{
	static inline const std::string WDL_EXT = ".lzw";
	static inline const std::string WDL_GEN_EXT = ".lzw.gen";
	static inline const std::string WDL_TMP_EXT[COLOR_NB] = { ".w.evtb", ".b.evtb" };
	static inline const std::string DTM_TMP_EXT[COLOR_NB] = { ".w.egtb", ".b.egtb" };
	static inline const std::string DTC_EXT = ".lzdtc";
	static inline const std::string DTM_EXT = ".lzdtm";
	static inline const std::string INFO_EXT = ".info";

	EGTB_Paths()
	{
	}

	void add_dtm_path(std::filesystem::path s)
	{
		m_dtm_paths.emplace_back(std::move(s));
	}

	void add_dtc_path(std::filesystem::path s)
	{
		m_dtc_paths.emplace_back(std::move(s));
	}

	void add_wdl_path(std::filesystem::path s)
	{
		m_wdl_paths.emplace_back(std::move(s));
	}

	void set_tmp_path(std::filesystem::path s)
	{
		m_tmp_path = std::move(s);
	}

	void init_directories() const
	{
		std::filesystem::create_directories(m_tmp_path);
		for (const auto& p : m_wdl_paths)
			std::filesystem::create_directories(p);
		for (const auto& p : m_dtc_paths)
			std::filesystem::create_directories(p);
		for (const auto& p : m_dtm_paths)
			std::filesystem::create_directories(p);
	}

	NODISCARD bool find_wdl_file(const Piece_Config& ps, std::filesystem::path* tb = nullptr, bool gen = false) const
	{
		const std::string& ext = gen ? WDL_GEN_EXT : WDL_EXT;
		return find_tb_file(ps, ext, m_wdl_paths, tb);
	}

	NODISCARD bool find_dtm_file(const Piece_Config& ps, std::filesystem::path* tb = nullptr) const
	{
		return find_tb_file(ps, DTM_EXT, m_dtm_paths, tb);
	}

	NODISCARD bool find_dtc_file(const Piece_Config& ps, std::filesystem::path* tb = nullptr) const
	{
		return find_tb_file(ps, DTC_EXT, m_dtc_paths, tb);
	}

	NODISCARD std::filesystem::path dtm_tmp_path(const Piece_Config& ps, Color c) const
	{
		return path_join(m_tmp_path, ps.name() + DTM_TMP_EXT[c]);
	}

	NODISCARD std::filesystem::path wdl_tmp_path(const Piece_Config& ps, Color c) const
	{
		return path_join(m_tmp_path, ps.name() + WDL_TMP_EXT[c]);
	}

	NODISCARD std::filesystem::path wdl_save_path(const Piece_Config& ps) const
	{
		return path_join(m_wdl_paths[0], ps.name() + WDL_EXT);
	}

	NODISCARD std::filesystem::path wdl_gen_save_path(const Piece_Config& ps) const
	{
		return path_join(m_wdl_paths[0], ps.name() + WDL_GEN_EXT);
	}

	NODISCARD std::filesystem::path dtc_save_path(const Piece_Config& ps) const
	{
		return path_join(m_dtc_paths[0], ps.name() + DTC_EXT);
	}

	NODISCARD std::filesystem::path dtc_info_save_path(const Piece_Config& ps) const
	{
		return path_join(m_dtc_paths[0], ps.name() + INFO_EXT);
	}

	NODISCARD std::filesystem::path dtm_save_path(const Piece_Config& ps) const
	{
		return path_join(m_dtm_paths[0], ps.name() + DTM_EXT);
	}

	NODISCARD std::filesystem::path dtm_info_save_path(const Piece_Config& ps) const
	{
		return path_join(m_dtm_paths[0], ps.name() + INFO_EXT);
	}

private:
	std::filesystem::path m_tmp_path = "./tmp/";
	std::vector<std::filesystem::path> m_dtc_paths = { "./dtc/" };
	std::vector<std::filesystem::path> m_dtm_paths = { "./dtm/" };
	std::vector<std::filesystem::path> m_wdl_paths = { "./wdl/" };

	NODISCARD bool find_tb_file(
		const Piece_Config& ps,
		const std::string& ext,
		const std::vector<std::filesystem::path>& paths,
		std::filesystem::path* tb = nullptr
	) const
	{
		const std::string name = ps.name() + ext;

		for (const auto& dir : paths)
		{
			const auto path = path_join(dir, name);
			if (std::filesystem::exists(path))
			{
				if (tb != nullptr)
					*tb = path;
				return true;
			}
		}
		return false;
	}
};

// Represents a single entry in a WDL EGTB.
// NOTE: After serialization ILLEGAL entries may not be marked ILLEGAL.
//       This is done to improve compression.
enum struct WDL_Entry : uint8_t {
	ILLEGAL = 3, WIN = 2, LOSE = 1, DRAW = 0
};

// Represents 4 single WDL entries packed into a single byte.
enum Packed_WDL_Entries : uint8_t {};

// Number of single WDL entries in a single packed WDL entry.
static constexpr size_t WDL_ENTRY_PACK_RATIO = 4;

// Packs 4 WDL entries, passed individually, into a single packed WDL entry.
NODISCARD constexpr Packed_WDL_Entries pack_wdl_entries(WDL_Entry v0, WDL_Entry v1, WDL_Entry v2, WDL_Entry v3)
{
	return static_cast<Packed_WDL_Entries>(
		  (static_cast<uint8_t>(v0) << 0)
		| (static_cast<uint8_t>(v1) << 2)
		| (static_cast<uint8_t>(v2) << 4)
		| (static_cast<uint8_t>(v3) << 6)
		);
}

// Packs 4 WDL entries, passed as an array, into a single packed WDL entry.
NODISCARD constexpr Packed_WDL_Entries pack_wdl_entries(const WDL_Entry v[4])
{
	return pack_wdl_entries(v[0], v[1], v[2], v[3]);
}

// Packs an array of WDL entries into an array of packed WDL entries.
// The sizes of both input and output spans must be exact.
// The number of elements in the input must be divisible by WDL_ENTRY_PACK_RATIO.
inline constexpr void pack_wdl_entries(Const_Span<WDL_Entry> in, Span<Packed_WDL_Entries> out)
{
	ASSERT(in.size() == out.size() * WDL_ENTRY_PACK_RATIO);
	ASSERT(in.size() % WDL_ENTRY_PACK_RATIO == 0);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = pack_wdl_entries(in.data() + i * WDL_ENTRY_PACK_RATIO);
}

// Unpacks a packed WDL entry into an array of WDL entries.
constexpr void unpack_wdl_entries(Packed_WDL_Entries packed, WDL_Entry out[4])
{
	out[0] = static_cast<WDL_Entry>((packed >> 0) & 3);
	out[1] = static_cast<WDL_Entry>((packed >> 2) & 3);
	out[2] = static_cast<WDL_Entry>((packed >> 4) & 3);
	out[3] = static_cast<WDL_Entry>((packed >> 6) & 3);
}

// Unpacks an array of packed WDL entries into an array of WDL entries.
// The sizes of both input and output spans must be exact.
// All packed WDL entries in the input are assumed to contain WDL_ENTRY_PACK_RATIO WDL entries.
inline constexpr void unpack_wdl_entries(Const_Span<Packed_WDL_Entries> in, Span<WDL_Entry> out)
{
	ASSERT(in.size() * WDL_ENTRY_PACK_RATIO == out.size());
	for (size_t i = 0; i < in.size(); ++i)
		unpack_wdl_entries(in[i], out.data() + i * WDL_ENTRY_PACK_RATIO);
}

// Inverted mask of a WDL entry in a packed WDL entry, by its index.
// Used for faster replacement of WDL entries in a packed WDL entry.
static constexpr uint8_t PACKED_WDL_ENTRY_INV_MASK[4] = {
	0b11111100,
	0b11110011,
	0b11001111,
	0b00111111
};

// Sets a single WDL entry at the given position in a packed WDL entry.
// `pos` must be less than WDL_ENTRY_PACK_RATIO.
constexpr void set_wdl_entry(Packed_WDL_Entries& packed, size_t pos, WDL_Entry new_value)
{
	ASSERT(pos < WDL_ENTRY_PACK_RATIO);
	packed = static_cast<Packed_WDL_Entries>((packed & PACKED_WDL_ENTRY_INV_MASK[pos]) | (static_cast<uint8_t>(new_value) << (pos * 2)));
}

// Retrieves a single WDL entry at the given position from a packed WDL entry.
// `pos` must be less than WDL_ENTRY_PACK_RATIO.
NODISCARD constexpr WDL_Entry get_wdl_value(Packed_WDL_Entries packed, size_t pos)
{
	ASSERT(pos < WDL_ENTRY_PACK_RATIO);
	return static_cast<WDL_Entry>((packed >> (pos * 2)) & 3);
}

enum DTC_Intermediate_Entry_Flag : uint16_t {
	DTC_FLAG_CAP_DRAW = 1 << 14,
	DTC_FLAG_CHECK = 1 << 13,
	DTC_FLAG_CHASE = 1 << 12,
	DTC_FLAG_IN_CHECK = 1 << 11,
	DTC_FLAG_IN_CHASE = 1 << 10,
	DTC_FLAG_CHASE_WIN = 1 << 9,
	DTC_FLAG_CHASE_LOSE = 1 << 8,
	DTC_FLAG_CHECK_WIN = 1 << 7,
	DTC_FLAG_CHECK_LOSE = 1 << 6,
};

ENUM_ENABLE_OPERATOR_OR(DTC_Intermediate_Entry_Flag);
ENUM_ENABLE_OPERATOR_OR_EQ(DTC_Intermediate_Entry_Flag);

enum DTC_Final_Entry_Flag : uint16_t {
	DTC_FLAG_ORDER_128 = 1 << 9,
	DTC_VALUE_MASK_64 = 0x3ff, DTC_VALUE_MASK_128 = 0x1ff,
};

ENUM_ENABLE_OPERATOR_OR(DTC_Final_Entry_Flag);
ENUM_ENABLE_OPERATOR_OR_EQ(DTC_Final_Entry_Flag);

enum DTC_Score : uint16_t {
	DTC_SCORE_ZERO = 0,
	DTC_SCORE_TERMINAL_LOSS = 1,
	DTC_SCORE_TERMINAL_WIN = 2,
	DTC_SCORE_MAX_ORDER_128 = 510,
	DTC_SCORE_MAX_ORDER_64 = 1020,
};

ENUM_ENABLE_OPERATOR_ADD(DTC_Score);
ENUM_ENABLE_OPERATOR_INC(DTC_Score);

enum DTC_Order : uint16_t {
	DTC_ORDER_ZERO = 0,
	DTC_ORDER_MAX_ORDER_64 = 63,
	DTC_ORDER_MAX_ORDER_128 = 127,
};

ENUM_ENABLE_OPERATOR_INC(DTC_Order);

enum struct DTC_Entry_Order
{
	ORDER_64,
	ORDER_128
};

struct DTC_Intermediate_Entry
{
	template <typename FlagT>
	static constexpr bool is_allowed_flag_type = std::is_same_v<FlagT, DTC_Intermediate_Entry_Flag>;

	constexpr DTC_Intermediate_Entry() :
		m_data(0)
	{
	}

	NODISCARD static constexpr DTC_Intermediate_Entry make_cap_draw()
	{
		DTC_Intermediate_Entry entry;
		entry.set_flag(DTC_FLAG_CAP_DRAW);
		return entry;
	}

	NODISCARD constexpr bool operator==(const DTC_Intermediate_Entry& other) const
	{
		return m_data == other.m_data;
	}

	NODISCARD constexpr bool operator!=(const DTC_Intermediate_Entry& other) const
	{
		return m_data != other.m_data;
	}

	constexpr void set_flag(DTC_Intermediate_Entry_Flag flag)
	{
		ASSERT(is_modifiable_flag(flag));
		m_data |= flag;
	}

	NODISCARD constexpr bool has_flag(DTC_Intermediate_Entry_Flag flag) const
	{
		ASSERT(is_modifiable_flag(flag));
		return m_data & flag;
	}

	constexpr void clear_flag(DTC_Intermediate_Entry_Flag flag)
	{
		ASSERT(is_modifiable_flag(flag));
		m_data &= ~flag;
	}

private:
	uint16_t m_data;

	NODISCARD static constexpr bool is_modifiable_flag(DTC_Intermediate_Entry_Flag flag)
	{
		return
			(flag &
				(  DTC_FLAG_CAP_DRAW
				 | DTC_FLAG_CHECK
				 | DTC_FLAG_CHASE
				 | DTC_FLAG_IN_CHECK
				 | DTC_FLAG_IN_CHASE
				 | DTC_FLAG_CHASE_WIN
				 | DTC_FLAG_CHASE_LOSE
				 | DTC_FLAG_CHECK_WIN
				 | DTC_FLAG_CHECK_LOSE)
				) == flag;
	}
};

struct DTC_Final_Entry
{
	template <typename FlagT>
	static constexpr bool is_allowed_flag_type = std::is_same_v<FlagT, DTC_Final_Entry_Flag>;

	template <DTC_Entry_Order ORDER>
	static constexpr DTC_Score MAX_STEP = ORDER == DTC_Entry_Order::ORDER_64 ? DTC_SCORE_MAX_ORDER_64 : DTC_SCORE_MAX_ORDER_128;

	NODISCARD static constexpr bool is_value_ambiguous_with_order_128(DTC_Score value)
	{
		return value >= MAX_STEP<DTC_Entry_Order::ORDER_128>;
	}

	constexpr DTC_Final_Entry() :
		m_data(0)
	{
	}

	NODISCARD static constexpr DTC_Final_Entry make_illegal()
	{
		DTC_Final_Entry e;
		e.m_data = 0xffff;
		return e;
	}

	NODISCARD static constexpr DTC_Final_Entry make_draw()
	{
		return {};
	}

	NODISCARD static constexpr DTC_Final_Entry make_win()
	{
		DTC_Final_Entry e;
		e.set_value<DTC_Entry_Order::ORDER_64>(DTC_SCORE_TERMINAL_WIN, DTC_ORDER_ZERO);
		return e;
	}

	NODISCARD static constexpr DTC_Final_Entry make_lose()
	{
		DTC_Final_Entry e;
		e.set_value<DTC_Entry_Order::ORDER_64>(DTC_SCORE_TERMINAL_LOSS, DTC_ORDER_ZERO);
		return e;
	}

	template <DTC_Entry_Order ORDER>
	NODISCARD static constexpr DTC_Final_Entry make_score(DTC_Score value, DTC_Order order)
	{
		DTC_Final_Entry e;
		e.set_value<ORDER>(value, order);
		return e;
	}

	NODISCARD constexpr bool operator==(const DTC_Final_Entry& other) const
	{
		return m_data == other.m_data;
	}

	NODISCARD constexpr bool operator!=(const DTC_Final_Entry& other) const
	{
		return m_data != other.m_data;
	}

	template <DTC_Entry_Order ORDER>
	NODISCARD constexpr DTC_Score value() const
	{
		return static_cast<DTC_Score>(m_data & (ORDER == DTC_Entry_Order::ORDER_64 ? DTC_VALUE_MASK_64 : DTC_VALUE_MASK_128));
	}

	NODISCARD constexpr bool is_legal() const
	{
		return m_data != 0xffff;
	}

	template <DTC_Entry_Order ORDER>
	NODISCARD constexpr bool is_win() const
	{
		return is_legal() && value<ORDER>() && (value<ORDER>() & 1) == 0;
	}

	template <DTC_Entry_Order ORDER>
	NODISCARD constexpr bool is_loss_or_draw() const
	{
		return is_legal() && (value<ORDER>() == 0 || (value<ORDER>() & 1) == 1);
	}

private:
	uint16_t m_data;

	template <DTC_Entry_Order ORDER>
	constexpr void set_value(DTC_Score value, DTC_Order order)
	{
		value = std::min(value, MAX_STEP<ORDER>);

		if constexpr (ORDER == DTC_Entry_Order::ORDER_64)
			m_data = static_cast<uint16_t>((std::min(order, DTC_ORDER_MAX_ORDER_64) << 10) | value);
		else if (order > DTC_ORDER_MAX_ORDER_64)
			m_data = static_cast<uint16_t>(((std::min(order, DTC_ORDER_MAX_ORDER_128) - 64) << 10) | DTC_FLAG_ORDER_128 | value);
		else
			m_data = static_cast<uint16_t>((order << 10) | value);
	}
};

using DTC_Any_Entry = std::variant<DTC_Intermediate_Entry, DTC_Final_Entry>;

enum DTM_Rule_Flag : uint16_t {
	DTM_FLAG_CHECK_WIN = 1 << 15,
	DTM_FLAG_CHECK_LOSE = 1 << 14,
	DTM_FLAG_CHASE_WIN = 1 << 13,
	DTM_FLAG_CHASE_LOSE = 1 << 12,

	DTM_FLAG_LOSE_BAN = DTM_FLAG_CHECK_LOSE | DTM_FLAG_CHASE_LOSE,
	DTM_FLAG_WIN_BAN = DTM_FLAG_CHECK_WIN | DTM_FLAG_CHASE_WIN,
};

ENUM_ENABLE_OPERATOR_OR(DTM_Rule_Flag);
ENUM_ENABLE_OPERATOR_OR_EQ(DTM_Rule_Flag);

enum DTM_Intermediate_Entry_Flag : uint16_t {
	DTM_FLAG_CAP_CONVERT = 1 << 11,
};

ENUM_ENABLE_OPERATOR_OR(DTM_Intermediate_Entry_Flag);
ENUM_ENABLE_OPERATOR_OR_EQ(DTM_Intermediate_Entry_Flag);

enum DTM_Final_Entry_Flag : uint16_t {
	DTM_FLAG_WIN = 1 << 11,
};

ENUM_ENABLE_OPERATOR_OR(DTM_Final_Entry_Flag);
ENUM_ENABLE_OPERATOR_OR_EQ(DTM_Final_Entry_Flag);

enum DTM_Entry_Masks : uint16_t {
	DTM_ILLEGAL = 0xffff,
	DTM_RULE_MASK = 0xf000,
	DTM_SCORE_MASK = 0x7ff,
};

enum DTM_Score : uint16_t {
	DTM_SCORE_ZERO = 0,
	DTM_SCORE_TERMINAL_LOSS = 1,
	DTM_SCORE_TERMINAL_WIN = 2,
	DTM_SCORE_MAX = 2040, 
};

ENUM_ENABLE_OPERATOR_INC(DTM_Score);
ENUM_ENABLE_OPERATOR_ADD(DTM_Score);

struct DTM_Final_Entry;

struct DTM_Entry_Base
{
	template <typename FlagT>
	static constexpr bool is_allowed_flag_type = std::is_same_v<FlagT, DTM_Rule_Flag>;

	constexpr DTM_Entry_Base() :
		m_data(0)
	{
	}

	constexpr void set_flag(DTM_Rule_Flag flag)
	{
		ASSERT(is_modifiable_flag(flag));
		m_data |= flag;
	}

	NODISCARD constexpr bool has_flag(DTM_Rule_Flag flag) const
	{
		ASSERT(is_modifiable_flag(flag));
		return m_data & flag;
	}

	constexpr void clear_flag(DTM_Rule_Flag flag)
	{
		ASSERT(is_modifiable_flag(flag));
		m_data &= ~flag;
	}

	NODISCARD constexpr bool is_ban_lose() const
	{
		return m_data != DTM_ILLEGAL && (m_data & DTM_FLAG_LOSE_BAN) != 0;
	}

	NODISCARD constexpr bool is_ban_win() const
	{
		return m_data != DTM_ILLEGAL && (m_data & DTM_FLAG_WIN_BAN) != 0;
	}

	NODISCARD constexpr bool is_ban(WDL_Entry type) const
	{
		ASSERT(type == WDL_Entry::WIN || type == WDL_Entry::LOSE);
		return type == WDL_Entry::WIN ? is_ban_win() : is_ban_lose();
	}

	NODISCARD constexpr bool is_legal() const
	{
		return m_data != 0xffff;
	}

	constexpr void remove_rule_bits()
	{
		m_data = m_data & ~DTM_RULE_MASK;
	}

protected:
	uint16_t m_data;

	NODISCARD static constexpr bool is_modifiable_flag(DTM_Rule_Flag flag)
	{
		return (flag & DTM_RULE_MASK) == flag;
	}
};

struct DTM_Intermediate_Entry : public DTM_Entry_Base
{
	template <typename FlagT>
	static constexpr bool is_allowed_flag_type = DTM_Entry_Base::is_allowed_flag_type<FlagT> || std::is_same_v<FlagT, DTM_Intermediate_Entry_Flag>;

	friend struct DTM_Final_Entry;

	constexpr DTM_Intermediate_Entry() :
		DTM_Entry_Base{}
	{
	}

	NODISCARD static constexpr DTM_Intermediate_Entry make_cap_score(DTM_Score score)
	{
		DTM_Intermediate_Entry e;
		e.m_data = static_cast<uint16_t>(DTM_FLAG_CAP_CONVERT | score);
		return e;
	}

	NODISCARD static constexpr DTM_Intermediate_Entry make_empty()
	{
		return {};
	}

	NODISCARD friend bool operator==(DTM_Intermediate_Entry lhs, DTM_Intermediate_Entry rhs) noexcept
	{
		return lhs.m_data == rhs.m_data;
	}

	NODISCARD friend bool operator!=(DTM_Intermediate_Entry lhs, DTM_Intermediate_Entry rhs) noexcept
	{
		return lhs.m_data != rhs.m_data;
	}

	NODISCARD constexpr bool has_cap_score() const
	{
		return m_data & DTM_FLAG_CAP_CONVERT;
	}

	NODISCARD constexpr DTM_Score cap_score() const
	{
		return static_cast<DTM_Score>(m_data & DTM_SCORE_MASK);
	}

	constexpr void clear_flags()
	{
		m_data = m_data & (DTM_RULE_MASK | DTM_SCORE_MASK);
	}

	NODISCARD constexpr bool is_cap_win() const
	{
		return (m_data & DTM_FLAG_CAP_CONVERT) && (m_data & 1) == 0;
	}

	NODISCARD constexpr bool is_cap_lose() const
	{
		return (m_data & DTM_FLAG_CAP_CONVERT) && (m_data & 1) != 0;
	}
};

struct DTM_Final_Entry : public DTM_Entry_Base
{
	template <typename FlagT>
	static constexpr bool is_allowed_flag_type = DTM_Entry_Base::is_allowed_flag_type<FlagT> || std::is_same_v<FlagT, DTM_Final_Entry_Flag>;

	constexpr DTM_Final_Entry() :
		DTM_Entry_Base{}
	{
	}

	static DTM_Final_Entry copy_rule(const DTM_Intermediate_Entry entry)
	{
		DTM_Final_Entry e;
		e.m_data = entry.m_data & DTM_RULE_MASK;
		return e;
	}

	NODISCARD static constexpr DTM_Final_Entry make_illegal()
	{
		DTM_Final_Entry e;
		e.m_data = 0xffff;
		return e;
	}

	NODISCARD static constexpr DTM_Final_Entry make_draw()
	{
		return {};
	}

	NODISCARD static constexpr DTM_Final_Entry make_loss(DTM_Score score)
	{
		DTM_Final_Entry e;
		e.m_data = static_cast<uint16_t>(score);
		return e;
	}

	constexpr void set_score_win(DTM_Score score)
	{
		ASSERT((score & DTM_SCORE_MASK) == score);
		m_data = static_cast<uint16_t>((m_data & DTM_RULE_MASK) | DTM_FLAG_WIN | score);
	}

	constexpr void set_score_lose(DTM_Score score)
	{
		ASSERT((score & DTM_SCORE_MASK) == score);
		m_data = static_cast<uint16_t>((m_data & DTM_RULE_MASK) | score);
	}

	constexpr void set_score(DTM_Score score)
	{
		ASSERT((score & DTM_SCORE_MASK) == score);
		m_data = static_cast<uint16_t>((m_data & ~DTM_SCORE_MASK) | score);
	}

	NODISCARD constexpr DTM_Score score() const
	{
		return static_cast<DTM_Score>(m_data & DTM_SCORE_MASK);
	}

	NODISCARD constexpr bool is_draw() const
	{
		return m_data == 0;
	}

	NODISCARD constexpr bool is_win() const
	{
		return m_data != DTM_ILLEGAL && (m_data & DTM_FLAG_WIN) != 0;
	}

	NODISCARD constexpr bool is_lose() const
	{
		return m_data != DTM_ILLEGAL && score() != 0 && (m_data & DTM_FLAG_WIN) == 0;
	}
};

using DTM_Any_Entry = std::variant<DTM_Intermediate_Entry, DTM_Final_Entry>;

template <typename MainEntryT, typename... OtherEntryTs>
struct EGTB_File_For_Probe
{
	static constexpr size_t NUM_ENTRY_VARIANTS = 1 + sizeof...(OtherEntryTs);
	static constexpr size_t ENTRY_SIZE = sizeof(MainEntryT);
	static_assert(((sizeof(OtherEntryTs) == ENTRY_SIZE) && ...));
	static_assert(ENTRY_SIZE == 1 || ENTRY_SIZE == 2 || ENTRY_SIZE == 4 || ENTRY_SIZE == 8);

	using Underlying_Entry_Type = Unsigned_Int_Of_Size<ENTRY_SIZE>;

	friend void load_egtb_table(
		Out_Param<EGTB_File_For_Probe<MainEntryT, OtherEntryTs...>> egtb,
		const Piece_Config& ps,
		std::filesystem::path sub_evtb,
		const std::filesystem::path tmp[COLOR_NB],
		EGTB_Magic evtb_magic
	);

	static size_t uncompressed_file_size(size_t num_entries)
	{
		return num_entries * sizeof(DTM_Final_Entry);
	}

	EGTB_File_For_Probe() :
		m_is_singular_draw{ false, false }
	{
	}

	EGTB_File_For_Probe(const EGTB_Paths& egtb_files, const Piece_Config& ps) :
		EGTB_File_For_Probe()
	{
		open(egtb_files, ps);
	}

	EGTB_File_For_Probe(const EGTB_File_For_Probe&) = delete;
	EGTB_File_For_Probe(EGTB_File_For_Probe&&) noexcept = default;

	EGTB_File_For_Probe& operator=(const EGTB_File_For_Probe&) = delete;
	EGTB_File_For_Probe& operator=(EGTB_File_For_Probe&&) noexcept = default;

	~EGTB_File_For_Probe()
	{
		close();
	}

	void open(const EGTB_Paths& egtb_files, const Piece_Config& ps)
	{
		std::filesystem::path path;
		if (!egtb_files.find_dtm_file(ps, &path))
			print_and_abort("找不到子库：%s\n", ps.name().c_str());

		const std::filesystem::path tmp[COLOR_NB] = {
			m_tmp_files.track_path(egtb_files.dtm_tmp_path(ps, WHITE)),
			m_tmp_files.track_path(egtb_files.dtm_tmp_path(ps, BLACK))
		};

		load_egtb_table(out_param(*this), ps, path, tmp, EGTB_Magic::DTM_MAGIC);
	}

	void close()
	{
		m_files[WHITE].close();
		m_files[BLACK].close();
		m_tmp_files.clear();
		m_is_singular_draw[WHITE] = m_is_singular_draw[BLACK] = false;
	}

	template <size_t N = NUM_ENTRY_VARIANTS>
	NODISCARD std::enable_if_t<N == 1, MainEntryT> read(Color color, Board_Index pos) const
	{
		if (m_is_singular_draw[color])
			return MainEntryT::make_draw();

		MainEntryT entry;
		std::memcpy(&entry, m_files[color].data() + pos * sizeof(MainEntryT), sizeof(MainEntryT));
		return entry;
	}

	template<typename T, size_t N = NUM_ENTRY_VARIANTS>
	NODISCARD std::enable_if_t<N != 1, T> read(Color color, Board_Index pos) const
	{
		if (m_is_singular_draw[color])
			return T::make_draw();

		T entry;
		std::memcpy(&entry, m_files[color].data() + pos * sizeof(T), sizeof(T));
		return entry;
	}

private:
	bool m_is_singular_draw[COLOR_NB];
	Memory_Mapped_File m_files[COLOR_NB];
	Temporary_File_Tracker m_tmp_files;
};

template <>
struct EGTB_File_For_Probe<WDL_Entry>
{
	friend void load_evtb_table(
		Out_Param<EGTB_File_For_Probe<WDL_Entry>> evtb,
		const Piece_Config& ps,
		std::filesystem::path sub_evtb,
		const std::filesystem::path tmp[COLOR_NB],
		EGTB_Magic evtb_magic
	);

	static size_t uncompressed_file_size(size_t num_entries)
	{
		return ceil_div(num_entries, WDL_ENTRY_PACK_RATIO);
	}

	EGTB_File_For_Probe() :
		m_is_singular{ false, false },
		m_single_val{ WDL_Entry::DRAW, WDL_Entry::DRAW }
	{
	}

	EGTB_File_For_Probe(const EGTB_Paths& egtb_files, const Piece_Config& ps, bool table_symmetric) :
		EGTB_File_For_Probe()
	{
		open(egtb_files, ps, table_symmetric);
	}

	EGTB_File_For_Probe(const EGTB_File_For_Probe&) = delete;
	EGTB_File_For_Probe(EGTB_File_For_Probe&&) noexcept = default;

	EGTB_File_For_Probe& operator=(const EGTB_File_For_Probe&) = delete;
	EGTB_File_For_Probe& operator=(EGTB_File_For_Probe&&) noexcept = default;

	~EGTB_File_For_Probe()
	{
		close();
	}

	void open(const EGTB_Paths& egtb_files, const Piece_Config& ps, bool table_symmetric)
	{
		std::filesystem::path path;
		if (!egtb_files.find_wdl_file(ps, &path, table_symmetric))
			throw std::runtime_error("Could not find a WDL file for " + ps.name());

		const std::filesystem::path tmp[COLOR_NB] = {
			m_tmp_files.track_path(egtb_files.wdl_tmp_path(ps, WHITE)),
			m_tmp_files.track_path(egtb_files.wdl_tmp_path(ps, BLACK))
		};

		load_evtb_table(out_param(*this), ps, path, tmp, EGTB_Magic::WDL_MAGIC);
	}

	void close()
	{
		m_files[WHITE].close();
		m_files[BLACK].close();
		m_tmp_files.clear();
		m_is_singular[WHITE] = m_is_singular[BLACK] = false;
		m_single_val[WHITE] = m_single_val[BLACK] = WDL_Entry::DRAW;
	}

	NODISCARD WDL_Entry read(Color color, Board_Index pos) const
	{
		if (m_is_singular[color])
			return m_single_val[color];

		Packed_WDL_Entries entry;
		std::memcpy(&entry, m_files[color].data() + pos / 4 * sizeof(Packed_WDL_Entries), sizeof(Packed_WDL_Entries));
		return get_wdl_value(entry, pos % 4);
	}

private:
	bool m_is_singular[COLOR_NB];
	WDL_Entry m_single_val[COLOR_NB];
	Memory_Mapped_File m_files[COLOR_NB];
	Temporary_File_Tracker m_tmp_files;
};

using WDL_File_For_Probe = EGTB_File_For_Probe<WDL_Entry>;
using DTM_File_For_Probe = EGTB_File_For_Probe<DTM_Final_Entry>;

struct EGTB_Info
{
	EGTB_Info()
	{
		clear();
	}

	void clear()
	{
		memset(this, 0, sizeof(EGTB_Info));
	}

	void maybe_update_longest_win(Color color, size_t idx, size_t value)
	{
		if (value > longest_win[color])
		{
			longest_win[color] = narrowing_static_cast<uint16_t>(value);
			longest_idx[color] = idx;
		}
	}

	void add_result(Color color, WDL_Entry value)
	{
		switch (value)
		{
		case WDL_Entry::DRAW:
			draw_cnt[color] += 1;
			break;
		case WDL_Entry::LOSE:
			lose_cnt[color] += 1;
			break;
		case WDL_Entry::WIN:
			win_cnt[color] += 1;
			break;
		case WDL_Entry::ILLEGAL:
			illegal_cnt[color] += 1;
			break;
		default:
			ASSUME(false);
		}
	}

	template <typename IterT>
	void consolidate_from(IterT begin, IterT end, Color color)
	{
		// Since the order of position subranges that threads check
		// can interleave between threads we do not have an implicit
		// tie-breaker ordering by position index, therefore the
		// longest win tie-break condition is now explicit to keep
		// the behaviour consistent.
		while (begin != end)
		{
			const auto& info = *begin;

			win_cnt[color] += info.win_cnt[color];
			draw_cnt[color] += info.draw_cnt[color];
			lose_cnt[color] += info.lose_cnt[color];
			illegal_cnt[color] += info.illegal_cnt[color];
			if (longest_win[color] < info.longest_win[color]
				|| (longest_win[color] == info.longest_win[color]
					&& longest_idx[color] > info.longest_idx[color]))
			{
				longest_win[color] = info.longest_win[color];
				longest_idx[color] = info.longest_idx[color];
			}

			++begin;
		}
	}

	uint64_t win_cnt[COLOR_NB];
	uint64_t lose_cnt[COLOR_NB];
	uint64_t draw_cnt[COLOR_NB];
	uint64_t illegal_cnt[COLOR_NB];
	uint16_t longest_win[COLOR_NB];

	static_assert(MAX_FEN_LENGTH == 120, "For compatibility. Otherwise additional checks are required.");
	char longest_fen[COLOR_NB][MAX_FEN_LENGTH];

	uint8_t loop_cnt[COLOR_NB];
	uint64_t longest_idx[COLOR_NB];
};
static_assert(sizeof(EGTB_Info) == 328);
