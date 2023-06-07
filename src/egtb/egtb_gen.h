#pragma once

#include "egtb.h"

#include "chess/piece_config.h"
#include "chess/position.h"

#include "util/algo.h"
#include "util/allocation.h"
#include "util/fixed_vector.h"
#include "util/intrin.h"
#include "util/span.h"
#include "util/param.h"
#include "util/math.h"
#include "util/utility.h"
#include "util/filesystem.h"
#include "util/thread_pool.h"
#include "util/progress_bar.h"
#include "util/division.h"
#include "util/compress.h"

#include <atomic>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <chrono>
#include <climits>
#include <utility>
#include <optional>
#include <filesystem>

struct Piece_Config_For_Gen : public Piece_Config
{
private:
	static constexpr size_t MAX_NUM_POSITIONS = 0xffffffffffffull;

	NODISCARD static bool try_init(Piece_Config_For_Gen& info)
	{
		info.m_both_sides_have_free_attackers = 
			   info.has_any_free_attackers(WHITE)
			&& info.has_any_free_attackers(BLACK);

		const auto pc = info.piece_counts();
		for (const Piece p : ALL_PIECES)
			info.m_piece_counts[p] = narrowing_static_cast<int8_t>(pc[p]);

		fill_set_ids_from_piece_counts(out_param(info.m_groups), pc);

		info.m_compress_id = compute_compress_id(info.m_groups);

		memset(info.m_weight_by_group, 0, sizeof(info.m_weight_by_group));
		info.m_num_populated_classes = 0;

		size_t w = 1;
		for (Piece_Class i = PIECE_CLASS_START; i < PIECE_CLASS_END; ++i)
		{
			if (info.m_groups[i] != nullptr)
			{
				info.m_num_positions_by_group[i] =
					i == info.m_compress_id
					? info.m_groups[i]->compress_size()
					: info.m_groups[i]->table_size();

				info.m_populated_classes[info.m_num_populated_classes++] = i;
				info.m_weight_by_group[i] = w;
				if (w != 1)
					info.m_weight_divider_by_group[i] = w;
				const size_t next_w = w * info.m_num_positions_by_group[i];
				if (next_w < w || next_w > MAX_NUM_POSITIONS)
				{
					info.m_num_positions = std::numeric_limits<size_t>::max();
					return false;
				}
				w = next_w;
			}
		}
		info.m_num_positions = w;
		return true;
	}

public:
	NODISCARD static std::optional<size_t> num_positions_safe(const Piece_Config& ps)
	{
		bool ok;
		const Piece_Config_For_Gen epsi(ps, out_param(ok));
		if (ok)
			return epsi.num_positions();
		else
			return std::nullopt;

	}

	explicit Piece_Config_For_Gen(const Piece_Config& ps) :
		Piece_Config(ps)
	{
		if (!try_init(*this))
			throw std::runtime_error("Piece set too large, would overflow size.");
	}

	Piece_Config_For_Gen(const Piece_Config& ps, Out_Param<bool> ok) :
		Piece_Config(ps)
	{
		*ok = try_init(*this);
	}

	template <bool ASSUME_LEGAL>
	bool fill_board(const Decomposed_Board_Index& index, Out_Param<Position> board) const
	{
		std::memset(board->m_pieces, 0, sizeof(board->m_pieces));
		std::memset(board->m_squares, 0, sizeof(board->m_squares));

		for (size_t i = 0; i < m_num_populated_classes; ++i)
		{
			const Piece_Class ix = m_populated_classes[i];
			const Piece_Group* info = m_groups[ix];
			const Piece_Group::Placement& list = info->squares(index[ix]);
			const size_t num_pieces = info->size();

			Bitboard color_bb = Bitboard::make_empty();

			for (size_t j = 0; j < num_pieces; ++j)
			{
				// Return false immediately if illegal
				const Square sq = list[j];
				if constexpr (!ASSUME_LEGAL)
					if (!board->is_empty(sq))
						return false;

				const Piece piece = info->piece(j);
				const Bitboard& bb = square_bb(sq);
				board->m_squares[sq] = piece;
				board->m_pieces[piece] |= bb;
				color_bb |= bb;
			}

			const Color color = piece_class_color(ix);
			board->m_pieces[piece_occupy(color)] |= color_bb;
		}

		static_assert(sizeof(board->m_piece_counts) == sizeof(m_piece_counts));
		std::memcpy(board->m_piece_counts, m_piece_counts, sizeof(m_piece_counts));
		board->m_occupied = board->m_pieces[WHITE_OCCUPY] | board->m_pieces[BLACK_OCCUPY];

		return true;
	}

	void step_to_next(In_Out_Param<Decomposed_Board_Index> index) const
	{
		for (size_t i = 0; i < m_num_populated_classes; ++i)
		{
			const Piece_Class ix = m_populated_classes[i];
			if (++index[ix] == m_num_positions_by_group[ix])
				index[ix] = Piece_Group::ZERO_INDEX;
			else
				break;
		}
	}

	void decompose_board_index(Out_Param<Decomposed_Board_Index> index, Board_Index current_pos) const
	{
		index->fill(Piece_Group::ZERO_INDEX);
		for (ptrdiff_t i = m_num_populated_classes - 1; i >= 1; --i)
		{
			const Piece_Class ix = m_populated_classes[i];
			ASSERT(m_weight_by_group[ix] != 1);
			index[ix] = narrowing_static_cast<Piece_Group::Placement_Index>(static_cast<size_t>(current_pos) / m_weight_divider_by_group[ix]);
			current_pos -= index[ix] * m_weight_by_group[ix];
		}
		index[0] = narrowing_static_cast<Piece_Group::Placement_Index>(current_pos);
	}

	NODISCARD Board_Index compose_board_index(const Decomposed_Board_Index& index_tb) const
	{
		Board_Index index = BOARD_INDEX_ZERO;
		for (size_t i = 0; i < m_num_populated_classes; ++i)
		{
			const Piece_Class ix = m_populated_classes[i];
			index += m_weight_by_group[ix] * index_tb[ix];
		}
		return index;
	}

	NODISCARD Board_Index compose_mirr_board_index(const Decomposed_Board_Index& index_tb) const
	{
		Board_Index index = BOARD_INDEX_ZERO;
		for (size_t i = 0; i < m_num_populated_classes; ++i)
		{
			const Piece_Class ix = m_populated_classes[i];
			index += m_weight_by_group[ix] * m_groups[ix]->mirr_index(index_tb[ix]);
		}
		return index;
	}

	template <typename F>
	NODISCARD Board_Index compose_board_index(F&& func) const
	{
		static_assert(std::is_same_v<decltype(&F::operator()), Piece_Group::Placement_Index(F::*)(const Piece_Group&, Piece_Class) const>);
		Board_Index index = BOARD_INDEX_ZERO;
		for (size_t i = 0; i < m_num_populated_classes; ++i)
		{
			const Piece_Class ix = m_populated_classes[i];
			index += m_weight_by_group[ix] * func(*m_groups[ix], ix);
		}
		return index;
	}

	NODISCARD Board_Index change_single_group_index(
		Board_Index pos,
		Piece_Group::Placement_Index old_index,
		Piece_Group::Placement_Index new_index,
		Piece_Class set
	) const
	{
		const ptrdiff_t diff = static_cast<ptrdiff_t>(new_index) - static_cast<ptrdiff_t>(old_index);
		return pos + diff * static_cast<ptrdiff_t>(m_weight_by_group[set]);
	}

	NODISCARD size_t num_positions() const
	{
		return m_num_positions;
	}

	NODISCARD bool both_sides_have_free_attackers() const
	{
		return m_both_sides_have_free_attackers;
	}

	NODISCARD Piece_Class compress_id() const
	{
		return m_compress_id;
	}

	NODISCARD const Piece_Group& group(Piece_Class set) const
	{
		ASSERT(m_groups[set]);
		return *m_groups[set];
	}

	NODISCARD const Piece_Group::Placement& squares(const Decomposed_Board_Index& index, Piece_Class set) const
	{
		const auto& info = group(set);
		return info.squares(index[set]);
	}

private:
	size_t m_num_positions;
	size_t m_num_populated_classes;
	Piece_Class m_populated_classes[PIECE_CLASS_NB];
	Piece_Class m_compress_id;
	bool m_both_sides_have_free_attackers;
	const Piece_Group* m_groups[PIECE_CLASS_NB];
	size_t m_num_positions_by_group[PIECE_CLASS_NB];
	size_t m_weight_by_group[PIECE_CLASS_NB];
	Divider<size_t> m_weight_divider_by_group[PIECE_CLASS_NB];
	int8_t m_piece_counts[PIECE_NB];

	NODISCARD static Piece_Class compute_compress_id(const Piece_Group* set_id[PIECE_CLASS_NB])
	{
		Piece_Class compress_id = PIECE_CLASS_NONE;
		double best_ratio = std::numeric_limits<double>::max();
		for (Piece_Class i = PIECE_CLASS_START; i < PIECE_CLASS_END; ++i)
		{
			if (set_id[i] == nullptr)
				continue;

			const double r = set_id[i]->compress_ratio();
			if (r < best_ratio)
			{
				best_ratio = r;
				compress_id = i;
			}
		}

		return compress_id;
	}
};

// NOTE: this struct is not "const thread-safe"
struct Position_For_Gen
{
	Position_For_Gen(const Piece_Config_For_Gen& info, Board_Index pos, Color turn = WHITE);

	// Constructs a child position of the passed `parent`, 
	// after a quiet move `move`, and with the corresponding board index `next_ix`.
	// The move must be quiet and consistent with next_ix.
	// If the parent had the board initialized this function is faster than creating a new instance.
	// If the move results in the board being mirrored `mirr` must be true.
	Position_For_Gen(const Position_For_Gen& parent, Move move, Board_Index next_ix, bool mirr);

	Position_For_Gen& operator++()
	{
		m_board_index += 1;
		m_epsi->step_to_next(inout_param(m_index));
		return *this;
	}

	NODISCARD bool operator<(Board_Index other_pos) const
	{
		return m_board_index < other_pos;
	}

	NODISCARD const Position& board() const
	{
		init_board<true>();
		return m_board;
	}

	NODISCARD Position& board()
	{
		init_board<true>();
		return m_board;
	}

	NODISCARD const auto& index() const
	{
		return m_index;
	}

	void get_fen(Span<char> out) const
	{
		init_board<true>();
		m_board.to_fen(out);
	}

	void set_turn(Color color)
	{
		m_turn = color;
		if (m_board_index == m_cached_board_index)
			m_board.set_turn(color);
	}

	NODISCARD bool is_legal() const
	{
		init_board<false>();
		return m_legal;
	}

	NODISCARD Board_Index board_index() const
	{
		return m_board_index;
	}

	void set_board_index(Board_Index pos)
	{
		m_epsi->decompose_board_index(out_param(m_index), pos);
		m_board_index = pos;
	}

private:
	const Piece_Config_For_Gen* m_epsi;
	Board_Index m_board_index;
	Color m_turn;
	Decomposed_Board_Index m_index;

	mutable Board_Index m_cached_board_index;
	static_assert(std::is_trivial_v<Position>);
	mutable Position m_board;
	mutable bool m_legal;
	
	template <bool ASSUME_LEGAL>
	void init_board() const
	{
		if (m_board_index == m_cached_board_index)
			return;
		
		m_legal = m_epsi->fill_board<ASSUME_LEGAL>(m_index, out_param(m_board));
		m_board.set_turn(m_turn);
		m_cached_board_index = m_board_index;
	}
};

struct EGTB_Bits
{
	using Underlying_Storage_Type = uint64_t;
	static constexpr size_t ELEMENT_BITS = sizeof(Underlying_Storage_Type) * CHAR_BIT;
	static constexpr Underlying_Storage_Type ONE = 1;
	static constexpr size_t CLEAR_BLOCK_SIZE = 1024 * 1024;

	EGTB_Bits() :
		m_num_bits(0)
	{
	}

	EGTB_Bits(size_t pos_cnt) :
		EGTB_Bits()
	{
		alloc(pos_cnt);
	}

	EGTB_Bits(const EGTB_Bits&) = delete;
	EGTB_Bits(EGTB_Bits&& other) noexcept :
		m_elements(std::move(other.m_elements)),
		m_num_bits(std::exchange(other.m_num_bits, 0))
	{
	}

	EGTB_Bits& operator=(const EGTB_Bits&) = delete;
	EGTB_Bits& operator=(EGTB_Bits&& other) noexcept
	{
		m_elements = std::move(other.m_elements);
		m_num_bits = std::exchange(other.m_num_bits, 0);
		return *this;
	}

	void clear(In_Out_Param<Thread_Pool> thread_pool)
	{
		const Span<Underlying_Storage_Type> data(m_elements);
		std::atomic<size_t> next_block_id(0);
		thread_pool->run_sync_task_on_all_threads(
			[&](size_t thread_id) {
			for (;;)
			{
				const size_t block_id = next_block_id.fetch_add(1);
				auto block = data.nth_chunk(block_id, CLEAR_BLOCK_SIZE);
				if (block.empty())
					return;
				std::memset(block.data(), 0, block.size() * sizeof(Underlying_Storage_Type));
			}
		}
		);
	}

	NODISCARD size_t size() const
	{
		return m_num_bits;
	}

	NODISCARD bool empty() const
	{
		for (const auto& element : m_elements)
			if (element)
				return false;
		return true;
	}

	void set_bit(Board_Index pos)
	{
		ASSERT(pos < m_num_bits);
		m_elements[pos / ELEMENT_BITS] |= (ONE << (pos % ELEMENT_BITS));
	}

	void clear_bit(Board_Index pos)
	{
		ASSERT(pos < m_num_bits);
		m_elements[pos / ELEMENT_BITS] &= ~(ONE << (pos % ELEMENT_BITS));
	}

	void lock_set_bit(Board_Index pos)
	{
		ASSERT(pos < m_num_bits);
		atomic_fetch_or(&m_elements[pos / ELEMENT_BITS], ONE << (pos % ELEMENT_BITS));
	}

	NODISCARD bool bit_is_set(Board_Index pos) const
	{
		ASSERT(pos < m_num_bits);
		return m_elements[pos / ELEMENT_BITS] & (ONE << (pos % ELEMENT_BITS));
	}

	struct Set_Bits_View
	{
		struct iterator_sentinel {};

		struct const_iterator
		{
			const_iterator() = default;

			const_iterator(const EGTB_Bits& provider, size_t begin, size_t end) :
				m_provider(&provider),
				m_curr_element_bits(0)
			{
				// Enforce that we won't be discarding set bits from an element.
				// Simplifies implemenation.
				if (   (begin != provider.size() && begin % ELEMENT_BITS != 0)
					|| (end != provider.size() && end % ELEMENT_BITS != 0))
					throw std::runtime_error("Insufficient alignment of begin and end bit indices for set bit iterator.");

				// -1 because we increment in operator++ to get to the first to search
				// We use ceil_div also for begin because we're either guaranteed that
				// it's exactly divisible (in which case no rounding occurs) or
				// begin == end, in which case we want m_curr_element == m_end_element.
				m_curr_element = ceil_div<size_t>(begin, ELEMENT_BITS) - 1;
				m_end_element = ceil_div<size_t>(end, ELEMENT_BITS);

				this->operator++();
			}

			const_iterator(const const_iterator&) = default;
			const_iterator(const_iterator&&) = default;

			const_iterator& operator=(const const_iterator&) = default;
			const_iterator& operator=(const_iterator&&) = default;

			NODISCARD bool is_end() const
			{
				return m_board_index == BOARD_INDEX_NONE;
			}

			NODISCARD friend bool operator==(const const_iterator& lhs, const iterator_sentinel& rhs)
			{
				return lhs.is_end();
			}

			NODISCARD friend bool operator!=(const const_iterator& lhs, const iterator_sentinel& rhs)
			{
				return !lhs.is_end();
			}

			const_iterator& operator++()
			{
				if (m_curr_element_bits == 0)
				{
					m_curr_element += 1;
					m_curr_element_bits = m_provider->find_next_nonzero_element(m_curr_element, m_end_element);
					if (m_curr_element_bits == 0)
					{
						m_board_index = BOARD_INDEX_NONE;
						return *this;
					}
				}

				m_board_index = static_cast<Board_Index>(pop_first_bit(m_curr_element_bits) + m_curr_element * ELEMENT_BITS);

				return *this;
			}

			NODISCARD Board_Index operator*() const
			{
				return m_board_index;
			}

		private:
			const EGTB_Bits* m_provider;
			size_t m_curr_element;
			size_t m_end_element;
			Underlying_Storage_Type m_curr_element_bits;
			Board_Index m_board_index;
		};

		Set_Bits_View(const EGTB_Bits& provider, size_t begin, size_t end) :
			m_provider(&provider),
			m_begin(begin),
			m_end(end)
		{
		}

		NODISCARD const_iterator begin() const
		{
			return const_iterator(*m_provider, m_begin, m_end);
		}

		NODISCARD iterator_sentinel end() const
		{
			return {};
		}

	private:
		const EGTB_Bits* m_provider;
		size_t m_begin;
		size_t m_end;
	};

	NODISCARD Set_Bits_View set_bits(size_t begin, size_t end) const
	{
		return Set_Bits_View(*this, begin, end);
	}

private:
	Huge_Array<Underlying_Storage_Type> m_elements;
	size_t m_num_bits;

	void alloc(size_t pos_cnt)
	{
		if (m_num_bits != pos_cnt)
		{
			free();
			m_num_bits = pos_cnt;
			const size_t num_elements = ceil_div(pos_cnt, ELEMENT_BITS);
			m_elements = Huge_Array<Underlying_Storage_Type>(For_Overwrite_Tag{}, num_elements);
		}

		clear();
	}

	void clear()
	{
		std::memset(m_elements.data(), 0, m_elements.size() * sizeof(Underlying_Storage_Type));
	}

	void free()
	{
		m_elements.clear();
		m_num_bits = 0;
	}

	NODISCARD Underlying_Storage_Type find_next_nonzero_element(size_t& start_idx, const size_t& end_idx) const
	{
		ASSERT(start_idx <= m_elements.size() && end_idx <= m_elements.size());
		while (start_idx < end_idx && m_elements[start_idx] == 0)
			start_idx += 1;

		if (start_idx >= end_idx)
			return 0;

		return m_elements[start_idx];
	}
};

struct EGTB_Bits_Pool
{
	EGTB_Bits_Pool(size_t pool_size, size_t bits_size) :
		m_num_bits(bits_size)
	{
		for (size_t i = 0; i < pool_size; ++i)
		{
			m_pool.emplace_back(bits_size, false);
		}
	}

	NODISCARD EGTB_Bits acquire_cleared(In_Out_Param<Thread_Pool> thread_pool)
	{
		if (m_pool.size() == 0)
			throw std::runtime_error("No bits to acquire.");

		EGTB_Bits bits = std::move(m_pool.back().first);
		const bool dirty = m_pool.back().second;
		m_pool.pop_back();

		if (dirty)
			bits.clear(thread_pool);

		return bits;
	}

	NODISCARD EGTB_Bits acquire_dirty()
	{
		if (m_pool.size() == 0)
			throw std::runtime_error("No bits to acquire.");

		EGTB_Bits bits = std::move(m_pool.back().first);
		m_pool.pop_back();

		return bits;
	}

	void release(EGTB_Bits bits)
	{
		if (bits.size() != m_num_bits)
			throw std::runtime_error("Tried to release bits of wrong size.");
		m_pool.emplace_back(std::move(bits), true);
	}

	void clear()
	{
		m_pool.clear();
	}

private:
	std::vector<std::pair<EGTB_Bits, bool>> m_pool;
	size_t m_num_bits;
};

#define VERIFY_EGTB_GEN_ACCESS_CONSISTENCY false

#if VERIFY_EGTB_GEN_ACCESS_CONSISTENCY

template <typename MainEntryT, typename... OtherEntryTs>
struct EGTB_File_For_Gen_Consistency_Check
{
private:
	template <typename EntryT, typename U, typename... Us>
	NODISCARD static constexpr uint8_t entry_index()
	{
		if constexpr (std::is_same_v<EntryT, U>)
			return 0;
		else if constexpr (sizeof...(Us))
			return 1 + entry_index<EntryT, Us...>();
	}

	template <typename EntryT>
	NODISCARD static constexpr uint8_t entry_index()
	{
		return entry_index<EntryT, MainEntryT, OtherEntryTs...>();
	}

	template <typename FlagT, typename U, typename... Us>
	NODISCARD static bool can_modify_flag(uint8_t entry_id)
	{
		if constexpr (U::template is_allowed_flag_type<FlagT>)
			if (entry_index<U>() == entry_id)
				return true;

		if constexpr (sizeof...(Us))
			return can_modify_flag<FlagT, Us...>(entry_id);
		else
			return false;
	}

	template <typename EntryT, typename U, typename... Us>
	NODISCARD static bool can_read_entry(uint8_t entry_id)
	{
		if constexpr (std::is_same_v<EntryT, U> || std::is_base_of_v<EntryT, U>)
			if (entry_index<U>() == entry_id)
				return true;

		if constexpr (sizeof...(Us))
			return can_read_entry<EntryT, Us...>(entry_id);
		else
			return false;
	}

	template <typename FlagT>
	NODISCARD static bool can_modify_flag(uint8_t entry_id)
	{
		return can_modify_flag<FlagT, MainEntryT, OtherEntryTs...>(entry_id);
	}

	template <typename EntryT>
	NODISCARD static bool can_read_entry(uint8_t entry_id)
	{
		return can_read_entry<EntryT, MainEntryT, OtherEntryTs...>(entry_id);
	}

public:
	void on_create(size_t num_entries)
	{
		active_entries = make_unique_ex<uint8_t[], Make_Unique_Ex_Flags::DEFAULT_INIT>(num_entries);
		std::memset(active_entries.get(), 0xff, num_entries);
	}

	template <typename EntryT>
	void on_write(Board_Index idx)
	{
		active_entries[idx] = entry_index<EntryT>();
	}

	template <typename EntryT>
	void on_read(Board_Index idx) const
	{
		ASSERT_ALWAYS(can_read_entry<EntryT>(active_entries[idx]));
	}

	template <typename FlagT>
	void on_flag_change(Board_Index idx) const
	{
		ASSERT_ALWAYS(can_modify_flag<FlagT>(active_entries[idx]));
	}

private:
	std::unique_ptr<uint8_t[]> active_entries;
};

#else

template <typename MainEntryT, typename... OtherEntryTs>
struct EGTB_File_For_Gen_Consistency_Check
{
	void on_create(size_t num_entries) {}

	template <typename EntryT>
	void on_write(Board_Index idx) {}

	template <typename EntryT>
	void on_read(Board_Index idx) const {}

	template <typename FlagT>
	void on_flag_change(Board_Index idx) const {}
};

#endif

template <typename MainEntryT, typename... OtherEntryTs>
struct EGTB_File_For_Gen : public EGTB_File_For_Gen_Consistency_Check<MainEntryT, OtherEntryTs...>
{
	static constexpr size_t NUM_ENTRY_VARIANTS = 1 + sizeof...(OtherEntryTs);
	static constexpr size_t ENTRY_SIZE = sizeof(MainEntryT);
	static_assert(((sizeof(OtherEntryTs) == ENTRY_SIZE) && ...));
	static_assert(ENTRY_SIZE == 1 || ENTRY_SIZE == 2 || ENTRY_SIZE == 4 || ENTRY_SIZE == 8);

	using Underlying_Entry_Type = Unsigned_Int_Of_Size<ENTRY_SIZE>;

	using Consistency = EGTB_File_For_Gen_Consistency_Check<MainEntryT, OtherEntryTs...>;

	EGTB_File_For_Gen() = default;

	~EGTB_File_For_Gen()
	{
		close();
	}

	void create(size_t sz)
	{
		Consistency::on_create(sz);
		m_entries = Huge_Array<Underlying_Entry_Type>(For_Overwrite_Tag{}, sz);
	}

	template <size_t N = NUM_ENTRY_VARIANTS>
	NODISCARD std::enable_if_t<N == 1, MainEntryT> read(Board_Index pos) const
	{
		ASSERT(pos < m_entries.size());
		Consistency::template on_read<MainEntryT>(pos);
		MainEntryT entry;
		std::memcpy(&entry, m_entries.data() + pos, sizeof(MainEntryT));
		return entry;
	}

	template<typename T, size_t N = NUM_ENTRY_VARIANTS>
	NODISCARD std::enable_if_t<N != 1, T> read(Board_Index pos) const
	{
		static_assert(   std::is_same_v<T, MainEntryT> || (std::is_same_v<T, OtherEntryTs> || ...) 
			          || std::is_base_of_v<T, MainEntryT> || (std::is_base_of_v<T, OtherEntryTs> || ...));
		ASSERT(pos < m_entries.size());
		Consistency::template on_read<T>(pos);
		T entry;
		std::memcpy(&entry, m_entries.data() + pos, sizeof(T));
		return entry;
	}

	template<typename T>
	void write(const T& tt, Board_Index pos)
	{
		static_assert(std::is_same_v<T, MainEntryT> || (std::is_same_v<T, OtherEntryTs> || ...));
		ASSERT(pos < m_entries.size());
		Consistency::template on_write<T>(pos);
		std::memcpy(m_entries.data() + pos, &tt, sizeof(T));
	}

	template<typename T>
	void lock_add_flags(Board_Index pos, T flags)
	{
		static_assert(sizeof(T) == sizeof(Underlying_Entry_Type));
		static_assert(MainEntryT::template is_allowed_flag_type<T> || (OtherEntryTs::template is_allowed_flag_type<T> || ...));
		ASSERT(pos < m_entries.size());
		Consistency::template on_flag_change<T>(pos);
		atomic_fetch_or(m_entries.data() + pos, flags);
	}

	template<typename T>
	void add_flags(Board_Index pos, T flags)
	{
		static_assert(sizeof(T) == sizeof(Underlying_Entry_Type));
		static_assert(MainEntryT::template is_allowed_flag_type<T> || (OtherEntryTs::template is_allowed_flag_type<T> || ...));
		Consistency::template on_flag_change<T>(pos);
		ASSERT(pos < m_entries.size());
		m_entries[pos] |= static_cast<Underlying_Entry_Type>(flags);
	}

	void close()
	{
		m_entries.clear();
	}

	NODISCARD Const_Span<Underlying_Entry_Type> entry_span() const
	{
		return Const_Span<Underlying_Entry_Type>(m_entries);
	}

	NODISCARD Const_Span<uint8_t> data_span() const
	{
		return Const_Span(reinterpret_cast<const uint8_t*>(m_entries.data()), m_entries.size() * ENTRY_SIZE);
	}

private:
	Huge_Array<Underlying_Entry_Type> m_entries;
};

template <>
struct EGTB_File_For_Gen<WDL_Entry>
{	
	EGTB_File_For_Gen() :
		m_num_entries(0)
	{
	}

	~EGTB_File_For_Gen()
	{
		close();
	}

	void create(size_t num_entries)
	{
		const size_t size = ceil_div(num_entries, WDL_ENTRY_PACK_RATIO);
		m_packed_entries = Huge_Array<Packed_WDL_Entries>(For_Overwrite_Tag{}, size);
		this->m_num_entries = num_entries;

		// Fill padding. We use DRAW instead of ILLEGAL to maintain backwards compatibility.
		for (size_t i = num_entries; i < size * WDL_ENTRY_PACK_RATIO; ++i)
			set_wdl_entry(m_packed_entries[i / 4], i % 4, WDL_Entry::DRAW);
	}

	void write(Board_Index pos, WDL_Entry new_value)
	{
		ASSERT(pos < m_num_entries);
		set_wdl_entry(m_packed_entries[pos / 4], pos % 4, new_value);
	}

	void close()
	{
		m_packed_entries.clear();
	}

	NODISCARD Const_Span<Packed_WDL_Entries> entry_span() const
	{
		return Const_Span<Packed_WDL_Entries>(m_packed_entries);
	}

	NODISCARD Span<Packed_WDL_Entries> entry_span()
	{
		return Span<Packed_WDL_Entries>(m_packed_entries);
	}

private:
	Huge_Array<Packed_WDL_Entries> m_packed_entries;
	size_t m_num_entries;
};

using WDL_File_For_Gen = EGTB_File_For_Gen<WDL_Entry>;
using DTC_File_For_Gen = EGTB_File_For_Gen<DTC_Intermediate_Entry, DTC_Final_Entry>;
using DTM_File_For_Gen = EGTB_File_For_Gen<DTM_Intermediate_Entry, DTM_Final_Entry>;

struct Shared_Board_Index_Iterator
{
private:
	template <typename IterT>
	struct Sentineled_Self_Iterator
	{
		struct iterator_sentinel {};

		NODISCARD friend bool operator==(const IterT& lhs, const iterator_sentinel& rhs)
		{
			return lhs.is_end();
		}

		NODISCARD friend bool operator!=(const IterT& lhs, const iterator_sentinel& rhs)
		{
			return !lhs.is_end();
		}

		NODISCARD IterT& begin()
		{
			return static_cast<IterT&>(*this);
		}

		NODISCARD iterator_sentinel end()
		{
			return {};
		}
	};

public:

	struct Chunk_Iterator : Sentineled_Self_Iterator<Chunk_Iterator>
	{
		explicit Chunk_Iterator(Shared_Board_Index_Iterator& provider) :
			m_provider(&provider)
		{
			this->operator++();
		}

		NODISCARD bool is_end() const
		{
			return m_chunk_start == m_chunk_end;
		}

		Chunk_Iterator& operator++()
		{
			auto [s, e] = this->m_provider->next_range();
			m_chunk_start = s;
			m_chunk_end = e;
			return *this;
		}

		NODISCARD std::pair<Board_Index, Board_Index> operator*() const
		{
			return { m_chunk_start, m_chunk_end };
		}

	private:
		Shared_Board_Index_Iterator* m_provider;
		Board_Index m_chunk_start;
		Board_Index m_chunk_end;
	};

	struct Index_Iterator : Sentineled_Self_Iterator<Index_Iterator>
	{
		explicit Index_Iterator(Shared_Board_Index_Iterator& provider) :
			m_provider(&provider)
		{
			auto [s, e] = this->m_provider->next_range();
			m_chunk_curr = s;
			m_chunk_end = e;
		}

		NODISCARD bool is_end() const
		{
			return m_chunk_curr == m_chunk_end;
		}

		Index_Iterator& operator++()
		{
			m_chunk_curr += 1;

			if (m_chunk_curr == m_chunk_end)
			{
				auto [s, e] = this->m_provider->next_range();
				m_chunk_curr = s;
				m_chunk_end = e;
			}

			return *this;
		}

		NODISCARD Board_Index operator*() const
		{
			return m_chunk_curr;
		}

	private:
		Shared_Board_Index_Iterator* m_provider;
		Board_Index m_chunk_curr;
		Board_Index m_chunk_end;
	};

	struct Sparse_Index_Iterator : Sentineled_Self_Iterator<Sparse_Index_Iterator>
	{
		Sparse_Index_Iterator(Shared_Board_Index_Iterator& provider, const EGTB_Bits& bits) :
			m_provider(&provider),
			m_bits(&bits)
		{
			ASSERT(provider.num_indices() == bits.size());

			for (;;)
			{
				auto [s, e] = this->m_provider->next_range();
				m_set_bits_curr = m_bits->set_bits(s, e).begin(); // This is okay, because Set_Bits_View is just a view.

				if (s == e || !m_set_bits_curr.is_end())
					break;
			}
		}

		NODISCARD bool is_end() const
		{
			return m_set_bits_curr.is_end();
		}

		Sparse_Index_Iterator& operator++()
		{
			++m_set_bits_curr;
			while (m_set_bits_curr.is_end())
			{
				auto [s, e] = this->m_provider->next_range();
				if (s == e)
					break;

				m_set_bits_curr = m_bits->set_bits(s, e).begin();
			}

			return *this;
		}

		NODISCARD Board_Index operator*() const
		{
			return *m_set_bits_curr;
		}

	private:
		Shared_Board_Index_Iterator* m_provider;
		const EGTB_Bits* m_bits;
		EGTB_Bits::Set_Bits_View::const_iterator m_set_bits_curr;
	};

	struct Board_Iterator : Sentineled_Self_Iterator<Board_Iterator>
	{
		Board_Iterator(Shared_Board_Index_Iterator& provider, const Piece_Config_For_Gen& epsi, Color turn = WHITE) :
			m_provider(&provider),
			m_chunk(this->m_provider->next_range()),
			m_pos_gen(epsi, m_chunk.first, turn) // doesn't fail on illegal index so it's fine
		{
		}

		NODISCARD bool is_end() const
		{
			return m_chunk.first == m_chunk.second;
		}

		Board_Iterator& operator++()
		{
			m_chunk.first += 1;

			if (m_chunk.first == m_chunk.second)
			{
				m_chunk = this->m_provider->next_range();
				if (!is_end())
					m_pos_gen.set_board_index(m_chunk.first);
			}
			else
				++m_pos_gen;

			return *this;
		}

		NODISCARD const Position_For_Gen& operator*() const
		{
			return m_pos_gen;
		}

		NODISCARD Position_For_Gen& operator*()
		{
			return m_pos_gen;
		}

	private:
		Shared_Board_Index_Iterator* m_provider;
		std::pair<Board_Index, Board_Index> m_chunk;
		Position_For_Gen m_pos_gen;
	};

	Shared_Board_Index_Iterator(Board_Index start_idx, Board_Index end_idx, size_t chunk_size) :
		m_start_idx(start_idx),
		m_end_idx(end_idx),
		m_chunk_size(chunk_size),
		m_current_chunk_index(0)
	{
	}

	Shared_Board_Index_Iterator(const Shared_Board_Index_Iterator&) = delete;

	NODISCARD std::pair<Board_Index, Board_Index> next_range()
	{
		const size_t chunk_index = m_current_chunk_index.fetch_add(1);

		// This should not happen because it's 64-bit index. 
		// Ideally we would do a saturating fetch add, 
		// but it would require either locking or more complex logic.
		ASSERT(chunk_index != std::numeric_limits<size_t>::max());

		const Board_Index chunk_start = std::min(m_start_idx + chunk_index * m_chunk_size, m_end_idx);
		const Board_Index chunk_end = std::min(chunk_start + m_chunk_size, m_end_idx);

		return { chunk_start, chunk_end };
	}

	NODISCARD Chunk_Iterator chunks()
	{
		return Chunk_Iterator(*this);
	}

	NODISCARD Index_Iterator indices()
	{
		return Index_Iterator(*this);
	}

	NODISCARD Sparse_Index_Iterator indices(const EGTB_Bits& bits)
	{
		return Sparse_Index_Iterator(*this, bits);
	}

	NODISCARD Board_Iterator boards(const Piece_Config_For_Gen& epsi, Color turn = WHITE)
	{
		return Board_Iterator(*this, epsi, turn);
	}

	NODISCARD size_t num_indices() const
	{
		return m_end_idx - m_start_idx;
	}

private:
	Board_Index m_start_idx;
	Board_Index m_end_idx;
	size_t m_chunk_size;
	std::atomic<size_t> m_current_chunk_index;
};

struct EGTB_Generation_Info
{
	size_t num_positions;
	size_t uncompressed_size;
	size_t uncompressed_sub_tb_size;
	size_t memory_required_for_generation;
};

struct EGTB_Generator
{
	EGTB_Generator(const Piece_Config& ps);

	NODISCARD inline Fixed_Vector<Color, 2> table_colors() const
	{
		const size_t table_num = m_is_symmetric ? 1 : 2;
		return ::egtb_table_colors(table_num);
	}

protected:
	Piece_Config_For_Gen m_epsi;

	std::map<Material_Key, Piece_Config_For_Gen> m_sub_epsi_by_material;

	const Piece_Config_For_Gen* m_sub_epsi_by_capture[PIECE_NB];
	Color m_sub_read_color_by_capture[PIECE_NB];
	bool m_sub_needs_mirror_by_capture[PIECE_NB];

	bool m_is_symmetric;

	NODISCARD Board_Index next_cap_index(const Position_For_Gen& pos_for_gen, Move move) const;
	NODISCARD Board_Index next_quiet_index(const Position_For_Gen& pos_for_gen, Move move) const;
	NODISCARD Board_Index next_quiet_index(const Position_For_Gen& pos_for_gen, Move move, Out_Param<bool> mirr) const;
	NODISCARD Fixed_Vector<Board_Index, 2> next_quiet_index_with_mirror(const Position_For_Gen& pos_for_gen, Move move) const;

	NODISCARD Shared_Board_Index_Iterator make_gen_iterator() const;
};
