#pragma once

#include "egtb.h"
#include "egtb_paged.h"

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

	// The mixed-radix weight of a group's digit in the board index.
	NODISCARD size_t weight(Piece_Class set) const
	{
		ASSERT(m_groups[set]);
		return m_weight_by_group[set];
	}

	// The radix of a group's digit. This is compress_size() for the
	// compression group and table_size() for every other group.
	NODISCARD size_t num_positions_in_group(Piece_Class set) const
	{
		ASSERT(m_groups[set]);
		return m_num_positions_by_group[set];
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
		m_num_bits(std::exchange(other.m_num_bits, 0)),
		m_shared_element_writes(std::exchange(other.m_shared_element_writes, false))
	{
	}

	EGTB_Bits& operator=(const EGTB_Bits&) = delete;
	EGTB_Bits& operator=(EGTB_Bits&& other) noexcept
	{
		m_elements = std::move(other.m_elements);
		m_num_bits = std::exchange(other.m_num_bits, 0);
		m_shared_element_writes = std::exchange(other.m_shared_element_writes, false);
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

	// Whether set_bit/clear_bit must tolerate two workers sharing an element.
	// The flat sweep's chunks are multiples of ELEMENT_BITS, so no element is
	// ever shared; paged dispatch hands out slice ranges, whose units start and
	// end wherever the slice weight puts them.
	void set_shared_element_writes(bool shared)
	{
		m_shared_element_writes = shared;
	}

	void set_bit(Board_Index pos)
	{
		ASSERT(pos < m_num_bits);
		if (m_shared_element_writes)
			lock_set_bit(pos);
		else
			m_elements[pos / ELEMENT_BITS] |= (ONE << (pos % ELEMENT_BITS));
	}

	void clear_bit(Board_Index pos)
	{
		ASSERT(pos < m_num_bits);
		if (m_shared_element_writes)
			atomic_fetch_and(&m_elements[pos / ELEMENT_BITS], ~(ONE << (pos % ELEMENT_BITS)));
		else
			m_elements[pos / ELEMENT_BITS] &= ~(ONE << (pos % ELEMENT_BITS));
	}

	void lock_set_bit(Board_Index pos)
	{
		ASSERT(pos < m_num_bits);
		Underlying_Storage_Type& element = m_elements[pos / ELEMENT_BITS];
		const Underlying_Storage_Type mask = ONE << (pos % ELEMENT_BITS);

		// The frontier is marked once per predecessor and predecessors are
		// shared, so most of these find the bit already set. Testing first
		// leaves the line shared, where the atomic would take it exclusive
		// every time. Within a pass these bits are only ever set, never
		// cleared, so a skipped OR cannot lose a mark.
		if ((element & mask) != 0)
			return;

		atomic_fetch_or(&element, mask);
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

			// [begin, end) need not be element-aligned: the partially covered
			// first and last elements are masked, and only the fully covered
			// interior is bulk-scanned.
			const_iterator(const EGTB_Bits& provider, size_t begin, size_t end) :
				m_provider(&provider),
				m_curr_element_bits(0)
			{
				if (begin >= end)
				{
					m_curr_element = 0;
					m_last_element = 0;
					m_last_mask = 0;
					m_board_index = BOARD_INDEX_NONE;
					return;
				}

				ASSERT(end <= provider.size());

				m_curr_element = begin / ELEMENT_BITS;
				m_last_element = (end - 1) / ELEMENT_BITS;

				const size_t end_bit = end % ELEMENT_BITS;
				m_last_mask = end_bit == 0
					? ~static_cast<Underlying_Storage_Type>(0)
					: ((ONE << end_bit) - 1);

				const Underlying_Storage_Type first_mask =
					~static_cast<Underlying_Storage_Type>(0) << (begin % ELEMENT_BITS);

				m_curr_element_bits = m_provider->element(m_curr_element) & first_mask;
				if (m_curr_element == m_last_element)
					m_curr_element_bits &= m_last_mask;

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
				while (m_curr_element_bits == 0)
				{
					if (m_curr_element >= m_last_element)
					{
						m_board_index = BOARD_INDEX_NONE;
						return *this;
					}

					m_curr_element += 1;

					if (m_curr_element < m_last_element)
					{
						m_curr_element_bits =
							m_provider->find_next_nonzero_element(m_curr_element, m_last_element);
						if (m_curr_element_bits != 0)
							break;
						// The scan is exhausted, so it stopped on the last element.
					}

					ASSERT(m_curr_element == m_last_element);
					m_curr_element_bits = m_provider->element(m_last_element) & m_last_mask;
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
			size_t m_last_element;                      // inclusive
			Underlying_Storage_Type m_last_mask;
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
	bool m_shared_element_writes = false;

	NODISCARD Underlying_Storage_Type element(size_t idx) const
	{
		return m_elements[idx];
	}

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
	// `shared_element_writes` is forwarded to every bitset handed out.
	EGTB_Bits_Pool(size_t pool_size, size_t bits_size, bool shared_element_writes) :
		m_num_bits(bits_size),
		m_shared_element_writes(shared_element_writes)
	{
		for (size_t i = 0; i < pool_size; ++i)
		{
			m_pool.emplace_back(bits_size, false);
			m_pool.back().first.set_shared_element_writes(shared_element_writes);
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
		bits.set_shared_element_writes(m_shared_element_writes);
		m_pool.emplace_back(std::move(bits), true);
	}

	void clear()
	{
		m_pool.clear();
	}

private:
	std::vector<std::pair<EGTB_Bits, bool>> m_pool;
	size_t m_num_bits;
	bool m_shared_element_writes;
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

// A distance/entry array for generation, in one of two modes.
//
//   FLAT  - one contiguous Huge_Array indexed directly by Board_Index, chosen
//           whenever the table fits the budget.
//   PAGED - cut along the slice dimension into pages that spill to `tmpdir`.
//           Board_Index is translated to (slice, offset) and then to a page.
//
// Logical indexing is identical in both modes: nothing is renumbered.
template <typename MainEntryT, typename... OtherEntryTs>
struct EGTB_File_For_Gen :
	public EGTB_File_For_Gen_Consistency_Check<MainEntryT, OtherEntryTs...>,
	public Pageable_Table
{
	static constexpr size_t NUM_ENTRY_VARIANTS = 1 + sizeof...(OtherEntryTs);
	static constexpr size_t ENTRY_SIZE = sizeof(MainEntryT);
	static_assert(((sizeof(OtherEntryTs) == ENTRY_SIZE) && ...));
	static_assert(ENTRY_SIZE == 1 || ENTRY_SIZE == 2 || ENTRY_SIZE == 4 || ENTRY_SIZE == 8);

	using Underlying_Entry_Type = Unsigned_Int_Of_Size<ENTRY_SIZE>;

	using Consistency = EGTB_File_For_Gen_Consistency_Check<MainEntryT, OtherEntryTs...>;

	EGTB_File_For_Gen() = default;

	~EGTB_File_For_Gen() override
	{
		// Covers the exception path too, so an aborted run leaves nothing
		// behind in tmpdir.
		remove_scratch_files();
		close();
	}

	// Flat mode: the whole table resident, indexed directly.
	void create(size_t sz)
	{
		Consistency::on_create(sz);
		m_num_entries = sz;
		m_paged = false;
		m_entries = Huge_Array<Underlying_Entry_Type>(For_Overwrite_Tag{}, sz);
	}

	// Paged mode. Page files are named "<scratch_prefix>.p<page>"; `magic`
	// guards against a stale file from another table being read back.
	void create_paged(
		const Slice_Layout& slices,
		const Page_Layout& pages,
		std::filesystem::path scratch_prefix,
		uint64_t magic
	)
	{
		Consistency::on_create(slices.num_entries());

		m_paged = true;
		m_num_entries = slices.num_entries();
		m_slices = slices;
		m_pages = pages;
		m_scratch_prefix = std::move(scratch_prefix);
		m_magic = magic;

		m_page_data.clear();
		m_page_data.resize(m_pages.num_pages());
		m_page_base.assign(m_pages.num_pages(), nullptr);
		m_dirty.assign(m_pages.num_pages(), 0);

		// A page left by an aborted run has a valid magic and would be read back
		// as live data.
		remove_scratch_files();

		// Precomputed physical placement of every slice, in one array so that
		// the accessor needs a single lookup and no division beyond the ones in
		// Slice_Layout::locate.
		m_placement.resize(m_slices.num_slices());
		for (size_t s = 0; s < m_slices.num_slices(); ++s)
		{
			const size_t page = m_pages.page_of_slice(s);
			m_placement[s] = Slice_Placement{
				(s - m_pages.first_slice_of_page(page)) * m_slices.slice_size(),
				page
			};
		}
	}

	template <size_t N = NUM_ENTRY_VARIANTS>
	NODISCARD std::enable_if_t<N == 1, MainEntryT> read(Board_Index pos) const
	{
		Consistency::template on_read<MainEntryT>(pos);
		MainEntryT entry;
		std::memcpy(&entry, entry_ptr(pos), sizeof(MainEntryT));
		return entry;
	}

	template<typename T, size_t N = NUM_ENTRY_VARIANTS>
	NODISCARD std::enable_if_t<N != 1, T> read(Board_Index pos) const
	{
		static_assert(   std::is_same_v<T, MainEntryT> || (std::is_same_v<T, OtherEntryTs> || ...) 
			          || std::is_base_of_v<T, MainEntryT> || (std::is_base_of_v<T, OtherEntryTs> || ...));
		Consistency::template on_read<T>(pos);
		T entry;
		std::memcpy(&entry, entry_ptr(pos), sizeof(T));
		return entry;
	}

	template<typename T>
	void write(const T& tt, Board_Index pos)
	{
		static_assert(std::is_same_v<T, MainEntryT> || (std::is_same_v<T, OtherEntryTs> || ...));
		Consistency::template on_write<T>(pos);
		std::memcpy(entry_ptr_for_write(pos), &tt, sizeof(T));
	}

	template<typename T>
	void lock_add_flags(Board_Index pos, T flags)
	{
		static_assert(sizeof(T) == sizeof(Underlying_Entry_Type));
		static_assert(MainEntryT::template is_allowed_flag_type<T> || (OtherEntryTs::template is_allowed_flag_type<T> || ...));
		Consistency::template on_flag_change<T>(pos);
		atomic_fetch_or(entry_ptr_for_write(pos), flags);
	}

	template<typename T>
	void add_flags(Board_Index pos, T flags)
	{
		static_assert(sizeof(T) == sizeof(Underlying_Entry_Type));
		static_assert(MainEntryT::template is_allowed_flag_type<T> || (OtherEntryTs::template is_allowed_flag_type<T> || ...));
		Consistency::template on_flag_change<T>(pos);
		*entry_ptr_for_write(pos) |= static_cast<Underlying_Entry_Type>(flags);
	}

	void close()
	{
		m_entries.clear();
		m_page_data.clear();
		m_page_base.clear();
		m_dirty.clear();
		m_placement.clear();
		m_paged = false;
		m_num_entries = 0;
	}

	void remove_scratch_files()
	{
		if (!m_paged)
			return;
		std::error_code ec;
		for (size_t p = 0; p < m_pages.num_pages(); ++p)
			std::filesystem::remove(page_path(p), ec);
	}

	// Flat mode only; paged tables are swept in logical index order instead.
	NODISCARD Const_Span<Underlying_Entry_Type> entry_span() const
	{
		ASSERT(!m_paged);
		return Const_Span<Underlying_Entry_Type>(m_entries);
	}

	NODISCARD Const_Span<uint8_t> data_span() const
	{
		ASSERT(!m_paged);
		return Const_Span(reinterpret_cast<const uint8_t*>(m_entries.data()), m_entries.size() * ENTRY_SIZE);
	}

	// Pageable_Table.

	NODISCARD size_t num_pages() const override
	{
		return m_paged ? m_pages.num_pages() : 0;
	}

	NODISCARD bool is_page_resident(size_t page) const override
	{
		ASSERT(m_paged);
		return m_page_data[page].size() != 0;
	}

	NODISCARD bool is_page_dirty(size_t page) const override
	{
		ASSERT(m_paged);
		return m_dirty[page] != 0;
	}

	void load_page(size_t page) override
	{
		ASSERT(m_paged);
		if (is_page_resident(page))
			return;

		const std::filesystem::path path = page_path(page);
		const bool spilled = std::filesystem::exists(path);

		// Uninitialized on first touch, exactly as the flat path is: the init
		// pass writes every entry before anything reads it.
		m_page_data[page] = Huge_Array<Underlying_Entry_Type>(For_Overwrite_Tag{}, page_entries(page));

		if (spilled)
			load_page_raw(
				Span<uint8_t>(reinterpret_cast<uint8_t*>(m_page_data[page].data()), page_bytes(page)),
				path,
				m_magic);

		m_page_base[page] = m_page_data[page].data();
		m_dirty[page] = 0;
	}

	void evict_page(size_t page) override
	{
		ASSERT(m_paged);
		if (!is_page_resident(page))
			return;

		if (m_dirty[page] != 0)
		{
			save_page_raw(
				Const_Span<uint8_t>(reinterpret_cast<const uint8_t*>(m_page_data[page].data()), page_bytes(page)),
				page_path(page),
				m_magic);
			m_dirty[page] = 0;
		}

		m_page_base[page] = nullptr;
		m_page_data[page] = Huge_Array<Underlying_Entry_Type>{};
	}

private:
	bool m_paged = false;
	size_t m_num_entries = 0;

	// FLAT mode.
	Huge_Array<Underlying_Entry_Type> m_entries;

	// PAGED mode.
	Slice_Layout m_slices;
	Page_Layout m_pages;
	std::vector<Huge_Array<Underlying_Entry_Type>> m_page_data;
	// Each resident page's base pointer, null otherwise. Redundant with
	// m_page_data, but it saves the accessor one dependent load.
	std::vector<Underlying_Entry_Type*> m_page_base;
	std::vector<uint8_t> m_dirty;

	struct Slice_Placement
	{
		size_t base_in_page;
		size_t page;
	};
	std::vector<Slice_Placement> m_placement;
	std::filesystem::path m_scratch_prefix;
	uint64_t m_magic = 0;

	NODISCARD size_t page_entries(size_t page) const
	{
		return m_pages.slices_in_page(page) * m_slices.slice_size();
	}

	NODISCARD size_t page_bytes(size_t page) const
	{
		return page_entries(page) * ENTRY_SIZE;
	}

	NODISCARD std::filesystem::path page_path(size_t page) const
	{
		return m_scratch_prefix.string() + ".p" + std::to_string(page);
	}

	// The single point that translates a logical Board_Index into storage. In
	// paged mode the containing page must be pinned by the caller's dispatch,
	// and the returned pointer is valid only for that lease.
	NODISCARD INLINE const Underlying_Entry_Type* entry_ptr(Board_Index pos) const
	{
		ASSERT(static_cast<size_t>(pos) < m_num_entries);
		if (!m_paged)
			return m_entries.data() + static_cast<size_t>(pos);

		const Slice_Layout::Location loc = m_slices.locate(pos);
		const Slice_Placement& at = m_placement[loc.slice];
		const Underlying_Entry_Type* const base = m_page_base[at.page];
		ASSERT(base != nullptr);
		return base + at.base_in_page + loc.offset;
	}

	// As above, and marks the page dirty, so a page is spilled on eviction only
	// if something actually wrote to it.
	NODISCARD INLINE Underlying_Entry_Type* entry_ptr_for_write(Board_Index pos)
	{
		ASSERT(static_cast<size_t>(pos) < m_num_entries);
		if (!m_paged)
			return m_entries.data() + static_cast<size_t>(pos);

		const Slice_Layout::Location loc = m_slices.locate(pos);
		const Slice_Placement& at = m_placement[loc.slice];
		Underlying_Entry_Type* const base = m_page_base[at.page];
		ASSERT(base != nullptr);
		if (m_dirty[at.page] == 0)
			m_dirty[at.page] = 1;
		return base + at.base_in_page + loc.offset;
	}
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
		m_num_entries = num_entries;

		// Fill padding. We use DRAW instead of ILLEGAL to maintain backwards compatibility.
		for (size_t i = num_entries; i < size * WDL_ENTRY_PACK_RATIO; ++i)
			set_wdl_entry(m_packed_entries[i / WDL_ENTRY_PACK_RATIO],
				i % WDL_ENTRY_PACK_RATIO, WDL_Entry::DRAW);
	}

	// Four entries share a byte, so this read-modify-write is only safe while no
	// two workers hold neighbouring positions. Its one caller is the unpaged
	// sweep, whose chunks are multiples of the pack ratio; a paged run derives
	// the projection into a buffer of its own (DTC_Generator::next_wdl_band).
	void write(Board_Index pos, WDL_Entry new_value)
	{
		ASSERT(pos < m_num_entries);
		set_wdl_entry(m_packed_entries[pos / WDL_ENTRY_PACK_RATIO],
			pos % WDL_ENTRY_PACK_RATIO, new_value);
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

// Shared work source for a generation pass.
//
// A plan is a set of equally long *runs* of consecutive Board_Index values:
//
//   FLAT  - a single run covering [start, end), what every unpaged pass uses.
//   SLICE - one run per `high` step, each covering the logical indices of
//           slices [slice_begin, slice_end) at that step. Adjacent slices are
//           adjacent in the logical index at fixed `high`, so a page of
//           adjacent slices needs one run per step rather than one per index.
//
// Runs are split into work units of at most `chunk_size` indices and handed
// out in batches, so that a long run is shared out rather than given whole to
// one worker and a short run does not cost an atomic per index.
//
// Consumers only ever see (begin, end) pairs, so the four sub-iterators below
// are plan-agnostic.
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
	// Per-consumer cursor over the shared provider's work units.
	struct Run_Source
	{
		explicit Run_Source(Shared_Board_Index_Iterator& provider) :
			m_provider(&provider)
		{
		}

		NODISCARD std::pair<Board_Index, Board_Index> next_range()
		{
			if (m_next == m_batch_end)
			{
				const auto [first, last] = m_provider->claim_unit_batch();
				m_next = first;
				m_batch_end = last;
				if (m_next == m_batch_end)
					return m_provider->empty_range();
			}
			return m_provider->unit(m_next++);
		}

	private:
		Shared_Board_Index_Iterator* m_provider;
		size_t m_next = 0;
		size_t m_batch_end = 0;
	};

	struct Chunk_Iterator : Sentineled_Self_Iterator<Chunk_Iterator>
	{
		explicit Chunk_Iterator(Shared_Board_Index_Iterator& provider) :
			m_runs(provider)
		{
			this->operator++();
		}

		NODISCARD bool is_end() const
		{
			return m_chunk_start == m_chunk_end;
		}

		Chunk_Iterator& operator++()
		{
			auto [s, e] = m_runs.next_range();
			m_chunk_start = s;
			m_chunk_end = e;
			return *this;
		}

		NODISCARD std::pair<Board_Index, Board_Index> operator*() const
		{
			return { m_chunk_start, m_chunk_end };
		}

	private:
		Run_Source m_runs;
		Board_Index m_chunk_start;
		Board_Index m_chunk_end;
	};

	struct Index_Iterator : Sentineled_Self_Iterator<Index_Iterator>
	{
		explicit Index_Iterator(Shared_Board_Index_Iterator& provider) :
			m_runs(provider)
		{
			auto [s, e] = m_runs.next_range();
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
				auto [s, e] = m_runs.next_range();
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
		Run_Source m_runs;
		Board_Index m_chunk_curr;
		Board_Index m_chunk_end;
	};

	struct Sparse_Index_Iterator : Sentineled_Self_Iterator<Sparse_Index_Iterator>
	{
		Sparse_Index_Iterator(Shared_Board_Index_Iterator& provider, const EGTB_Bits& bits) :
			m_runs(provider),
			m_bits(&bits)
		{
			ASSERT(provider.index_space() == bits.size());

			for (;;)
			{
				auto [s, e] = m_runs.next_range();
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
				auto [s, e] = m_runs.next_range();
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
		Run_Source m_runs;
		const EGTB_Bits* m_bits;
		EGTB_Bits::Set_Bits_View::const_iterator m_set_bits_curr;
	};

	struct Board_Iterator : Sentineled_Self_Iterator<Board_Iterator>
	{
		Board_Iterator(Shared_Board_Index_Iterator& provider, const Piece_Config_For_Gen& epsi, Color turn = WHITE) :
			m_runs(provider),
			m_chunk(m_runs.next_range()),
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
				m_chunk = m_runs.next_range();
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
		Run_Source m_runs;
		std::pair<Board_Index, Board_Index> m_chunk;
		Position_For_Gen m_pos_gen;
	};

	// FLAT plan: one run covering [start, end).
	Shared_Board_Index_Iterator(Board_Index start_idx, Board_Index end_idx, size_t chunk_size)
	{
		init(start_idx, 0, static_cast<size_t>(end_idx - start_idx), 1,
			chunk_size, static_cast<size_t>(end_idx));
	}

	// SLICE plan: one run per `high` step, each holding the logical indices of
	// slices [slice_begin, slice_end) at that step.
	Shared_Board_Index_Iterator(
		const Slice_Layout& layout,
		size_t slice_begin,
		size_t slice_end,
		size_t chunk_size
	)
	{
		ASSERT(slice_begin < slice_end);
		ASSERT(slice_end <= layout.num_slices());

		init(
			layout.run_begin(slice_begin, 0),
			layout.low_weight() * layout.num_slices(),
			layout.low_weight() * (slice_end - slice_begin),
			layout.num_high(),
			chunk_size,
			layout.num_entries());
	}

	Shared_Board_Index_Iterator(const Shared_Board_Index_Iterator&) = delete;

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

	// Indices this plan yields.
	NODISCARD size_t num_indices() const
	{
		return m_num_indices;
	}

	// Size of the whole index space the plan is a subset of. Bitsets stay
	// full-size and resident, so they are sized against this, not num_indices.
	NODISCARD size_t index_space() const
	{
		return m_index_space;
	}

private:
	Board_Index m_start_idx = BOARD_INDEX_ZERO;
	size_t m_run_stride = 0;
	size_t m_run_length = 0;
	size_t m_num_runs = 0;

	size_t m_unit_length = 0;
	size_t m_units_per_run = 0;
	size_t m_num_units = 0;
	// Units handed out per atomic step: one when a unit is a full chunk,
	// several when the runs are short.
	size_t m_units_per_batch = 1;

	size_t m_num_indices = 0;
	size_t m_index_space = 0;
	std::atomic<size_t> m_next_batch{ 0 };

	void init(
		Board_Index start_idx,
		size_t run_stride,
		size_t run_length,
		size_t num_runs,
		size_t chunk_size,
		size_t index_space
	)
	{
		m_start_idx = start_idx;
		m_run_stride = run_stride;
		m_run_length = run_length;
		m_num_runs = run_length == 0 ? 0 : num_runs;

		m_unit_length = std::max<size_t>(1, std::min(run_length, chunk_size));
		m_units_per_run = ceil_div(run_length, m_unit_length);
		m_num_units = m_num_runs * m_units_per_run;
		m_units_per_batch = std::max<size_t>(1, chunk_size / m_unit_length);

		m_num_indices = m_num_runs * run_length;
		m_index_space = index_space;
	}

	NODISCARD std::pair<Board_Index, Board_Index> empty_range() const
	{
		return { m_start_idx, m_start_idx };
	}

	NODISCARD std::pair<size_t, size_t> claim_unit_batch()
	{
		const size_t batch = m_next_batch.fetch_add(1, std::memory_order_relaxed);

		// This should not happen because it's a 64-bit counter. Ideally we
		// would do a saturating fetch add, but it would require either locking
		// or more complex logic.
		ASSERT(batch != std::numeric_limits<size_t>::max());

		const size_t first = std::min(batch * m_units_per_batch, m_num_units);
		const size_t last = std::min(first + m_units_per_batch, m_num_units);
		return { first, last };
	}

	NODISCARD std::pair<Board_Index, Board_Index> unit(size_t unit_index) const
	{
		ASSERT(unit_index < m_num_units);

		const size_t run = m_units_per_run == 1 ? unit_index : unit_index / m_units_per_run;
		const size_t offset = (unit_index - run * m_units_per_run) * m_unit_length;

		const Board_Index begin = m_start_idx + run * m_run_stride + offset;
		return { begin, begin + std::min(m_unit_length, m_run_length - offset) };
	}
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

	// Slice geometry of this material. Always valid; only used when paging.
	Slice_Layout m_slice_layout;

	NODISCARD bool is_paged() const { return m_paged; }

	// Switches this generator to paged dispatch over `tables`, indexed by
	// color. `capacity_pages` caps residency, raised to the generation floor
	// if it is below it, since a dispatch's whole pin set must fit.
	void enable_paging(
		const Page_Layout& pages,
		std::vector<Pageable_Table*> tables,
		size_t capacity_pages
	);

	void disable_paging();

	NODISCARD const Page_Layout& page_layout() const { return m_page_layout; }
	NODISCARD Page_Cache& page_cache() const { return *m_page_cache; }

	// Leases the pages of `table`'s closed one-ply slice reach around `pos` for
	// the lifetime of the returned guard, on top of whatever the current
	// dispatch already holds. One branch of the check/chase look-ahead reads a
	// slice ply past what its dispatch declares; declaring that reach up front
	// would widen the floor for every dispatch of the pass. The pass is
	// single-threaded, so the lease costs only the pager's bookkeeping.
	NODISCARD Pinned_Pages lease_one_ply_reach(Color table, Board_Index pos) const;

	NODISCARD Board_Index next_cap_index(const Position_For_Gen& pos_for_gen, Move move) const;
	NODISCARD Board_Index next_quiet_index(const Position_For_Gen& pos_for_gen, Move move) const;
	NODISCARD Board_Index next_quiet_index(const Position_For_Gen& pos_for_gen, Move move, Out_Param<bool> mirr) const;
	NODISCARD Fixed_Vector<Board_Index, 2> next_quiet_index_with_mirror(const Position_For_Gen& pos_for_gen, Move move) const;

	NODISCARD Shared_Board_Index_Iterator make_gen_iterator() const;
	NODISCARD Shared_Board_Index_Iterator make_page_iterator(size_t page) const;

	// Memory the final logical-order sweep may use for one band. The page cache
	// is flushed by then, so the pages' budget is free.
	NODISCARD size_t sweep_band_budget_bytes() const { return m_sweep_band_budget_bytes; }
	void set_sweep_band_budget_bytes(size_t bytes) { m_sweep_band_budget_bytes = bytes; }

	// Runs `body(iterator)` over the whole index space, once per dispatch.
	//
	// Unpaged, or for a pass that dereferences no distance entry, there is a
	// single dispatch over the flat index space. Paged, the space is dispatched
	// one page of the mover's table at a time, with exactly the pages that pass
	// can touch pinned for the duration. Pins are taken and dropped by this
	// thread around the parallel region, so a worker can never hold a pointer
	// into a page whose lease has ended.
	//
	// Results from every dispatch are concatenated; the callers either any_of()
	// them or consolidate them, both of which are order independent.
	template <typename Fn>
	auto run_phase(In_Out_Param<Thread_Pool> thread_pool, Color me, Phase_Pages pages, Fn&& body)
	{
		using Result = decltype(body(std::declval<Shared_Board_Index_Iterator&>()));

		if constexpr (std::is_same_v<Result, void>)
		{
			for_each_dispatch(thread_pool, me, pages, [&](Shared_Board_Index_Iterator& it) {
				thread_pool->run_sync_task_on_all_threads([&](size_t) { body(it); });
			});
		}
		else
		{
			Vector_Not_Bool<Result> all;
			for_each_dispatch(thread_pool, me, pages, [&](Shared_Board_Index_Iterator& it) {
				auto part = thread_pool->run_sync_task_on_all_threads(
					[&](size_t) { return body(it); });
				for (auto& value : part)
					all.emplace_back(std::move(value));
			});
			return all;
		}
	}

	// Same dispatch, but the body runs on the calling thread only. The pool is
	// still used to page the dispatch's set in.
	template <typename Fn>
	auto run_phase_on_this_thread(
		In_Out_Param<Thread_Pool> thread_pool,
		Color me,
		Phase_Pages pages,
		Fn&& body
	)
	{
		using Result = decltype(body(std::declval<Shared_Board_Index_Iterator&>()));

		if constexpr (std::is_same_v<Result, void>)
		{
			for_each_dispatch(thread_pool, me, pages, [&](Shared_Board_Index_Iterator& it) { body(it); });
		}
		else
		{
			Vector_Not_Bool<Result> all;
			for_each_dispatch(thread_pool, me, pages, [&](Shared_Board_Index_Iterator& it) {
				all.emplace_back(body(it));
			});
			return all;
		}
	}

private:
	bool m_paged = false;
	Color m_slice_color = WHITE;
	size_t m_sweep_band_budget_bytes = 256ull * 1024 * 1024;
	Page_Layout m_page_layout;
	Slice_Reach m_slice_reach;
	std::unique_ptr<Page_Cache> m_page_cache;

	// Scratch page bitmaps and the pages set in them, one pair per table, plus
	// the pin set built from them, reused across dispatches. Only the
	// dispatching thread touches them.
	std::vector<uint8_t> m_page_needed[COLOR_NB];
	std::vector<size_t> m_page_touched[COLOR_NB];
	std::vector<Page_Cache::Key> m_page_pin_set;

	// Marks the pages of `table_idx` a dispatch of slices [slice_begin,
	// slice_end) needs at a given slice-group closure depth, appending the
	// newly marked ones to m_page_touched.
	void mark_needed_pages(size_t table_idx, size_t slice_begin, size_t slice_end, int slice_plies);

	template <typename Each>
	void for_each_dispatch(
		In_Out_Param<Thread_Pool> thread_pool,
		Color me,
		Phase_Pages pages,
		Each&& each
	)
	{
		if (!m_paged || (pages.me == PHASE_PLIES_NONE && pages.opp == PHASE_PLIES_NONE))
		{
			// Either everything is resident, or the pass never dereferences a
			// distance entry, so no lease is needed.
			Shared_Board_Index_Iterator it = make_gen_iterator();
			each(it);
			return;
		}

		const size_t me_table = static_cast<size_t>(me);
		const size_t opp_table = static_cast<size_t>(color_opp(me));

		for (size_t page = 0; page < m_page_layout.num_pages(); ++page)
		{
			const size_t slice_begin = m_page_layout.first_slice_of_page(page);
			const size_t slice_end = m_page_layout.end_slice_of_page(page);

			for (size_t table = 0; table < COLOR_NB; ++table)
			{
				for (const size_t p : m_page_touched[table])
					m_page_needed[table][p] = 0;
				m_page_touched[table].clear();
			}

			mark_needed_pages(me_table, slice_begin, slice_end,
				dispatch_slice_plies(pages.me, me, m_slice_color, pages.plies_start_with_mover));
			mark_needed_pages(opp_table, slice_begin, slice_end,
				dispatch_slice_plies(pages.opp, me, m_slice_color, pages.plies_start_with_mover));

			m_page_pin_set.clear();
			for (size_t table = 0; table < COLOR_NB; ++table)
				for (const size_t p : m_page_touched[table])
					m_page_pin_set.emplace_back(table, p);

			Pinned_Pages pins(*m_page_cache);
			pins.pin_all(
				thread_pool, Const_Span<Page_Cache::Key>(m_page_pin_set), opp_table);

			Shared_Board_Index_Iterator it = make_page_iterator(page);
			each(it);
		}
	}
};
