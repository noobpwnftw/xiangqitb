#include "egtb_gen.h"

#include <limits>

Position_For_Gen::Position_For_Gen(const Piece_Config_For_Gen& info, Board_Index pos, Color turn) :
	m_epsi(&info),
	m_turn(turn),
	m_cached_board_index(BOARD_INDEX_NONE)
{
	set_board_index(pos);
}

Position_For_Gen::Position_For_Gen(const Position_For_Gen& parent, Move move, Board_Index next_ix, bool mirr) :
	m_epsi(parent.m_epsi),
	m_turn(color_opp(parent.m_turn)),
	m_cached_board_index(BOARD_INDEX_NONE)
{
	set_board_index(next_ix);

	if (!mirr && parent.m_cached_board_index == parent.m_board_index)
	{
		ASSERT(parent.m_legal);
		ASSERT(parent.m_board.turn() == parent.m_turn);
		m_cached_board_index = next_ix;
		m_board = parent.m_board;
		m_legal = true;
		m_board.do_quiet_move(move);
	}
}

EGTB_Generator::EGTB_Generator(const Piece_Config& ps) :
	m_epsi(ps),
	m_slice_layout(m_epsi),
	m_slice_color(piece_class_color(m_epsi.compress_id()))
{
	const auto [mat_key, mir_key] = m_epsi.material_keys();

	m_is_symmetric = mat_key == mir_key;

	memset(m_sub_needs_mirror_by_capture, 0, sizeof(m_sub_needs_mirror_by_capture));
	memset(m_sub_epsi_by_capture, 0, sizeof(m_sub_epsi_by_capture));

	for (const auto& [piece, sub_ps] : m_epsi.sub_configs_by_capture())
	{
		const bool mirr = m_epsi.needs_mirror_after_capture(piece);
		m_sub_read_color_by_capture[piece] = color_maybe_opp(piece_color(piece), mirr);
		m_sub_needs_mirror_by_capture[piece] = mirr;

		const Material_Key mat_key = sub_ps.base_material_key();
		auto [it, _] = m_sub_epsi_by_material.try_emplace(mat_key, sub_ps);
		m_sub_epsi_by_capture[piece] = &(it->second);
	}
}

Board_Index EGTB_Generator::next_cap_index(const Position_For_Gen& pos_for_gen, Move move) const
{
	const auto& pos = pos_for_gen.board();
	const auto& index = pos_for_gen.index();

	const Square from = move.from();
	const Square to = move.to();
	const Piece piece = pos.piece_on(from);
	const Piece cap = pos.piece_on(to);

	const bool mirr = m_sub_needs_mirror_by_capture[cap];
	const auto& sub = m_sub_epsi_by_capture[cap];
	const Piece_Class capid = piece_class(cap);
	const Piece_Class pieceid = piece_class(piece);

	auto placement_after_capture = [&](const Piece_Class set)
	{
		const Piece_Class id = maybe_opp_piece_class(set, mirr);
		const Piece_Group::Placement& list = m_epsi.squares(index, id);

		Piece_Group::Placement sub_list;
		if (id == capid)
			sub_list = list.with_removed_square(to);
		else if (id == pieceid)
			sub_list = list.with_moved_square(from, to);
		else
			sub_list = list;

		if (mirr)
			sub_list.mirror_ranks();
		
		return sub_list;
	};

	const Piece_Class compress = sub->compress_id();
	const Piece_Group& compress_set = sub->group(compress);
	const Piece_Group::Full_Placement_Index compress_ix = compress_set.compound_index(placement_after_capture(compress));
	const bool lr_mirror = compress_ix.base() >= compress_set.compress_size();

	return sub->compose_board_index([&](const Piece_Group& info, Piece_Class set) {
		const Piece_Group::Full_Placement_Index ix = set == compress ? compress_ix : info.compound_index(placement_after_capture(set));
		return lr_mirror ? ix.mirr() : ix.base();
	});
}

enum struct Quiet_Index_Type
{
	MIRROR, NORMAL
};

template <Quiet_Index_Type MIRR>
static auto quiet_index(
	const Piece_Config_For_Gen& epsi, 
	const Position_For_Gen& pos_for_gen, 
	Move move, 
	Out_Param<bool> mirr
)
{
	Fixed_Vector<Board_Index, 2> ix_tb;

	const auto& pos = pos_for_gen.board();
	const auto& index = pos_for_gen.index();
	const Board_Index current_pos = pos_for_gen.board_index();

	const Piece piece = pos.piece_on(move.from());
	const Piece_Class id = piece_class(piece);

	*mirr = false;

	const Piece_Group& group = epsi.group(id);
	const Piece_Group::Full_Placement_Index ix = group.compound_index_after_quiet_move(index[id], move);
	const bool lr_mirror = ix.base() >= group.compress_size();

	if (id != epsi.compress_id() || !lr_mirror)
	{
		const Board_Index pre_idx = epsi.change_single_group_index(current_pos, index[id], ix.base(), id);

		if constexpr (MIRR == Quiet_Index_Type::NORMAL)
			return pre_idx;
		else
			ix_tb.emplace_back(pre_idx);
	}

	if (id == epsi.compress_id() && (lr_mirror || ix.is_mirrored_same()))
	{
		*mirr = true;
		const Board_Index mir_idx = epsi.compose_mirr_board_index(index);
		const Board_Index pre_idx = epsi.change_single_group_index(mir_idx, group.mirr_index(index[id]), ix.mirr(), id);

		if constexpr (MIRR == Quiet_Index_Type::NORMAL)
			return pre_idx;
		else
			ix_tb.emplace_back(pre_idx);
	}

	if constexpr (MIRR == Quiet_Index_Type::NORMAL)
	{
		ASSUME(false);
		return BOARD_INDEX_NONE;
	}
	else
		return ix_tb;
}

Fixed_Vector<Board_Index, 2> EGTB_Generator::next_quiet_index_with_mirror(
	const Position_For_Gen& pos_for_gen,
	Move move
) const
{
	bool mirr;
	return quiet_index<Quiet_Index_Type::MIRROR>(m_epsi, pos_for_gen, move, out_param(mirr));
}

Board_Index EGTB_Generator::next_quiet_index(
	const Position_For_Gen& pos_for_gen,
	Move move
) const
{
	bool mirr;
	return quiet_index<Quiet_Index_Type::NORMAL>(m_epsi, pos_for_gen, move, out_param(mirr));
}

Board_Index EGTB_Generator::next_quiet_index(
	const Position_For_Gen& pos_for_gen,
	Move move, 
	Out_Param<bool> mirr
) const
{
	return quiet_index<Quiet_Index_Type::NORMAL>(m_epsi, pos_for_gen, move, mirr);
}

// Chunk of consecutive indices handed to a worker at a time, and the target
// batch size for the paged plan, whose runs are usually much shorter.
static constexpr size_t GEN_CHUNK_SIZE = CACHE_LINE_SIZE * CHAR_BIT * 64;
static_assert(GEN_CHUNK_SIZE % EGTB_Bits::ELEMENT_BITS == 0,
	"The flat sweep relies on its chunks never sharing a bitset element.");
static_assert(GEN_CHUNK_SIZE % WDL_ENTRY_PACK_RATIO == 0,
	"The flat sweep relies on its chunks never sharing a packed WDL byte.");

Shared_Board_Index_Iterator EGTB_Generator::make_gen_iterator() const
{
	return Shared_Board_Index_Iterator(BOARD_INDEX_ZERO, static_cast<Board_Index>(m_epsi.num_positions()), GEN_CHUNK_SIZE);
}

Shared_Board_Index_Iterator EGTB_Generator::make_page_iterator(size_t page) const
{
	ASSERT(m_paged);
	return Shared_Board_Index_Iterator(
		m_slice_layout,
		m_page_layout.first_slice_of_page(page),
		m_page_layout.end_slice_of_page(page),
		GEN_CHUNK_SIZE);
}

void EGTB_Generator::enable_paging(
	const Page_Layout& pages,
	std::vector<Pageable_Table*> tables,
	size_t capacity_pages
)
{
	ASSERT(pages.num_slices() == m_slice_layout.num_slices());

	const bool check_chase = m_epsi.both_sides_have_free_attackers();

	size_t reach_plies = 1;
	for (const Phase_Pages& pass : generation_pass_shapes(check_chase))
		update_max(reach_plies, required_reach_plies(pass, m_slice_color));

	m_page_layout = pages;
	m_slice_reach = Slice_Reach(m_epsi.group(m_epsi.compress_id()), reach_plies);
	m_paged = true;

	for (size_t table = 0; table < COLOR_NB; ++table)
	{
		m_page_needed[table].assign(m_page_layout.num_pages(), 0);
		m_page_touched[table].clear();
	}

	// A dispatch's pin set plus the look-ahead's transient lease must fit, or
	// the sweep could not bring residency back under the cap. Same figure the
	// memory plan charged for, recomputed here so the two cannot drift.
	const size_t floor = generation_floor_page_count(
		m_page_layout, m_slice_reach, m_slice_color, check_chase);

	m_page_cache = std::make_unique<Page_Cache>(std::move(tables), std::max(capacity_pages, floor));
}

void EGTB_Generator::disable_paging()
{
	m_paged = false;
	m_page_cache.reset();
	for (size_t table = 0; table < COLOR_NB; ++table)
	{
		m_page_needed[table].clear();
		m_page_touched[table].clear();
	}
}

void EGTB_Generator::mark_needed_pages(
	size_t table_idx,
	size_t slice_begin,
	size_t slice_end,
	int slice_plies
)
{
	if (slice_plies < 0)
		return;

	std::vector<uint8_t>& needed = m_page_needed[table_idx];
	std::vector<size_t>& touched = m_page_touched[table_idx];

	const auto mark = [&](size_t target_slice) {
		const size_t page = m_page_layout.page_of_slice(target_slice);
		if (needed[page])
			return;
		needed[page] = 1;
		touched.emplace_back(page);
	};

	for (size_t slice = slice_begin; slice < slice_end; ++slice)
	{
		if (slice_plies == 0)
		{
			mark(slice);
			continue;
		}

		for (const int32_t target : m_slice_reach.closed(static_cast<size_t>(slice_plies), slice))
			mark(static_cast<size_t>(target));
	}
}

Pinned_Pages EGTB_Generator::lease_one_ply_reach(Color table, Board_Index pos) const
{
	if (!m_paged)
		return Pinned_Pages();

	Pinned_Pages pins(*m_page_cache);

	// The reach is sorted and page ids monotonic in the slice, so duplicates are
	// adjacent. Pins are refcounted, so overlapping the dispatch's own set is
	// harmless.
	size_t last = std::numeric_limits<size_t>::max();
	for (const int32_t target : m_slice_reach.closed(1, m_slice_layout.slice_of(pos)))
	{
		const size_t page = m_page_layout.page_of_slice(static_cast<size_t>(target));
		if (page == last)
			continue;
		last = page;
		pins.pin(static_cast<size_t>(table), page);
	}

	return pins;
}

