#include "egtb_gen_dtm.h"

#include "egtb_compress.h"

#include "util/lazy.h"
#include "util/compress.h"
#include "util/dispatch.h"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>

constexpr size_t MAX_NEXT_TB_ENTRIES = 64;

DTM_Generator::DTM_Generator(
	const Piece_Config& ps, 
	bool srb, 
	const EGTB_Paths& egtb_files
) :
	EGTB_Generator(ps),
	m_egtb_files(egtb_files),
	m_save_rule_bits(srb)
{
	memset(m_sub_dtm_by_capture, 0, sizeof(m_sub_dtm_by_capture));
}

void DTM_Generator::open_sub_egtb()
{
	m_wdl_file = WDL_File_For_Probe(m_egtb_files, m_epsi, m_is_symmetric);

	for (const Piece i : ALL_PIECES)
	{
		const Piece_Config* sub_ps = m_sub_epsi_by_capture[i];
		if (sub_ps == nullptr || !sub_ps->has_any_free_attackers())
			continue;

		const Material_Key mat_key = sub_ps->base_material_key();
		auto [it, inserted] = m_sub_dtm_by_material.try_emplace(mat_key, m_egtb_files, *sub_ps);
		m_sub_dtm_by_capture[i] = &(it->second);
	}
}

void DTM_Generator::close_sub_egtb()
{
	m_sub_dtm_by_material.clear();
	m_wdl_file.close();
	for (auto& v : m_sub_dtm_by_capture)
		v = nullptr;
	m_tmp_files.clear();
}

void DTM_Generator::save_egtb(In_Out_Param<Thread_Pool> thread_pool, const EGTB_Info& info)
{
	const std::string info_path = m_egtb_files.dtm_info_save_path(m_epsi).string();
	const std::string egtb_path = m_egtb_files.dtm_save_path(m_epsi).string();

	// 压缩写入egtb
	Compressed_EGTB save_info[COLOR_NB];
	for (const Color me : { WHITE, BLACK })
	{
		save_info[me] = save_compress_egtb(
			thread_pool, 
			m_dtm_file[me].data_span(),
			me, 
			info, 
			m_save_rule_bits
		);

		if (m_is_symmetric)
			break;
	}
	{
		const auto colors = table_colors();
		save_egtb_table(m_epsi, save_info, egtb_path, colors, EGTB_Magic::DTM_MAGIC);

		const size_t file_size = std::filesystem::file_size(egtb_path);
		const size_t uncompressed_size = colors.size() * m_epsi.num_positions() * sizeof(DTM_Final_Entry);
		const double compression_ratio = static_cast<double>(uncompressed_size) / file_size;
		printf("Saved compressed DTM file. Compression ratio: x%.2f\n", compression_ratio);
	}

	std::ofstream fp(info_path, std::ios_base::binary);
	fp.write(reinterpret_cast<const char*>(&info), sizeof(EGTB_Info));
	fp.close();
}

DTM_Score DTM_Generator::search_cap_win_score(const Position_For_Gen& pos_gen) const
{
	auto& pos = pos_gen.board();

	const bool in_check = pos.is_in_check();

	DTM_Score sub_step = DTM_SCORE_MAX;

	for (const Move move : pos.gen_pseudo_legal_captures())
	{
		if (!pos.is_pseudo_legal_move_legal(move, in_check))
			continue;
			
		const DTM_Final_Entry entry = read_sub_tb_dtm(pos_gen, move);
		if (entry.is_lose())
			update_min(sub_step, ceil_to_even(entry.score()));
	}

	return sub_step;
}

std::pair<DTM_Any_Entry, WDL_Entry> DTM_Generator::make_initial_entry(const Position_For_Gen& pos_gen) const
{
	auto& pos = pos_gen.board();

	if (!pos.is_legal())
		return { DTM_Final_Entry::make_illegal(), WDL_Entry::ILLEGAL };

	const WDL_Entry value = m_wdl_file.read(pos.turn(), pos_gen.board_index());
	if (value == WDL_Entry::DRAW)
		return { DTM_Final_Entry::make_draw(), value };

	ASSERT(value == WDL_Entry::WIN || value == WDL_Entry::LOSE);

	const bool in_check = pos.is_in_check();

	if (value == WDL_Entry::LOSE && pos.is_mate(in_check))
		return { DTM_Final_Entry::make_loss(DTM_SCORE_TERMINAL_LOSS), value };

	const Move_List list = pos.gen_pseudo_legal_captures();

	DTM_Score sub_step = value == WDL_Entry::LOSE ? DTM_SCORE_ZERO : DTM_SCORE_MAX;

	for (const Move move : list)
	{
		if (!pos.is_pseudo_legal_move_legal(move, in_check))
			continue;

		const DTM_Final_Entry entry = read_sub_tb_dtm(pos_gen, move);
		if (value == WDL_Entry::LOSE && entry.is_win())
			update_max(sub_step, ceil_to_odd(entry.score()));
		else if (value == WDL_Entry::WIN && entry.is_lose())
			update_min(sub_step, ceil_to_even(entry.score()));
	}

	if (value == WDL_Entry::LOSE && sub_step != 0)
	{
		// 当前局面输棋
		ASSERT((sub_step & 1) != 0);
		if (pos.is_quiet_mate(in_check))
			return { DTM_Final_Entry::make_loss(sub_step), value };
		else
			return { DTM_Intermediate_Entry::make_cap_score(sub_step), value };
	}
	else if (value == WDL_Entry::WIN && sub_step != DTM_SCORE_MAX)
	{
		// 吃子赢
		ASSERT(sub_step >= 2 && (sub_step & 1) == 0);
		return { DTM_Intermediate_Entry::make_cap_score(sub_step), value };
	}

	return { DTM_Intermediate_Entry::make_empty(), value };
}

void DTM_Generator::sp_init_entries(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	In_Out_Param<Concurrent_Progress_Bar> progress_bar
)
{
	constexpr size_t PROGRESS_BAR_UPDATE_PERIOD = 64 * 64;

	DTM_Score max_step[COLOR_NB] = { static_cast<DTM_Score>(2), static_cast<DTM_Score>(2) };

	auto update_max_step = [&](Color me, WDL_Entry sc, DTM_Score score) {
		if (sc != WDL_Entry::WIN && sc != WDL_Entry::LOSE)
			return;

		const Color c = sc == WDL_Entry::WIN ? me : color_opp(me);
		update_max(max_step[c], score);
	};

	size_t i = 0;

	for (Position_For_Gen& pos_gen : gen_iterator->boards(m_epsi))
	{
		const Board_Index current_pos = pos_gen.board_index();
		ASSERT(current_pos == m_epsi.compose_board_index(pos_gen.index()));

		if (!pos_gen.is_legal())
		{
			write_dtm(current_pos, WHITE, DTM_Final_Entry::make_illegal());
			write_dtm(current_pos, BLACK, DTM_Final_Entry::make_illegal());
			continue;
		}

		for (const Color me : { WHITE, BLACK })
		{
			pos_gen.set_turn(me);

			const auto [entry, sc] = make_initial_entry(pos_gen);
			std::visit(overload(
				[&, sc=sc](DTM_Final_Entry entry) {
					write_dtm(current_pos, me, entry);
					update_max_step(me, sc, entry.score());
				},
				[&, sc=sc](DTM_Intermediate_Entry entry) {
					write_dtm(current_pos, me, entry);
					m_unknown_bits[me].set_bit(current_pos);
					update_max_step(me, sc, entry.cap_score());
				}
			), entry);
		}

		if (++i % PROGRESS_BAR_UPDATE_PERIOD == 0)
			*progress_bar += PROGRESS_BAR_UPDATE_PERIOD;
	}

	for (const Color me : { WHITE, BLACK })
		atomic_update_max(m_max_build_step[me], max_step[me]);
}

void DTM_Generator::init_entries(In_Out_Param<Thread_Pool> thread_pool)
{
	const size_t PRINT_PERIOD = thread_pool->num_workers() * (1 << 20);

	m_max_build_step[WHITE] = static_cast<DTM_Score>(1);
	m_max_build_step[BLACK] = static_cast<DTM_Score>(1);

	auto gen_iterator = make_gen_iterator();
	Concurrent_Progress_Bar progress_bar(gen_iterator.num_indices(), PRINT_PERIOD, "init_entries");
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			sp_init_entries(inout_param(gen_iterator), inout_param(progress_bar));
		}
	);
	progress_bar.set_finished();
}

void DTM_Generator::build_steps(In_Out_Param<Thread_Pool> thread_pool, Color root_color, In_Out_Param<EGTB_Bits_Pool> tmp_bits)
{
	ASSUME(m_max_build_step[WHITE] >= 1 && m_max_build_step[BLACK] >= 1);

	const auto start_time = std::chrono::steady_clock::now();

	EGTB_Bits pre_bits = tmp_bits->acquire_dirty();
	EGTB_Bits win_bits = tmp_bits->acquire_cleared(thread_pool);
	EGTB_Bits gen_bits = tmp_bits->acquire_dirty();

	load_direct(thread_pool, root_color, out_param(gen_bits));

	Color me = root_color;
	Color opp = color_opp(root_color);
	DTM_Score new_step = DTM_SCORE_ZERO;
	for (DTM_Score n = static_cast<DTM_Score>(1);; ++n, std::swap(me, opp))
	{
		printf("build step %zu\r", static_cast<size_t>(n));
		fflush(stdout);

		// The way dtm generation works we are accumulating gen_bits from all iterations.
		// gen_pre_bits_normal does additional checks based on step.
		const bool more_work =
			   gen_pre_bits_normal(thread_pool, opp, n, gen_bits, out_param(pre_bits), inout_param(win_bits))
			&& (  me == root_color
				? save_win(thread_pool, me, n + 1, inout_param(gen_bits), pre_bits, inout_param(win_bits))
				: prove_lose(thread_pool, me, n + 1, inout_param(gen_bits), pre_bits, win_bits));

		if (more_work)
			update_max(new_step, n + 1);

		if (n >= m_max_build_step[root_color] && !more_work)
			break;
	}

	tmp_bits->release(std::move(pre_bits));
	tmp_bits->release(std::move(win_bits));
	tmp_bits->release(std::move(gen_bits));

	const auto end_time = std::chrono::steady_clock::now();

	printf("%s direct max step %zu. Done in %s\n",
		root_color == WHITE ? "white" : "black",
		static_cast<size_t>(new_step),
		format_elapsed_time(start_time, end_time).c_str()
	);
}

template <DTM_Generator::Gen_Pre_Bits_Type TypeV>
bool DTM_Generator::sp_gen_pre_bits(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const DTM_Score n,
	const EGTB_Bits& gen_bits,
	In_Out_Param<EGTB_Bits> pre_bits,
	Optional_In_Out_Param<EGTB_Bits> win_bits
)
{
	ASSUME(TypeV == Gen_Pre_Bits_Type::NORMAL || TypeV == Gen_Pre_Bits_Type::RULE);

	const Color opp = color_opp(me);
	bool ret = false;

	for (const Board_Index current_pos : gen_iterator->indices(gen_bits))
	{
		if constexpr (TypeV == Gen_Pre_Bits_Type::NORMAL)
		{
			// When building steps the gen_bits point to entries of all kind of scores
			// instead of just sequentially up (like during checking rules).
			// Because of that some filtering needs to be done, and
			// it has to happen here because we can't clear bits in gen_bits,
			// because there are entries that will be checked later but not readded to gen_bits.
			if (is_known(current_pos, me))
			{
				auto entry = read_dtm<DTM_Final_Entry>(current_pos, me);
				if (!entry.is_legal() || entry.score() != n)
					continue;
			}
			else
			{
				auto entry = read_dtm<DTM_Intermediate_Entry>(current_pos, me);
				if (!entry.is_cap_win() || entry.cap_score() != n)
					continue;

				// 吃子赢
				auto new_entry = DTM_Final_Entry::copy_rule(entry);
				new_entry.set_score_win(entry.cap_score());
				write_dtm(current_pos, me, new_entry);

				ASSERT(win_bits);
				win_bits->set_bit(current_pos);

				ASSERT(is_unknown(current_pos, me));
				m_unknown_bits[me].clear_bit(current_pos);
			}
		}

		Position_For_Gen pos_gen(m_epsi, current_pos, me);

		auto& board = pos_gen.board();
		ASSERT(board.is_legal());

		for (const Move move : board.gen_pseudo_legal_pre_quiets())
		{
			for (const Board_Index next_ix : pre_quiet_index(pos_gen, move))
			{
				if (is_unknown(next_ix, opp))
				{
					ret = true;
					pre_bits->lock_set_bit(next_ix);
				}
			}
		}
	}

	return ret;
}

bool DTM_Generator::gen_pre_bits_normal(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTM_Score n,
	const EGTB_Bits& gen_bits,
	Out_Param<EGTB_Bits> pre_bits,
	In_Out_Param<EGTB_Bits> win_bits
)
{
	pre_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_gen_pre_bits<Gen_Pre_Bits_Type::NORMAL>(inout_param(gen_iterator), me, n, gen_bits, inout_param(*pre_bits), win_bits);
		}
	);
		
	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

bool DTM_Generator::gen_pre_bits_rule(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTM_Score n,
	const EGTB_Bits& gen_bits,
	Out_Param<EGTB_Bits> pre_bits
)
{
	pre_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_gen_pre_bits<Gen_Pre_Bits_Type::RULE>(inout_param(gen_iterator), me, n, gen_bits, inout_param(*pre_bits), {});
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

bool DTM_Generator::sp_save_win(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const DTM_Score n,
	In_Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& pre_bits,
	In_Out_Param<EGTB_Bits> win_bits
)
{
	bool add_new = false;

	for (const Board_Index current_pos : gen_iterator->indices(pre_bits))
	{
		const auto entry = read_dtm<DTM_Intermediate_Entry>(current_pos, me);
		DTM_Final_Entry new_entry = DTM_Final_Entry::copy_rule(entry);
		new_entry.set_score_win(n);
		write_dtm(current_pos, me, new_entry);
		m_unknown_bits[me].clear_bit(current_pos);
		gen_bits->set_bit(current_pos);
		win_bits->set_bit(current_pos);

		add_new = true;
	}

	return add_new;
}

bool DTM_Generator::save_win(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTM_Score n,
	In_Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& pre_bits,
	In_Out_Param<EGTB_Bits> win_bits
)
{
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_save_win(inout_param(gen_iterator), me, n, gen_bits, pre_bits, win_bits);
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

bool DTM_Generator::sp_prove_lose(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const DTM_Score n,
	In_Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& pre_bits,
	const EGTB_Bits& win_bits
)
{
	const Color opp = color_opp(me);

	bool add_new = false;
	for (const Board_Index current_pos : gen_iterator->indices(pre_bits))
	{
		if (m_wdl_file.read(me, current_pos) != WDL_Entry::LOSE)
			continue;

		const auto entry = read_dtm<DTM_Intermediate_Entry>(current_pos, me);
		if (is_unknown(current_pos, me) && entry.is_cap_win())
			continue;

		Position_For_Gen pos_gen(m_epsi, current_pos, me);
		auto& board = pos_gen.board();
		ASSERT(board.is_legal());

		const bool in_check = board.is_in_check();

		bool lose = true;
		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			if (!board.is_pseudo_legal_move_legal(move, in_check))
				continue;

			const Board_Index next_ix = next_quiet_index(pos_gen, move);

			if (entry.is_ban_lose())
			{
				const auto entry2 = read_dtm<DTM_Entry_Base>(next_ix, opp);
				if (   entry2.is_ban_win()
					&& (   (entry.has_flag(DTM_FLAG_CHECK_LOSE) && entry2.has_flag(DTM_FLAG_CHECK_WIN))
						|| (entry.has_flag(DTM_FLAG_CHASE_LOSE) && entry2.has_flag(DTM_FLAG_CHASE_WIN))))
					continue;
			}

			if (!win_bits.bit_is_set(next_ix))
			{
				lose = false;
				break;
			}
		}

		if (lose)
		{
			const DTM_Score steps =
				entry.is_cap_lose() && entry.cap_score() > n 
				? entry.cap_score() 
				: n;
			DTM_Final_Entry new_entry = DTM_Final_Entry::copy_rule(entry);
			new_entry.set_score_lose(steps);
			write_dtm(current_pos, me, new_entry);
			m_unknown_bits[me].clear_bit(current_pos);
			gen_bits->set_bit(current_pos);

			add_new = true;
		}
	}

	return add_new;
}

bool DTM_Generator::prove_lose(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTM_Score n,
	In_Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& pre_bits,
	const EGTB_Bits& win_bits
)
{
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_prove_lose(inout_param(gen_iterator), me, n, gen_bits, pre_bits, win_bits);
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

template <WDL_Entry TypeV>
bool DTM_Generator::sp_remove_fake(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	In_Out_Param<EGTB_Bits> rule_bits
)
{
	static_assert(TypeV == WDL_Entry::WIN || TypeV == WDL_Entry::LOSE);

	bool add_new = 0;

	const auto flag_chase_good = TypeV == WDL_Entry::WIN ? DTM_FLAG_CHASE_WIN : DTM_FLAG_CHASE_LOSE;
	const auto flag_chase_bad = TypeV == WDL_Entry::WIN ? DTM_FLAG_CHASE_LOSE : DTM_FLAG_CHASE_WIN;
	const auto bad_type = TypeV == WDL_Entry::WIN ? WDL_Entry::LOSE : WDL_Entry::WIN;

	for (const Board_Index current_pos : gen_iterator->indices(*rule_bits))
	{
		const auto entry = read_dtm<DTM_Intermediate_Entry>(current_pos, me);
		if (!entry.is_ban(TypeV))
			continue;
			
		auto new_entry = entry;
		const WDL_Entry sc = m_wdl_file.read(me, current_pos);
		if (   sc == bad_type 
			&& new_entry.has_flag(flag_chase_good)
			&& !new_entry.has_flag(flag_chase_bad))
		{
			new_entry.clear_flag(flag_chase_good); // potentially makes entry.is_ban() false
		}

		if (new_entry.is_ban(TypeV))
		{
			Position_For_Gen pos_gen(m_epsi, current_pos, me);
			new_entry =
				TypeV == WDL_Entry::WIN
				? check_remove_win(pos_gen, new_entry)
				: check_remove_lose(pos_gen, new_entry);
		}

		if (entry != new_entry)
		{
			write_dtm(current_pos, me, new_entry);
			if (!new_entry.is_ban_win() && !new_entry.is_ban_lose())
				rule_bits->clear_bit(current_pos);

			add_new = true;
		}
	}

	return add_new;
}

bool DTM_Generator::remove_fake(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	WDL_Entry type, 
	In_Out_Param<EGTB_Bits> rule_bits
)
{
	ASSUME(type == WDL_Entry::WIN || type == WDL_Entry::LOSE);

	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return TEMPLATE_DISPATCH(
				(Template_Dispatch<WDL_Entry, WDL_Entry::WIN, WDL_Entry::LOSE>(type)),
				sp_remove_fake, inout_param(gen_iterator), me, rule_bits
			);
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

bool DTM_Generator::change_lose_pos(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTM_Score n,
	Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& pre_bits
)
{
	gen_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_load_bits<Load_Bits_Type::CHANGE_LOSE_POS>(inout_param(gen_iterator), me, n, inout_param(*gen_bits), &pre_bits);
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

bool DTM_Generator::load_lose_change(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTM_Score n,
	Out_Param<EGTB_Bits> gen_bits
)
{
	gen_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_load_bits<Load_Bits_Type::LOAD_LOSE_CHANGE>(inout_param(gen_iterator), me, n, inout_param(*gen_bits), {});
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

template <DTM_Generator::Load_Bits_Type TypeV>
bool DTM_Generator::sp_load_bits(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const DTM_Score n,
	In_Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits* pre_bits
)
{
	const Color opp = color_opp(me);

	const EGTB_Bits& bits = 
		TypeV == Load_Bits_Type::LOAD_LOSE_CHANGE 
		? m_unknown_bits[me] 
		: *pre_bits;
			
	DTM_Score max_crv = DTM_SCORE_ZERO;
	bool find_new = false;

	// 加载输棋
	for (const Board_Index current_pos : gen_iterator->indices(bits))
	{
		auto entry = read_dtm<DTM_Final_Entry>(current_pos, me);
		if (!entry.is_lose())
			continue;

		update_max(max_crv, entry.score());

		if (entry.score() != n)
		{
			if constexpr (TypeV == Load_Bits_Type::LOAD_LOSE_CHANGE)
				if (entry.score() < n)
					m_unknown_bits[me].clear_bit(current_pos);

			continue;
		}

		Position_For_Gen pos_gen(m_epsi, current_pos, me);
		auto& board = pos_gen.board();

		ASSERT(board.is_legal());

		Fixed_Vector<std::pair<Board_Index, Move>, MAX_NEXT_TB_ENTRIES> next_tb;
		DTM_Score max_step = DTM_SCORE_ZERO;

		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);
			const auto entry2 = read_dtm<DTM_Final_Entry>(next_ix, opp);

			if (!entry2.is_win() || entry2.score() < max_step)
				continue;
				
			if (entry2.score() > max_step)
			{
				max_step = entry2.score();
				// We only want entries with max_step, so clear earlier
				next_tb.clear();
			}

			next_tb.emplace_back(next_ix, move);
		}

		if (next_tb.empty() || max_step + 1 <= entry.score())
			continue;

		bool find_no_rule = false;
		bool find_check_lose = false;
		int chase_lose_count = 0;
		Bitboard cap_bb = board.occupied();
		for (const auto& [next_ix, move] : next_tb)
		{
			const auto entry2 = read_dtm<DTM_Final_Entry>(next_ix, opp);

			if (   entry.has_flag(DTM_FLAG_CHECK_LOSE)
				&& entry2.has_flag(DTM_FLAG_CHECK_WIN))
			{
				find_check_lose = true;
			}
			else
			{
				Bitboard bb;
				if (   entry.has_flag(DTM_FLAG_CHASE_LOSE)
					&& entry2.has_flag(DTM_FLAG_CHASE_WIN)
					&& board.has_attack_after_quiet_move(move, out_param(bb)))
				{
					cap_bb &= bb;
					chase_lose_count++;
				}
				else {
					find_no_rule = true;
					break;
				}
			}
		}

		if (!find_no_rule)
		{
			if (find_check_lose && chase_lose_count != 0)
			{
				find_no_rule = true;
			}
			else if (chase_lose_count >= 2 && cap_bb.empty())
			{
				find_no_rule = true;
			}
		}

		const DTM_Score best_step = find_no_rule ? max_step + 1 : max_step;

		if (best_step > entry.score())
		{
			update_max(max_crv, best_step);
			entry.set_score(best_step);
			write_dtm(current_pos, me, entry);
			gen_bits->set_bit(current_pos);
			find_new = true;
		}
	}

	atomic_update_max(m_max_step, max_crv);

	return find_new;
}

bool DTM_Generator::check_double_chase_win(
	Position_For_Gen& pos_gen,
	Move evt_move, 
	Board_Index next_idx,
	const Color me, 
	const bool mirr, 
	const DTM_Score max_step
) const
{
	const Color opp = color_opp(me);

	Position_For_Gen next_pos_gen(pos_gen, evt_move, next_idx, mirr);
	auto& next_board = next_pos_gen.board();

	DTM_Score min_step = DTM_SCORE_MAX;

	Move_List move_tb;
	bool find_no_cap = false;

	for (const Move move : next_board.gen_pseudo_legal_quiets())
	{
		const Board_Index next_ix = next_quiet_index(next_pos_gen, move);
		const auto entry = read_dtm<DTM_Final_Entry>(next_ix, opp);

		if (   !entry.is_lose() 
			|| entry.score() > min_step)
			continue;
		
		if (entry.score() < min_step)
		{
			min_step = entry.score();
			move_tb.clear();
			find_no_cap = false;
		}

		if (entry.has_flag(DTM_FLAG_CHASE_WIN) && entry.has_flag(DTM_FLAG_CHASE_LOSE))
			move_tb.add(move);
		else
			find_no_cap = true;
	}

	if (find_no_cap || min_step > max_step || move_tb.empty())
		return false;

	Bitboard evtbb;
	if (!pos_gen.board().is_move_evasion(evt_move, out_param(evtbb)))
		print_and_abort("Expected evt move");

	return next_board.always_has_attack_after_quiet_moves(move_tb, evtbb.maybe_mirror_files(mirr));
}

void DTM_Generator::loop_init_check_chase(In_Out_Param<Thread_Pool> thread_pool, In_Out_Param<EGTB_Bits_Pool> tmp_bits)
{
	if (!m_epsi.both_sides_have_free_attackers())
		return;

	EGTB_Bits rule_bits[COLOR_NB] = { tmp_bits->acquire_cleared(thread_pool), tmp_bits->acquire_cleared(thread_pool) };

	init_check_chase(thread_pool, inout_param(rule_bits));

	for (const Color me : { WHITE, BLACK })
	{
		const Color opp = color_opp(me);
		size_t i = 0;
		for (;;)
		{
			printf("remove_fake %d %zu\r", me, ++i);
			fflush(stdout);
			if (!remove_fake(thread_pool, opp, WDL_Entry::LOSE, inout_param(rule_bits[opp])))
				break;

			printf("remove_fake %d %zu\r", me, ++i);
			fflush(stdout);
			if (!remove_fake(thread_pool, me, WDL_Entry::WIN, inout_param(rule_bits[me])))
				break;
		}
		printf("remove_fake %d finished in %zu steps\n", me, i);
	}

	tmp_bits->release(std::move(rule_bits[WHITE]));
	tmp_bits->release(std::move(rule_bits[BLACK]));

}

void DTM_Generator::loop_build_check_chase(In_Out_Param<Thread_Pool> thread_pool, In_Out_Param<EGTB_Bits_Pool> tmp_bits)
{
	if (!m_epsi.both_sides_have_free_attackers())
		return;

	printf("final build steps...\n");

	EGTB_Bits pre_bits = tmp_bits->acquire_dirty();
	EGTB_Bits gen_bits = tmp_bits->acquire_dirty();
	EGTB_Bits win_bits = tmp_bits->acquire_dirty();

	for (const Color me : { WHITE, BLACK })
	{
		const auto start_time = std::chrono::steady_clock::now();

		const Color opp = color_opp(me);
		m_max_step = static_cast<DTM_Score>(5); // 后面会修改

		m_unknown_bits[WHITE] = tmp_bits->acquire_cleared(thread_pool);
		m_unknown_bits[BLACK] = tmp_bits->acquire_cleared(thread_pool);

		second_init(thread_pool, me);

		DTM_Score n = static_cast<DTM_Score>(3);
		for (; n <= m_max_step && n < DTM_SCORE_MAX; ++n)
		{
			printf("build step %zu\r", static_cast<size_t>(n));
			fflush(stdout);
			if (load_lose_change(thread_pool, opp, n, out_param(gen_bits)))
			{
				for (;;)
				{
					if (!gen_pre_bits_rule(thread_pool, opp, n, gen_bits, out_param(pre_bits)))
						break;
					bool find = change_win_pos_step1(thread_pool, me, n, out_param(gen_bits), out_param(win_bits), pre_bits);
					if (!change_win_pos_step2(thread_pool, me, n, inout_param(gen_bits), win_bits) && !find)
						break;
					if (!gen_pre_bits_rule(thread_pool, me, n, gen_bits, out_param(pre_bits)))
						break;
					if (!change_lose_pos(thread_pool, opp, n, out_param(gen_bits), pre_bits))
						break;
				}
			}
		}

		tmp_bits->release(std::move(m_unknown_bits[WHITE]));
		tmp_bits->release(std::move(m_unknown_bits[BLACK]));

		const auto end_time = std::chrono::steady_clock::now();

		printf(
			"%s max step %zu. Done in %s\n", 
			me == 0 ? "white" : "black", 
			static_cast<size_t>(n),
			format_elapsed_time(start_time, end_time).c_str()
		);

		if (n >= DTM_SCORE_MAX - 10)
			print_and_abort("more steps\n");
	}

	tmp_bits->release(std::move(pre_bits));
	tmp_bits->release(std::move(gen_bits));
	tmp_bits->release(std::move(win_bits));
}

void DTM_Generator::gen_rule_lose(In_Out_Param<Thread_Pool> thread_pool, In_Out_Param<EGTB_Bits_Pool> tmp_bits)
{
	for (const Color me : { WHITE, BLACK })
	{
		EGTB_Bits opp_bits = tmp_bits->acquire_dirty();
		EGTB_Bits me_bits = tmp_bits->acquire_dirty();

		find_rule_lose(thread_pool, me, out_param(me_bits), out_param(opp_bits));
		
		size_t i = 0;
		for (;;)
		{
			if (!remove_rule_lose(thread_pool, me, color_opp(me), inout_param(opp_bits), me_bits))
				break;
			if (!remove_rule_lose(thread_pool, me, me, inout_param(me_bits), opp_bits))
				break;
		}

		save_rule_lose(thread_pool, me, me_bits);

		tmp_bits->release(std::move(opp_bits));
		tmp_bits->release(std::move(me_bits));
	}
}

void DTM_Generator::gen(In_Out_Param<Thread_Pool> thread_pool)
{
	printf("%s gen dtm start...\n", m_epsi.name().c_str());

	for (const Color me : { WHITE, BLACK })
		m_dtm_file[me].create(m_epsi.num_positions());

	open_sub_egtb();

	EGTB_Bits_Pool tmp_bits(5, m_epsi.num_positions());

	m_unknown_bits[WHITE] = tmp_bits.acquire_cleared(thread_pool);
	m_unknown_bits[BLACK] = tmp_bits.acquire_cleared(thread_pool);

	init_entries(thread_pool);

	loop_init_check_chase(thread_pool, inout_param(tmp_bits));

	//第二步，迭代获取输赢信息

	gen_rule_lose(thread_pool, inout_param(tmp_bits));

	build_steps(thread_pool, WHITE, inout_param(tmp_bits));
	build_steps(thread_pool, BLACK, inout_param(tmp_bits));

	tmp_bits.release(std::move(m_unknown_bits[WHITE]));
	tmp_bits.release(std::move(m_unknown_bits[BLACK]));

	loop_build_check_chase(thread_pool, inout_param(tmp_bits));

	const EGTB_Info info = check_dtm_egtb(thread_pool);
	close_sub_egtb();

	// Release some memory for compression.
	tmp_bits.clear();

	save_egtb(thread_pool, info);

	for (const Color me : { WHITE, BLACK })
		m_dtm_file[me].close();
}

DTM_Intermediate_Entry DTM_Generator::check_remove_lose(Position_For_Gen& pos_gen, DTM_Intermediate_Entry tt) const
{
	if (!tt.has_flag(DTM_FLAG_CHECK_LOSE) && !tt.has_flag(DTM_FLAG_CHASE_LOSE))
		print_and_abort("Expected either chase or check lose flag.");

	Position& board = pos_gen.board();
	const Color me = board.turn();
	const Color opp = color_opp(me);
	const auto in_check = Lazy_Cached_Value([&]() { return board.is_in_check(); });

	auto is_long_chase = [&](Move move, Board_Index next_ix, bool mirr) {
		if (!board.has_attack_after_quiet_move(move))
			return false;

		// If not in check then we don't need further verification.
		if (!in_check)
			return true;

		Position_For_Gen next_pos_gen(pos_gen, move, next_ix, mirr);
		auto& next_board = next_pos_gen.board();

		for (const Move move2 : next_board.gen_pseudo_legal_quiets())
		{
			const Board_Index next_ix2 = next_quiet_index(next_pos_gen, move2);
			const auto entry2 = read_dtm<DTM_Entry_Base>(next_ix2, me);

			if (   entry2.is_legal()
				&& entry2.has_flag(DTM_FLAG_CHASE_LOSE)
				&& !next_board.is_move_check(move2)
				&& next_board.is_move_evasion(move2))
			{
				return true;
			}
		}

		return false;
	};

	// The meaning of these variables is only valid later, this is just initialization.
	bool long_check = !tt.has_flag(DTM_FLAG_CHECK_LOSE);
	bool long_chase = !tt.has_flag(DTM_FLAG_CHASE_LOSE);

	for (const Move move : board.gen_pseudo_legal_quiets())
	{
		bool mirr;
		const Board_Index next_ix = next_quiet_index(pos_gen, move, out_param(mirr));
		const auto entry = read_dtm<DTM_Entry_Base>(next_ix, opp);

		if (!entry.is_legal())
			continue;

		if (!long_check && entry.has_flag(DTM_FLAG_CHECK_WIN))
			long_check = true;

		if (!long_chase && entry.has_flag(DTM_FLAG_CHASE_WIN))
			long_chase = is_long_chase(move, next_ix, mirr);

		if (long_check && long_chase)
			break;
	}

	if (!long_check)
	{
		// The flag removal must not be idempotent, or otherwise the return value is wrong.
		ASSERT(tt.has_flag(DTM_FLAG_CHECK_LOSE));
		tt.clear_flag(DTM_FLAG_CHECK_LOSE);
	}

	if (!long_chase)
	{
		// The flag removal must not be idempotent, or otherwise the return value is wrong.
		ASSERT(tt.has_flag(DTM_FLAG_CHASE_LOSE));
		tt.clear_flag(DTM_FLAG_CHASE_LOSE);
	}

	return tt;
}

DTM_Intermediate_Entry DTM_Generator::check_remove_win(Position_For_Gen& pos_gen, DTM_Intermediate_Entry tt) const
{
	// Contrary to check_remove_lose, here we should never have both check and chase win flags set.
	// This is because of the way the flags are assigned in sp_init_check_chase.
	// The lose flags are assigned for each child position, 
	// so it can happen that both check and chase are assigned to the same entry.
	// The win flags are assigned once, for the position being considered, 
	// so we have a guarantee that only one is set.
	if (tt.has_flag(DTM_FLAG_CHECK_WIN) == tt.has_flag(DTM_FLAG_CHASE_WIN))
		print_and_abort("Expected exactly one of chase or check win flag.");

	Position& board = pos_gen.board();
	const Color me = board.turn();
	const Color opp = color_opp(me);

	if (tt.has_flag(DTM_FLAG_CHECK_WIN))
	{
		ASSERT(tt.has_flag(DTM_FLAG_CHECK_WIN));
		ASSERT(!tt.has_flag(DTM_FLAG_CHASE_WIN));

		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);
			const auto entry = read_dtm<DTM_Entry_Base>(next_ix, opp);

			if (   entry.is_legal()
				&& entry.has_flag(DTM_FLAG_CHECK_LOSE))
				return tt;
		}

		tt.clear_flag(DTM_FLAG_CHECK_WIN);
	}
	else
	{
		ASSERT(!tt.has_flag(DTM_FLAG_CHECK_WIN));
		ASSERT(tt.has_flag(DTM_FLAG_CHASE_WIN));

		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			bool mirr;
			const Board_Index next_ix = next_quiet_index(pos_gen, move, out_param(mirr));
			const auto entry = read_dtm<DTM_Entry_Base>(next_ix, opp);

			Bitboard evt_piecebb;
			if (   !entry.is_legal()
				|| !entry.has_flag(DTM_FLAG_CHASE_LOSE)
				|| !board.is_move_evasion(move, out_param(evt_piecebb)))
				continue;

			Position_For_Gen next_pos_gen(pos_gen, move, next_ix, mirr);
			auto& next_board = next_pos_gen.board();

			for (const Move move2 : next_board.gen_pseudo_legal_quiets())
			{
				const Board_Index next_ix2 = next_quiet_index(next_pos_gen, move2);
				const auto entry2 = read_dtm<DTM_Entry_Base>(next_ix2, me);

				Bitboard capbb;
				if (   entry2.is_legal()
					&& entry2.has_flag(DTM_FLAG_CHASE_WIN)
					&& next_board.has_attack_after_quiet_move(move2, out_param(capbb))
					&& (capbb & evt_piecebb.maybe_mirror_files(mirr)))
					return tt;
			}
		}

		tt.clear_flag(DTM_FLAG_CHASE_WIN);
	}

	return tt;
}

EGTB_Info DTM_Generator::check_dtm_egtb(In_Out_Param<Thread_Pool> thread_pool)
{
	auto gen_iterator = make_gen_iterator();
	auto infos = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_check_dtm_egtb(inout_param(gen_iterator));
		}
	);

	EGTB_Info info;
	for (const Color c : { WHITE, BLACK })
	{
		info.consolidate_from(infos.begin(), infos.end(), c);

		if (info.longest_win[c] != 0)
		{
			Position_For_Gen pos_gen(m_epsi, static_cast<Board_Index>(info.longest_idx[c]), c);
			pos_gen.get_fen(Span(info.longest_fen[c]));
		}
	}

	return info;
}

EGTB_Info DTM_Generator::sp_check_dtm_egtb(In_Out_Param<Shared_Board_Index_Iterator> gen_iterator)
{
	EGTB_Info info;

	for (const Board_Index current_pos : gen_iterator->indices())
	{
		for (const Color c : { WHITE, BLACK })
		{
			auto entry = read_dtm<DTM_Final_Entry>(current_pos, c);
			const WDL_Entry sc = m_wdl_file.read(c, current_pos);

			if (!entry.is_legal())
			{
				write_dtm(current_pos, c, DTM_Final_Entry::make_draw());
				info.illegal_cnt[c] += 1;

				continue;
			}
			else if (!m_save_rule_bits)
			{
				entry.remove_rule_bits();
				write_dtm(current_pos, c, entry);
			}

			auto on_wrong_result = [&](const char* result_str) {
				char fen[MAX_FEN_LENGTH];
				Position_For_Gen pos_gen(m_epsi, current_pos, c);
				pos_gen.get_fen(Span(fen));
				print_and_abort("%d find different! %s  %llu\n%s\n", c, result_str, current_pos, fen);
			};

			if (entry.score() == 0)
			{
				if (sc != WDL_Entry::DRAW)
					on_wrong_result("DRAW");

				info.draw_cnt[c] += 1;
			}
			else if (entry.is_lose())
			{
				if (sc != WDL_Entry::LOSE)
					on_wrong_result("LOSE");

				info.lose_cnt[c] += 1;
			}
			else if (entry.is_win())
			{
				if (sc != WDL_Entry::WIN)
					on_wrong_result("WIN");

				info.win_cnt[c] += 1;

				if (entry.score() > info.longest_win[c])
				{
					info.longest_win[c] = narrowing_static_cast<uint16_t>(entry.score());
					info.longest_idx[c] = current_pos;
				}
			}
			else
				on_wrong_result("NONE");
		}
	}

	return info;
}

DTM_Final_Entry DTM_Generator::read_sub_tb_dtm(
	const Position_For_Gen& pos_gen,
	Move move
) const
{
	const Position& pos = pos_gen.board();
	const Square to = move.to();
	const Piece piece = pos.piece_on(to);

	// The only unpopulated EGTBs are the ones without free attackers, so drawn.
	if (m_sub_dtm_by_capture[piece] == nullptr)
		return DTM_Final_Entry::make_draw();

	const Board_Index next_ix = next_cap_index(pos_gen, move);
	return m_sub_dtm_by_capture[piece]->read(m_sub_read_color_by_capture[piece], next_ix);
}

void DTM_Generator::sp_second_init(In_Out_Param<Shared_Board_Index_Iterator> gen_iterator, Color root_color)
{
	for (const Board_Index current_pos : gen_iterator->indices())
	{
		for (const Color turn : { root_color, color_opp(root_color) })
		{
			const auto entry = read_dtm<DTM_Final_Entry>(current_pos, turn);
			if (turn == root_color
				? entry.is_win() && entry.score() > 2
				: entry.is_lose() && entry.score() > 1)
				m_unknown_bits[turn].set_bit(current_pos);
		}
	}
}

void DTM_Generator::second_init(In_Out_Param<Thread_Pool> thread_pool, Color root_color)
{
	auto gen_iterator = make_gen_iterator();
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_second_init(inout_param(gen_iterator), root_color);
		}
	);
}

void DTM_Generator::sp_init_check_chase(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits,
	In_Out_Param<Concurrent_Progress_Bar> progress_bar
)
{
	constexpr size_t PROGRESS_BAR_UPDATE_PERIOD = 64 * 64;
	size_t i = 0;

	for (Position_For_Gen& pos_gen : gen_iterator->boards(m_epsi))
	{
		const Board_Index current_pos = pos_gen.board_index();
		bool in_check = false;

		for (const Color me : { WHITE, BLACK })
		{
			if (is_known(current_pos, me))
				continue;

			const WDL_Entry sc = m_wdl_file.read(me, current_pos);

			if (sc != WDL_Entry::WIN && sc != WDL_Entry::LOSE)
				continue;

			auto& board = pos_gen.board();
			board.set_turn(me);
			ASSERT(board.is_legal());

			// If the other side is already in check then this one is definitely not.
			in_check = !in_check && board.is_in_check();

			const Move_List list =
				in_check
				? board.gen_pseudo_legal_quiets()
				: board.gen_legal_capture_evasions();

			const Color opp = color_opp(me);

			bool find = false;
			for (const Move move : list)
			{
				if (in_check && !board.is_pseudo_legal_move_legal_in_check(move))
					continue;

				// We want to check the mirrored case by design.
				// Otherwise current_pos might not get marked correctly.
				for (const Board_Index next_ix : next_quiet_index_with_mirror(pos_gen, move))
				{
					if (is_known(next_ix, opp))
						continue;

					const WDL_Entry sc2 = m_wdl_file.read(opp, next_ix);
					if ((sc == WDL_Entry::WIN) ? (sc2 == WDL_Entry::LOSE) : (!in_check && sc2 == WDL_Entry::WIN))
					{
						find = true;
						lock_or_dtm(next_ix, opp, in_check ? DTM_FLAG_CHECK_LOSE : DTM_FLAG_CHASE_LOSE);
						rule_bits[opp].lock_set_bit(next_ix);
					}
				}
			}

			if (find)
			{
				lock_or_dtm(current_pos, me, in_check ? DTM_FLAG_CHECK_WIN : DTM_FLAG_CHASE_WIN);
				rule_bits[me].lock_set_bit(current_pos);
			}
		}

		if (++i % PROGRESS_BAR_UPDATE_PERIOD == 0)
			*progress_bar += PROGRESS_BAR_UPDATE_PERIOD;
	}
}

void DTM_Generator::init_check_chase(
	In_Out_Param<Thread_Pool> thread_pool, 
	In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
)
{
	const size_t PRINT_PERIOD = thread_pool->num_workers() * (1 << 20);

	auto gen_iterator = make_gen_iterator();
	Concurrent_Progress_Bar progress_bar(gen_iterator.num_indices(), PRINT_PERIOD, "init_check_chase");
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_init_check_chase(inout_param(gen_iterator), rule_bits, inout_param(progress_bar));
		}
	);
	progress_bar.set_finished();
}

void DTM_Generator::sp_load_direct(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	Color me, 
	In_Out_Param<EGTB_Bits> gen_bits
)
{
	const Color opp = color_opp(me);
	for (const Board_Index current_pos : gen_iterator->indices())
	{
		if (is_unknown(current_pos, me))
		{
			const auto entry = read_dtm<DTM_Intermediate_Entry>(current_pos, me);
			if (entry.is_cap_win())
			{
				gen_bits->set_bit(current_pos);
				continue;
			}
		}

		if (is_known(current_pos, opp))
		{
			const auto entry = read_dtm<DTM_Final_Entry>(current_pos, opp);
			if (entry.is_lose())
				gen_bits->set_bit(current_pos);
		}
	}
}

void DTM_Generator::load_direct(
	In_Out_Param<Thread_Pool> thread_pool,
	Color me,
	Out_Param<EGTB_Bits> gen_bits
)
{
	gen_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_load_direct(inout_param(gen_iterator), me, inout_param(*gen_bits));
		}
	);
}

void DTM_Generator::sp_find_rule_lose(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	Color me, 
	In_Out_Param<EGTB_Bits> me_bits, 
	In_Out_Param<EGTB_Bits> opp_bits
)
{
	const Color opp = color_opp(me);
	for (const Board_Index current_pos : gen_iterator->indices())
	{
		if (is_known(current_pos, me))
			continue;

		const auto entry = read_dtm<DTM_Intermediate_Entry>(current_pos, me);
		if (!entry.is_ban_lose())
			continue;
		if (entry.has_flag(DTM_FLAG_CHECK_LOSE) && entry.has_flag(DTM_FLAG_CHASE_LOSE))
			continue;
		if (m_wdl_file.read(me, current_pos) != WDL_Entry::LOSE)
			continue;

		Position_For_Gen pos_gen(m_epsi, current_pos, me);
		const auto& board = pos_gen.board();

		Fixed_Vector<Board_Index, MAX_NEXT_TB_ENTRIES> next_tb;
		bool is_rule_lose = true;
		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);
			const auto entry2 = read_dtm<DTM_Entry_Base>(next_ix, opp);

			if (!entry2.is_legal())
				continue;

			// 赢棋
			if (  (   !(entry.has_flag(DTM_FLAG_CHECK_LOSE) && entry2.has_flag(DTM_FLAG_CHECK_WIN))
					&& !(entry.has_flag(DTM_FLAG_CHASE_LOSE) && entry2.has_flag(DTM_FLAG_CHASE_WIN)))
				|| (m_wdl_file.read(opp, next_ix) != WDL_Entry::WIN))
			{
				is_rule_lose = false;
				break;
			}

			next_tb.emplace_back(next_ix);
		}

		if (is_rule_lose)
		{
			me_bits->set_bit(current_pos);
			for (const Board_Index next_ix : next_tb)
				opp_bits->lock_set_bit(next_ix);
		}
	}
}

void DTM_Generator::find_rule_lose(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me,
	Out_Param<EGTB_Bits> me_bits,
	Out_Param<EGTB_Bits> opp_bits
)
{
	me_bits->clear(thread_pool);
	opp_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_find_rule_lose(inout_param(gen_iterator), me, inout_param(*me_bits), inout_param(*opp_bits));
		}
	);
}

void DTM_Generator::sp_save_rule_lose(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	Color me, 
	const EGTB_Bits& me_bits
)
{
	for (const Board_Index current_pos : gen_iterator->indices(me_bits))
	{
		const auto entry = read_dtm<DTM_Intermediate_Entry>(current_pos, me);
		ASSERT(entry.is_ban_lose());

		auto new_entry = DTM_Final_Entry::copy_rule(entry);
		new_entry.set_score_lose(
			entry.has_cap_score()
			? entry.cap_score()
			: DTM_SCORE_TERMINAL_LOSS
		);
		write_dtm(current_pos, me, new_entry);
		m_unknown_bits[me].clear_bit(current_pos);
	}
}

void DTM_Generator::save_rule_lose(In_Out_Param<Thread_Pool> thread_pool, Color me, const EGTB_Bits& me_bits)
{
	auto gen_iterator = make_gen_iterator();
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_save_rule_lose(inout_param(gen_iterator), me, me_bits);
		}
	);
}

bool DTM_Generator::sp_remove_rule_lose(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	Color root_color,
	Color me,
	In_Out_Param<EGTB_Bits> gen_bits, 
	const EGTB_Bits& dst_bits
)
{
	const Color opp = color_opp(me);
	bool find_new = false;

	for (const Board_Index current_pos : gen_iterator->indices(*gen_bits))
	{
		Position_For_Gen pos_gen(m_epsi, current_pos, me);
		const auto& board = pos_gen.board();

		bool find = false;
		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);
			const auto entry2 = read_dtm<DTM_Entry_Base>(next_ix, opp);

			if (entry2.is_legal()
				&& (root_color == me
					? !dst_bits.bit_is_set(next_ix)
					: dst_bits.bit_is_set(next_ix)))
			{
				find = true;
				break;
			}
		}

		if (root_color == me ? find : !find)
		{
			find_new = true;
			gen_bits->clear_bit(current_pos);
		}
	}

	return find_new;
}

bool DTM_Generator::remove_rule_lose(
	In_Out_Param<Thread_Pool> thread_pool,
	Color root_color,
	Color me, 
	In_Out_Param<EGTB_Bits> me_bits,
	const EGTB_Bits& opp_bits
)
{
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_remove_rule_lose(inout_param(gen_iterator), root_color, me, me_bits, opp_bits);
		}
	);
	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

template <DTM_Generator::Change_Win_Pos_Step TypeV>
bool DTM_Generator::sp_change_win_pos(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	Color me, 
	DTM_Score n,
	In_Out_Param<EGTB_Bits> gen_bits,
	Optional_In_Out_Param<EGTB_Bits> win_bits,
	const EGTB_Bits& pre_bits
)
{
	const Color opp = color_opp(me);

	DTM_Score max_crv = DTM_SCORE_ZERO;
	bool find_new = false;

	for (const Board_Index current_pos : gen_iterator->indices(pre_bits))
	{
		auto entry = read_dtm<DTM_Final_Entry>(current_pos, me);

		if (!entry.is_win())
			continue;

		update_max(max_crv, entry.score());

		if (entry.score() != n && entry.score() != n + 1)
			continue;

		Position_For_Gen pos_gen(m_epsi, current_pos, me);
		auto& board = pos_gen.board();

		ASSERT(board.is_legal());

		DTM_Score min_step = DTM_SCORE_MAX;

		// Finding whether it's a check or chase win is split into 4 steps of increasing complexity.
		// Effectively, we check a full batch for each condition before 
		// moving to the next condition, allowing efficient short-circuit.
		Fixed_Vector<std::tuple<Board_Index, Move, bool>, MAX_NEXT_TB_ENTRIES> next_tb;
		bool find_no_rule = false;

		// Step 1.a.
		// Gather all quiet move candidates - moves that lead to entries of the lower score.
		// We don't know the min score apriori.
		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			bool mirr;
			const Board_Index next_ix = next_quiet_index(pos_gen, move, out_param(mirr));
			const auto entry2 = read_dtm<DTM_Final_Entry>(next_ix, opp);

			if (!entry2.is_lose())
				continue;

			ASSERT(entry2.score() != 0);

			find_no_rule = 
				   find_no_rule 
				|| !entry2.has_flag(DTM_FLAG_CHASE_WIN) 
				|| !board.has_attack_after_quiet_move(move);

			// 输棋
			if (entry2.score() > min_step)
				continue;
			
			if (entry2.score() < min_step)
			{
				min_step = entry2.score();
				next_tb.clear();
			}

			next_tb.emplace_back(next_ix, move, mirr);
		}

		// Step 1.b.
		// First short-circuit. No need to check further if
		// score is already lower. We can only check score to higher here.
		// `min_step + 1` is because in some cases 1 may be added to the score before writing.
		if (min_step == DTM_SCORE_MAX || min_step + 1 <= entry.score())
			continue;

		// Step 2.a.
		// Both min_step and cap_step must be larger than entry.score() to alter
		// the resulting score. search_cap_win_score is costly because it reads sub tbs,
		// so we postpone it to until after step 1 is done. Otherwise we could have done
		// it in the first loop by generating all the moves and splitting the logic.
		const DTM_Score cap_step = search_cap_win_score(pos_gen);

		if (cap_step != DTM_SCORE_MAX)
		{
			// Step 2.b.
			// Second short-circuit. We require both min_step and cap_step to be larger,
			// because only the minimum of them can affect the result.
			if (cap_step <= entry.score())
				continue;

			find_no_rule = true;
		}

		bool check_or_chase_win = false;

		Fixed_Vector<size_t, MAX_NEXT_TB_ENTRIES> chase_idx;

		// Step 3.a.
		// Either we find immediately that there's a check or chase win
		// or we make a list of entries to check further.
		for (size_t i = 0; i < next_tb.size(); ++i)
		{
			const auto& [next_ix, move, mirr] = next_tb[i];
			const auto entry2 = read_dtm<DTM_Final_Entry>(next_ix, opp);

			if (   entry.has_flag(DTM_FLAG_CHECK_WIN)
				&& entry2.has_flag(DTM_FLAG_CHECK_LOSE))
			{
				check_or_chase_win = true;
				break;
			}
				
			if (   !entry.has_flag(DTM_FLAG_CHASE_WIN)
				|| !entry2.has_flag(DTM_FLAG_CHASE_LOSE)
				|| !board.is_move_evasion(move))
				continue;
			
			if (   !find_no_rule
				|| !entry.has_flag(DTM_FLAG_CHASE_LOSE)
				|| !entry2.has_flag(DTM_FLAG_CHASE_WIN)
				|| !board.has_attack_after_quiet_move(move))
			{
				check_or_chase_win = true;
				break;
			}

			chase_idx.emplace_back(i);
		}

		// Step 3.b.
		// Third short-circuit, no need to check further if we already know it's a check or chase win.
		// Otherwise it requires checking mutual chases.
		if (!check_or_chase_win && chase_idx.size())
		{
			if constexpr (TypeV == Change_Win_Pos_Step::STEP_1)
				win_bits->set_bit(current_pos);
			else
			{
				// Step 4.a.
				// Go through the moves that lead to chases and verify them.
				// This part has circular references to the data of the same side, parallelizing
				// may cause loops remain unresolved per current iteration, while it should be able
				// to eventually work out, it breaks determinism of the output files.
				for (const size_t chase_ix : chase_idx)
				{
					const auto& [next_ix, move, mirr] = next_tb[chase_ix];

					Position_For_Gen next_pos_gen(pos_gen, move, next_ix, mirr);
					auto& next_board = next_pos_gen.board();

					Fixed_Vector<std::tuple<Board_Index, Move, bool>, MAX_NEXT_TB_ENTRIES> next_tb2;
					DTM_Score max_step = DTM_SCORE_ZERO;

					// Step 4.b.
					// First we need to get the list of moves that lead to entries of maximal score.
					// This is split-off by necessity, as we don't know the max score apriori.
					for (const Move move2 : next_board.gen_pseudo_legal_quiets())
					{
						bool mirr2;
						const Board_Index next_ix2 = next_quiet_index(next_pos_gen, move2, out_param(mirr2));
						const auto entry2 = read_dtm<DTM_Final_Entry>(next_ix2, me);

						if (!entry2.is_win() || entry2.score() < max_step)
							continue;

						if (entry2.score() > max_step)
						{
							max_step = entry2.score();
							next_tb2.clear();
						}

						if (entry2.has_flag(DTM_FLAG_CHASE_LOSE) && entry2.has_flag(DTM_FLAG_CHASE_WIN))
							next_tb2.emplace_back(next_ix2, move2, mirr2);
					}

					// Step 4.c.
					// And if there are any such entries we perform all the expensive checks required.
					bool find_evt = false;
					if (!next_tb2.empty())
					{
						Bitboard capbb;
						// This was checked during Step 3.a., but we need to do it again
						// to retrieve the cap bitboard. This step happens rarely enough that
						// it's not worth to store this bitboard in Step 3.a.
						if (!board.has_attack_after_quiet_move(move, out_param(capbb)))
							print_and_abort("Expected capture move.");

						for (const auto& [next_ix2, move2, mirr2] : next_tb2)
						{
							Bitboard evtbb;
							if (   next_board.is_move_evasion(move2, out_param(evtbb))
								&& (capbb & evtbb.maybe_mirror_files(mirr))
								&& check_double_chase_win(next_pos_gen, move2, next_ix2, me, mirr2, max_step))
							{
								find_evt = true;
								break;
							}
						}
					}

					// Step 4.d.
					// Finally, of no valid evasion is found we know it's a chase win.
					if (!find_evt)
					{
						check_or_chase_win = true;
						break;
					}
				}
			}
		}

		const DTM_Score best_step = (TypeV == Change_Win_Pos_Step::STEP_1) ? min_step : (check_or_chase_win ? min_step : (min_step + 1));
		// We can safely use min because DTM_SCORE_MAX is always greater than all other scores.
		const DTM_Score write_step = std::min(best_step, cap_step);

		// And (because of check_or_chase_win) we only now know whether to even write the new value.
		if (write_step > entry.score())
		{
			entry.set_score(write_step);
			write_dtm(current_pos, me, entry);
			gen_bits->set_bit(current_pos);
			update_max(max_crv, write_step);
			find_new = true;
		}
	}

	atomic_update_max(m_max_step, max_crv);

	return find_new;
}

bool DTM_Generator::change_win_pos_step1(
	In_Out_Param<Thread_Pool> thread_pool,
	Color me, 
	DTM_Score n,
	Out_Param<EGTB_Bits> gen_bits,
	Out_Param<EGTB_Bits> win_bits,
	const EGTB_Bits& pre_bits
)
{
	gen_bits->clear(thread_pool);
	win_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_change_win_pos<Change_Win_Pos_Step::STEP_1>(inout_param(gen_iterator), me, n, inout_param(*gen_bits), inout_param(*win_bits), pre_bits);
		}
	);
	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

bool DTM_Generator::change_win_pos_step2(
	In_Out_Param<Thread_Pool> thread_pool,
	Color me,
	DTM_Score n,
	In_Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& win_bits
)
{
	auto gen_iterator = make_gen_iterator();
	return sp_change_win_pos<Change_Win_Pos_Step::STEP_2>(inout_param(gen_iterator), me, n, inout_param(*gen_bits), {}, win_bits);
}
