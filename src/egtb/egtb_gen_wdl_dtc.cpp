#include "egtb_gen_wdl_dtc.h"

#include "egtb_compress.h"

#include "util/compress.h"
#include "util/dispatch.h"

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>

using namespace std::chrono_literals;

using EGTB_Order_Template_Dispatch = 
	Template_Dispatch<
		DTC_Entry_Order, 
		DTC_Entry_Order::ORDER_64, 
		DTC_Entry_Order::ORDER_128
	>;

using Remove_Fake_Template_Dispatch = 
	Template_Dispatch<
		DTC_Generator::Remove_Fake_Step, 
		DTC_Generator::Remove_Fake_Step::STEP_1, 
		DTC_Generator::Remove_Fake_Step::STEP_2, 
		DTC_Generator::Remove_Fake_Step::STEP_3
	>;

DTC_Generator::DTC_Generator(
	const Piece_Config& ps,
	bool save_wdl,
	bool save_dtc,
	const EGTB_Paths& egtb_files
) :
	EGTB_Generator(ps),
	m_egtb_files(egtb_files),
	m_save_wdl(save_wdl),
	m_save_dtc(save_dtc),
	m_entry_order(DTC_Entry_Order::ORDER_64)
{
	if (!save_wdl && !save_dtc)
		return;

	memset(m_sub_wdl_by_capture, 0, sizeof(m_sub_wdl_by_capture));
}

void DTC_Generator::open_sub_evtb()
{
	for (const Piece i : ALL_PIECES)
	{
		const Piece_Config* sub_ps = m_sub_epsi_by_capture[i];
		if (sub_ps == nullptr || !sub_ps->has_any_free_attackers())
			continue;

		const Material_Key mat_key = sub_ps->base_material_key();
		auto [it, inserted] = m_sub_wdl_by_material.try_emplace(mat_key, m_egtb_files, *sub_ps, false); // never load .gen files
		m_sub_wdl_by_capture[i] = &(it->second);
	}
}

void DTC_Generator::close_sub_evtb()
{
	m_sub_wdl_by_material.clear();
	for (auto& v : m_sub_wdl_by_capture)
		v = nullptr;
	m_tmp_files.clear();
}

EGTB_Info DTC_Generator::gen_evtb(In_Out_Param<Thread_Pool> thread_pool)
{
	EGTB_Info info;

	auto gen_iterator = make_gen_iterator();
	const auto infos = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return TEMPLATE_DISPATCH(
				(EGTB_Order_Template_Dispatch(m_entry_order)),
				sp_gen_evtb, inout_param(gen_iterator)
			);
		}
	);

	for (const Color me : { WHITE, BLACK })
	{
		info.consolidate_from(infos.begin(), infos.end(), me);

		if (info.longest_win[me] > 0)
		{
			info.longest_win[me] -= 1;
			Position_For_Gen pos_gen(m_epsi, static_cast<Board_Index>(info.longest_idx[me]), me);
			pos_gen.get_fen(Span(info.longest_fen[me]));
		}

		info.loop_cnt[0] = info.loop_cnt[1] = narrowing_static_cast<uint8_t>(m_max_order);
	}

	return info;
}

template <DTC_Entry_Order ORDER>
EGTB_Info DTC_Generator::sp_gen_evtb(In_Out_Param<Shared_Board_Index_Iterator> gen_iterator)
{
	EGTB_Info info;

	for (const Board_Index current_pos : gen_iterator->indices())
	{
		for (const Color me : { WHITE, BLACK })
		{
			bool legal;
			DTC_Score value;

			const bool known = is_known(current_pos, me);

			if (known)
			{
				const auto entry = read_dtc<DTC_Final_Entry>(current_pos, me);
				legal = entry.is_legal();
				value = entry.value<ORDER>();
			}
			else
			{
				legal = true;
				value = DTC_SCORE_ZERO;
			}

			WDL_Entry data;
			if (!legal)
			{
				write_dtc(current_pos, me, DTC_Final_Entry::make_draw());
				data = WDL_Entry::ILLEGAL;
			}
			else if (known && (value & 1))
			{
				data = WDL_Entry::LOSE;
			}
			else if (known && value != 0)
			{
				data = WDL_Entry::WIN;
				info.maybe_update_longest_win(me, current_pos, value);
			}
			else
			{
				write_dtc(current_pos, me, DTC_Final_Entry::make_draw());
				data = WDL_Entry::DRAW;
			}

			info.add_result(me, data);

			m_wdl_file[me].write(current_pos, data);
		}
	}

	return info;
}

void DTC_Generator::save_egtb(In_Out_Param<Thread_Pool> thread_pool)
{
	for (const Color me : { WHITE, BLACK })
		m_wdl_file[me].create(m_epsi.num_positions());

	EGTB_Info info = gen_evtb(thread_pool);

	if (m_save_wdl)
	{
		const std::filesystem::path wdl_path = m_egtb_files.wdl_save_path(m_epsi);
		const std::filesystem::path wdl_gen_path = m_egtb_files.wdl_gen_save_path(m_epsi);

		Compressed_EGTB save_info[COLOR_NB];

		for (const Color me : { WHITE, BLACK })
		{
			prepare_evtb_for_compression(thread_pool, m_wdl_file[me].entry_span());

			save_info[me] = save_compress_evtb(
				thread_pool,
				m_wdl_file[me].entry_span(),
				me, 
				info
			);
		}

		{
			const auto colors = table_colors();
			save_evtb_table(m_epsi, save_info, wdl_path, colors, EGTB_Magic::WDL_MAGIC);

			const size_t file_size = std::filesystem::file_size(wdl_path);
			const size_t uncompressed_size = colors.size() * m_epsi.num_positions() / WDL_ENTRY_PACK_RATIO;
			const double compression_ratio = static_cast<double>(uncompressed_size) / file_size;
			printf("Saved compressed WDL file. Compression ratio: x%.2f\n", compression_ratio);
		}

		if (m_is_symmetric)
		{ 
			// force saving both tables
			save_evtb_table(m_epsi, save_info, wdl_gen_path, { WHITE, BLACK }, EGTB_Magic::WDL_MAGIC);

			const size_t file_size = std::filesystem::file_size(wdl_gen_path);
			const size_t uncompressed_size = 2 * m_epsi.num_positions() / WDL_ENTRY_PACK_RATIO;
			const double compression_ratio = static_cast<double>(uncompressed_size) / file_size;
			printf("Saved compressed WDL gen file. Compression ratio: x%.2f\n", compression_ratio);
		}

		for (const Color me : { WHITE, BLACK })
			m_wdl_file[me].close();
	}

	// 压缩写入egtb
	if (m_save_dtc)
	{
		const std::filesystem::path info_path = m_egtb_files.dtc_info_save_path(m_epsi);
		const std::filesystem::path dtc_path = m_egtb_files.dtc_save_path(m_epsi);

		Compressed_EGTB save_info[COLOR_NB];

		for (const Color me : { WHITE, BLACK })
		{
			save_info[me] = save_compress_egtb(
				thread_pool,
				m_dtc_file[me].data_span(),
				me, 
				info, 
				m_entry_order == DTC_Entry_Order::ORDER_128
			);

			if (m_is_symmetric)
				break;
		}

		{
			const auto colors = table_colors();
			save_egtb_table(m_epsi, save_info, dtc_path, colors, EGTB_Magic::DTC_MAGIC);

			const size_t file_size = std::filesystem::file_size(dtc_path);
			const size_t uncompressed_size = colors.size() * m_epsi.num_positions() * sizeof(DTC_File_For_Gen::ENTRY_SIZE);
			const double compression_ratio = static_cast<double>(uncompressed_size) / file_size;
			printf("Saved compressed DTC file. Compression ratio: x%.2f\n", compression_ratio);
		}

		std::ofstream fp(info_path, std::ios_base::binary);
		fp.write(reinterpret_cast<const char*>(&info), sizeof(EGTB_Info));
	}
}

bool DTC_Generator::sp_gen_pre_bits(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const EGTB_Bits& gen_bits,
	In_Out_Param<EGTB_Bits> pre_bits
)
{
	const Color opp = color_opp(me);
	bool ret = false;

	for (const Board_Index current_pos : gen_iterator->indices(gen_bits))
	{
		Position_For_Gen gen_pos(m_epsi, current_pos, me);

		auto& board = gen_pos.board();
		ASSERT(board.is_legal());

		for (const Move move : board.gen_pseudo_legal_pre_quiets())
		{
			for (const Board_Index next_ix : pre_quiet_index(gen_pos, move))
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

bool DTC_Generator::gen_pre_bits(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	const EGTB_Bits& gen_bits,
	Out_Param<EGTB_Bits> pre_bits
)
{
	pre_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_gen_pre_bits(inout_param(gen_iterator), me, gen_bits, inout_param(*pre_bits));
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

template <DTC_Entry_Order ORDER>
bool DTC_Generator::sp_save_win(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const DTC_Score n,
	const EGTB_Bits& pre_bits,
	In_Out_Param<EGTB_Bits> gen_bits,
	In_Out_Param<EGTB_Bits> win_bits
)
{
	bool added_new = false;
	for (const Board_Index current_pos : gen_iterator->indices(pre_bits))
	{
		added_new = true;
		write_dtc(current_pos, me, DTC_Final_Entry::make_score<ORDER>(n, m_max_order));
		m_unknown_bits[me].clear_bit(current_pos);
		gen_bits->set_bit(current_pos);
		win_bits->set_bit(current_pos);
	}

	return added_new;
}

bool DTC_Generator::save_win(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTC_Score n,
	const EGTB_Bits& pre_bits,
	Out_Param<EGTB_Bits> gen_bits,
	In_Out_Param<EGTB_Bits> win_bits
)
{
	gen_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return TEMPLATE_DISPATCH(
				(EGTB_Order_Template_Dispatch(m_entry_order)),
				sp_save_win, inout_param(gen_iterator), me, n, pre_bits, inout_param(*gen_bits), win_bits
			);
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

template <DTC_Entry_Order ORDER>
bool DTC_Generator::sp_prove_lose(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const DTC_Score n,
	const EGTB_Bits& pre_bits,
	In_Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& win_bits
)
{
	bool added_new = 0;
	for (const Board_Index current_pos : gen_iterator->indices(pre_bits))
	{
		ASSERT(is_unknown(current_pos, me));
		auto entry = read_dtc<DTC_Intermediate_Entry>(current_pos, me);
		if (entry.has_flag(DTC_FLAG_CAP_DRAW))
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
			if (!win_bits.bit_is_set(next_ix))
			{
				lose = false;
				break;
			}
		}

		if (lose)
		{
			added_new = true;
			write_dtc(current_pos, me, DTC_Final_Entry::make_score<ORDER>(n, m_max_order));
			m_unknown_bits[me].clear_bit(current_pos);
			gen_bits->set_bit(current_pos);
		}
	}

	return added_new;
}

bool DTC_Generator::prove_lose(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	DTC_Score n,
	const EGTB_Bits& pre_bits,
	Out_Param<EGTB_Bits> gen_bits,
	const EGTB_Bits& win_bits
)
{
	gen_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return TEMPLATE_DISPATCH(
				(EGTB_Order_Template_Dispatch(m_entry_order)),
				sp_prove_lose, inout_param(gen_iterator), me, n, pre_bits, inout_param(*gen_bits), win_bits
			);
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

DTC_Any_Entry DTC_Generator::make_initial_entry(const Position_For_Gen& pos_gen) const
{
	enum Value : int {
		ValueNone = -32767,
		ValueDraw = 0,
		ValueLoss = -20000,
		ValueWin = 20000
	};

	auto& pos = pos_gen.board();

	if (!pos.is_legal())
		return DTC_Final_Entry::make_illegal();

	if (pos.is_draw())
		return DTC_Final_Entry::make_draw();

	const bool in_check = pos.is_in_check();

	if (pos.is_mate(in_check))
		return DTC_Final_Entry::make_lose();

	Value best_value = ValueNone;
	for (const Move move : pos.gen_pseudo_legal_captures())
	{
		if (!pos.is_pseudo_legal_move_legal(move, in_check))
			continue;

		const WDL_Entry ss = read_sub_tb(pos_gen, move);

		Value value = ValueDraw;
		switch (ss)
		{
		case WDL_Entry::WIN:
			value = ValueLoss;
			break;
		case WDL_Entry::LOSE:
			value = ValueWin;
			break;
		default:
			ASSUME(ss == WDL_Entry::DRAW);
		}

		update_max(best_value, value);
		if (best_value > ValueDraw)
			break;
	}

	if (best_value == ValueNone)
		return DTC_Intermediate_Entry{};
	else if (best_value > 0)
		return DTC_Final_Entry::make_win();
	else if (best_value < 0)
		return pos.is_quiet_mate(in_check) 
			   ? DTC_Any_Entry(DTC_Final_Entry::make_lose()) 
		       : DTC_Any_Entry(DTC_Intermediate_Entry{});
	else
		return DTC_Intermediate_Entry::make_cap_draw();
}

void DTC_Generator::sp_init_entries(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	In_Out_Param<Concurrent_Progress_Bar> progress_bar
)
{
	constexpr size_t PROGRESS_BAR_UPDATE_PERIOD = 64 * 64;

	size_t i = 0;

	for (Position_For_Gen& pos_gen : gen_iterator->boards(m_epsi))
	{
		const Board_Index current_pos = pos_gen.board_index();
		ASSERT(current_pos == m_epsi.compose_board_index(pos_gen.index()));

		if (!pos_gen.is_legal())
		{
			write_dtc(current_pos, WHITE, DTC_Final_Entry::make_illegal());
			write_dtc(current_pos, BLACK, DTC_Final_Entry::make_illegal());
			continue;
		}

		for (const Color us : { WHITE, BLACK })
		{
			pos_gen.set_turn(us);

			const DTC_Any_Entry result = make_initial_entry(pos_gen);
			std::visit(overload(
				[&](DTC_Final_Entry entry) {
					write_dtc(current_pos, us, entry); 
				},
				[&](DTC_Intermediate_Entry entry) {
					write_dtc(current_pos, us, entry);
					m_unknown_bits[us].set_bit(current_pos);
				}
			), result);
		}

		if (++i % PROGRESS_BAR_UPDATE_PERIOD == 0)
			*progress_bar += PROGRESS_BAR_UPDATE_PERIOD;
	}
}

void DTC_Generator::init_entries(In_Out_Param<Thread_Pool> thread_pool)
{
	const size_t PRINT_PERIOD = thread_pool->num_workers() * (1 << 20);

	auto gen_iterator = make_gen_iterator();
	Concurrent_Progress_Bar progress_bar(gen_iterator.num_indices(), PRINT_PERIOD, "init_entries");
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			sp_init_entries(inout_param(gen_iterator), inout_param(progress_bar));
		}
	);
	progress_bar.set_finished();
}

void DTC_Generator::load_win_bits(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me, 
	Out_Param<EGTB_Bits> win_bits
)
{
	win_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return TEMPLATE_DISPATCH(
				(EGTB_Order_Template_Dispatch(m_entry_order)),
				sp_load_win_bits, inout_param(gen_iterator), me, inout_param(*win_bits)
			);
		}
	);
}

void DTC_Generator::load_gen_bits(
	In_Out_Param<Thread_Pool> thread_pool,
	Color me,
	DTC_Score n,
	Out_Param<EGTB_Bits> gen_bits
)
{
	ASSERT(m_entry_order == DTC_Entry_Order::ORDER_64);
	gen_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			sp_load_gen_bits(inout_param(gen_iterator), me, n, inout_param(*gen_bits));
		}
	);
}

template <DTC_Entry_Order ORDER>
void DTC_Generator::sp_load_win_bits(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	In_Out_Param<EGTB_Bits> win_bits
)
{
	for (const Board_Index current_pos : gen_iterator->indices())
		if (is_win<ORDER>(current_pos, me))
			win_bits->set_bit(current_pos);
}

void DTC_Generator::sp_load_gen_bits(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me,
	const DTC_Score n,
	In_Out_Param<EGTB_Bits> bits
)
{
	for (const Board_Index current_pos : gen_iterator->indices())
	{
		if (is_unknown(current_pos, me))
			continue;

		const auto entry = read_dtc<DTC_Final_Entry>(current_pos, me);
		if (entry.is_legal() && entry.value<DTC_Entry_Order::ORDER_64>() == n)
			bits->set_bit(current_pos);
	}
}

void DTC_Generator::gen(In_Out_Param<Thread_Pool> thread_pool)
{
	printf("%s gen dtc start...\n", m_epsi.name().c_str());

	for (const Color turn : { WHITE, BLACK })
		m_dtc_file[turn].create(m_epsi.num_positions());

	open_sub_evtb();

	EGTB_Bits_Pool tmp_bits(5, m_epsi.num_positions());

	m_unknown_bits[WHITE] = tmp_bits.acquire_cleared(thread_pool);
	m_unknown_bits[BLACK] = tmp_bits.acquire_cleared(thread_pool);

	init_entries(thread_pool);
	close_sub_evtb();

	// 第二步，迭代获取输赢信息
	m_max_order = DTC_ORDER_ZERO;
	m_max_conv = DTC_SCORE_ZERO;
	m_entry_order = DTC_Entry_Order::ORDER_64;

	build_steps(thread_pool, WHITE, inout_param(tmp_bits));
	build_steps(thread_pool, BLACK, inout_param(tmp_bits));

	loop_build_check_chase(thread_pool, inout_param(tmp_bits));

	// Release some memory for WDL tables and for compression.
	tmp_bits.clear();

	save_egtb(thread_pool);

	tmp_bits.release(std::move(m_unknown_bits[WHITE]));
	tmp_bits.release(std::move(m_unknown_bits[BLACK]));

	for (const Color turn : { WHITE, BLACK })
		m_dtc_file[turn].close();
}

void DTC_Generator::build_steps(
	In_Out_Param<Thread_Pool> thread_pool,
	Color root_color,
	In_Out_Param<EGTB_Bits_Pool> tmp_bits
)
{
	const auto start_time = std::chrono::steady_clock::now();

	EGTB_Bits pre_bits = tmp_bits->acquire_dirty();
	EGTB_Bits win_bits = tmp_bits->acquire_dirty();
	EGTB_Bits gen_bits = tmp_bits->acquire_dirty();

	load_win_bits(thread_pool, root_color, out_param(win_bits));

	Color me = root_color;
	Color opp = color_opp(root_color);
	DTC_Score new_conv = DTC_SCORE_ZERO;
	for (DTC_Score n = static_cast<DTC_Score>(1);; ++n, std::swap(me, opp))
	{
		printf("build conv %zu\r", static_cast<size_t>(n));
		fflush(stdout);

		// For the first two iterations (first for each color) we need to
		// populate the initial gen bits.
		// Next iterations use gen_bits from save_win/prove_lose
		if (n <= 2)
			load_gen_bits(thread_pool, opp, n, out_param(gen_bits));

		// New gen bits are made every iteration.
		const bool more_work = 
			   gen_pre_bits(thread_pool, opp, gen_bits, out_param(pre_bits))
			&& (  me == root_color
				? save_win(thread_pool, me, n + 1, pre_bits, out_param(gen_bits), inout_param(win_bits))
				: prove_lose(thread_pool, me, n + 1, pre_bits, out_param(gen_bits), win_bits));

		if (more_work)
			update_max(new_conv, n + 1);

		if (n >= 2 && !more_work)
			break;
	};

	tmp_bits->release(std::move(pre_bits));
	tmp_bits->release(std::move(win_bits));
	tmp_bits->release(std::move(gen_bits));

	update_min(m_max_conv, new_conv);

	const auto end_time = std::chrono::steady_clock::now();

	printf("%s direct max conv %zu. Done in %s\n",
		root_color == WHITE ? "white" : "black",
		static_cast<size_t>(new_conv),
		format_elapsed_time(start_time, end_time).c_str()
	);
}

bool DTC_Generator::sp_init_check_chase(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	In_Out_Param<Concurrent_Progress_Bar> progress_bar
)
{
	constexpr size_t PROGRESS_BAR_UPDATE_PERIOD = 64 * 64;

	bool label = false;

	size_t i = 0;

	for (Position_For_Gen& pos_gen : gen_iterator->boards(m_epsi))
	{
		const Board_Index current_pos = pos_gen.board_index();
		bool in_check = false;

		for (const Color me : { WHITE, BLACK })
		{
			if (is_known(current_pos, me))
				continue;

			const Color opp = color_opp(me);
			auto& board = pos_gen.board();
			board.set_turn(me);

			ASSERT(board.is_legal());

			// If the other side is already in check then this one is definitely not.
			in_check = !in_check && board.is_in_check();

			const Move_List list =
				in_check
				? board.gen_pseudo_legal_quiets()
				: board.gen_legal_capture_evasions();

			bool find = false;
			for (const Move move : list)
			{
				if (in_check && !board.is_pseudo_legal_move_legal_in_check(move))
					continue;

				// We want to check the mirrored case by design.
				// Otherwise current_pos might not get marked correctly.
				for (const Board_Index next_ix : next_quiet_index_with_mirror(pos_gen, move))
				{
					if (is_unknown(next_ix, opp))
					{
						lock_or_dtc(next_ix, opp, in_check ? (DTC_FLAG_CHECK | DTC_FLAG_CHECK_LOSE) : (DTC_FLAG_CHASE | DTC_FLAG_CHASE_LOSE));
						find = true;
					}
				}
			}

			if (find)
			{
				lock_or_dtc(current_pos, me, in_check ? (DTC_FLAG_IN_CHECK | DTC_FLAG_CHECK_WIN) : (DTC_FLAG_IN_CHASE | DTC_FLAG_CHASE_WIN));
				label = true;
			}
		}

		if (++i % PROGRESS_BAR_UPDATE_PERIOD == 0)
			*progress_bar += PROGRESS_BAR_UPDATE_PERIOD;
	}

	return label;
}

bool DTC_Generator::init_check_chase(In_Out_Param<Thread_Pool> thread_pool)
{
	const size_t PRINT_PERIOD = thread_pool->num_workers() * (1 << 20);

	auto gen_iterator = make_gen_iterator();
	Concurrent_Progress_Bar progress_bar(gen_iterator.num_indices(), PRINT_PERIOD, "init_check_chase");
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return sp_init_check_chase(inout_param(gen_iterator), inout_param(progress_bar));
		}
	);
	progress_bar.set_finished();

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

void DTC_Generator::sp_label_may_check_chase(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
)
{
	constexpr DTC_Intermediate_Entry_Flag set_flags = DTC_FLAG_CHECK_WIN | DTC_FLAG_CHECK_LOSE | DTC_FLAG_CHASE_WIN | DTC_FLAG_CHASE_LOSE;
	constexpr DTC_Intermediate_Entry_Flag rule_flags = DTC_FLAG_IN_CHECK | DTC_FLAG_CHECK | DTC_FLAG_IN_CHASE | DTC_FLAG_CHASE;

	for (const Board_Index current_pos : gen_iterator->indices())
	{
		for (const Color me : { WHITE, BLACK })
		{
			if (is_known(current_pos, me))
				continue;

			const auto entry = read_dtc<DTC_Intermediate_Entry>(current_pos, me);
			if (!entry.has_flag(rule_flags))
				continue;

			auto new_entry = entry;
			new_entry.clear_flag(set_flags);

			if (new_entry.has_flag(DTC_FLAG_IN_CHECK))
				new_entry.set_flag(DTC_FLAG_CHECK_WIN);

			if (new_entry.has_flag(DTC_FLAG_IN_CHASE))
				new_entry.set_flag(DTC_FLAG_CHASE_WIN);

			if (new_entry.has_flag(DTC_FLAG_CHECK))
				new_entry.set_flag(DTC_FLAG_CHECK_LOSE);

			if (new_entry.has_flag(DTC_FLAG_CHASE))
				new_entry.set_flag(DTC_FLAG_CHASE_LOSE);

			if (new_entry.has_flag(DTC_FLAG_CAP_DRAW))
			{
				if (new_entry.has_flag(DTC_FLAG_CHECK_LOSE) && !new_entry.has_flag(DTC_FLAG_CHECK_WIN))
					new_entry.clear_flag(DTC_FLAG_CHECK_LOSE);

				if (new_entry.has_flag(DTC_FLAG_CHASE_LOSE) && !new_entry.has_flag(DTC_FLAG_CHASE_WIN))
					new_entry.clear_flag(DTC_FLAG_CHASE_LOSE);
			}

			if (new_entry != entry)
				write_dtc(current_pos, me, new_entry);

			if (new_entry.has_flag(set_flags))
				rule_bits[me].set_bit(current_pos);
		}
	}
}

void DTC_Generator::label_may_check_chase(
	In_Out_Param<Thread_Pool> thread_pool, 
	Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
)
{
	rule_bits[WHITE].clear(thread_pool);
	rule_bits[BLACK].clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			sp_label_may_check_chase(inout_param(gen_iterator), inout_param(*rule_bits));
		}
	);
}

template <DTC_Entry_Order ORDER>
bool DTC_Generator::sp_label_real_check_chase(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me,
	In_Out_Param<EGTB_Bits> gen_bits
)
{
	bool find_new = false;

	for (const Board_Index current_pos : gen_iterator->indices())
	{
		if (is_known(current_pos, me))
			continue;

		auto entry = read_dtc<DTC_Intermediate_Entry>(current_pos, me);

		for (const auto &[flag_lose, flag_win] : { std::make_pair(DTC_FLAG_CHECK_LOSE, DTC_FLAG_CHECK_WIN),
												   std::make_pair(DTC_FLAG_CHASE_LOSE, DTC_FLAG_CHASE_WIN) })
		{
			if (!entry.has_flag(flag_lose | flag_win))
				continue;

			if (    entry.has_flag(flag_lose)
				&& !entry.has_flag(flag_win)
				&& !entry.has_flag(DTC_FLAG_CAP_DRAW))
			{
				find_new = true;
				write_dtc(current_pos, me, DTC_Final_Entry::make_score<ORDER>(DTC_SCORE_TERMINAL_LOSS, m_max_order));
				gen_bits->set_bit(current_pos);
				m_unknown_bits[me].clear_bit(current_pos);
			}

			break;
		}
	}

	return find_new;
}

bool DTC_Generator::label_real_check_chase(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color me,
	Out_Param<EGTB_Bits> gen_bits
)
{
	gen_bits->clear(thread_pool);
	auto gen_iterator = make_gen_iterator();
	const auto ret = thread_pool->run_sync_task_on_all_threads(
		[&](size_t thread_id) {
			return TEMPLATE_DISPATCH(
				(EGTB_Order_Template_Dispatch(m_entry_order)),
				sp_label_real_check_chase, inout_param(gen_iterator), me, inout_param(*gen_bits)
			);
		}
	);

	return std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
}

template <DTC_Entry_Order ORDER, DTC_Generator::Remove_Fake_Step TypeV>
bool DTC_Generator::sp_remove_fake(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me, 
	const DTC_Score n,
	In_Out_Param<EGTB_Bits> rule_bits
)
{
	const auto flag_mask =
		(n & 1)
		? (DTC_FLAG_CHASE_LOSE | DTC_FLAG_CHECK_LOSE)
		: (DTC_FLAG_CHASE_WIN | DTC_FLAG_CHECK_WIN);

	bool find = false;
	for (const Board_Index current_pos : gen_iterator->indices(*rule_bits))
	{
		if (is_known(current_pos, me))
			continue;

		const auto entry = read_dtc<DTC_Intermediate_Entry>(current_pos, me);

		// 检验输棋盘面
		// 检验赢棋盘面
		if (!entry.has_flag(flag_mask))
			continue;

		Position_For_Gen pos_gen(m_epsi, current_pos, me);

		const auto new_entry =
			(n & 1)
			? check_remove_lose<ORDER, TypeV>(pos_gen, entry)
			: check_remove_win<ORDER, TypeV>(pos_gen, entry);

		if (entry != new_entry)
		{
			find = true;
			write_dtc(current_pos, me, new_entry);

			if (!new_entry.has_flag(DTC_FLAG_CHASE_WIN | DTC_FLAG_CHECK_WIN | DTC_FLAG_CHASE_LOSE | DTC_FLAG_CHECK_LOSE))
				rule_bits->clear_bit(current_pos);
		}
	}

	return find;
}

bool DTC_Generator::sp_remove_fake_step4(
	In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
	const Color me,
	const EGTB_Bits& rule_bits
)
{
	bool find = false;
	for (const Board_Index current_pos : gen_iterator->indices(rule_bits))
	{
		if (is_known(current_pos, me))
			continue;

		auto entry = read_dtc<DTC_Intermediate_Entry>(current_pos, me);

		const bool is_check_winlose = entry.has_flag(DTC_FLAG_CHECK_WIN) && entry.has_flag(DTC_FLAG_CHECK_LOSE);
		const bool is_chase_winlose = entry.has_flag(DTC_FLAG_CHASE_WIN) && entry.has_flag(DTC_FLAG_CHASE_LOSE);
		if (is_check_winlose || is_chase_winlose)
		{
			find = true;

			if (is_check_winlose)
				entry.clear_flag(DTC_FLAG_CHECK_WIN | DTC_FLAG_CHECK_LOSE);

			if (is_chase_winlose)
				entry.clear_flag(DTC_FLAG_CHASE_WIN | DTC_FLAG_CHASE_LOSE);

			write_dtc(current_pos, me, entry);
		}
	}

	return find;
}

template <DTC_Entry_Order ORDER, DTC_Generator::Remove_Fake_Step TypeV>
DTC_Intermediate_Entry DTC_Generator::check_remove_lose(
	Position_For_Gen& pos_gen,
	DTC_Intermediate_Entry tt
) const
{
	bool long_check = tt.has_flag(DTC_FLAG_CHECK_LOSE);
	bool long_chase = tt.has_flag(DTC_FLAG_CHASE_LOSE);

	if (tt.has_flag(DTC_FLAG_CAP_DRAW))
	{
		bool changed = false;
		if (long_check && !tt.has_flag(DTC_FLAG_CHECK_WIN))
		{
			tt.clear_flag(DTC_FLAG_CHECK_LOSE);
			changed = true;
		}
		if (long_chase && !tt.has_flag(DTC_FLAG_CHASE_WIN))
		{
			tt.clear_flag(DTC_FLAG_CHASE_LOSE);
			changed = true;
		}
		if (changed)
			return tt;
	}

	const bool double_check = long_check && tt.has_flag(DTC_FLAG_CHECK_WIN);
	const bool double_chase = long_chase && tt.has_flag(DTC_FLAG_CHASE_WIN);
	Position& board = pos_gen.board();
	const Board_Index current_pos = pos_gen.board_index();
	const Color me = board.turn();
	const Color opp = color_opp(me);

	auto resolve_long_chase = [&](const Move_List& chase_list) {
		if (chase_list.empty()) // 没找到捉子招法，所有招法都是输
		{
			if (tt.has_flag(DTC_FLAG_CAP_DRAW))
				tt.clear_flag(DTC_FLAG_CHASE_LOSE | DTC_FLAG_CHASE_WIN);
			else
			{
				printf("find chase pos %zu is wrong\n", current_pos);
				board.display();
				abort();
			}

			return;
		}

		if (!board.always_has_attack_after_quiet_moves(chase_list))
		{
			tt.clear_flag(DTC_FLAG_CHASE_LOSE);
			return;
		}

		if (   TypeV == Remove_Fake_Step::STEP_1
			|| !board.is_in_check())
			return;

		// 被将军了，判断对方是否常将
		for (const Move move : chase_list)
		{
			bool mirr;
			const Board_Index next_ix = next_quiet_index(pos_gen, move, out_param(mirr));

			Position_For_Gen next_pos_gen(pos_gen, move, next_ix, mirr);
			auto& next_board = next_pos_gen.board();

			bool find_no_check = false;
			for (const Move move2 : next_board.gen_pseudo_legal_quiets())
			{
				const Board_Index next_ix2 = next_quiet_index(next_pos_gen, move2);
				if (is_known(next_ix2, me))
					continue;
			
				const auto entry2 = read_dtc<DTC_Intermediate_Entry>(next_ix2, me);

				if (   entry2.has_flag(DTC_FLAG_CHASE_LOSE)
					&& !next_board.is_move_check(move2)
					&& next_board.is_move_evasion(move2))
				{
					find_no_check = true;
					break;
				}
			}

			if (!find_no_check)
			{
				// 都在将军
				tt.clear_flag(DTC_FLAG_CHASE_LOSE);
				return;
			}
		}
	};

	auto on_check_list_empty = [&]() {
		if (tt.has_flag(DTC_FLAG_CAP_DRAW))
			tt.clear_flag(DTC_FLAG_CHECK_LOSE | DTC_FLAG_CHECK_WIN);
		else
		{
			printf("find chase pos %zu is wrong\n", current_pos);
			board.display();
			abort();
		}
	};

	const Move_List list = board.gen_pseudo_legal_quiets();

	if (double_check)
	{
		bool check_list_empty = true;
		bool find_check = false;
		bool find_no_check = false;
		bool other_check_win = false;

		for (const Move move : list)
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);

			if (is_known(next_ix, opp))
			{
				const auto entry = read_dtc<DTC_Final_Entry>(next_ix, opp);
				if (entry.is_loss_or_draw<ORDER>())
					find_no_check = true;
			}
			else
			{
				const auto entry = read_dtc<DTC_Intermediate_Entry>(next_ix, opp);

				if (!entry.has_flag(DTC_FLAG_CHECK_WIN))
				{
					find_no_check = true;
					if (TypeV == Remove_Fake_Step::STEP_3 && entry.has_flag(DTC_FLAG_CHECK_LOSE))
						other_check_win = true;
				}
				else
				{
					check_list_empty = false;
					if (entry.has_flag(DTC_FLAG_CHECK_LOSE))
						find_check = true;
				}
			}

			// Check for early exit
			// find_check == true implies check_list_empty == false
			// other_check_win == true implies find_no_check == true
			// Before STEP_3 other_check_win is irrelevant.
			if (find_check && (TypeV == Remove_Fake_Step::STEP_3 ? other_check_win : find_no_check))
				break;
		}

		// other_check_win can only be true when type == Remove_Fake_Step::STEP_3
		if (find_check && !other_check_win)
			return tt;

		if (find_no_check)
		{
			tt.clear_flag(DTC_FLAG_CHECK_LOSE);
			return tt;
		}

		if (check_list_empty) // 没找到将军招法，所有招法都是输
			on_check_list_empty();
	}
	else if (double_chase)
	{
		Move_List chase_list;

		bool find_chase = false;
		bool find_no_chase = false;
		bool other_chase_win = false;
		for (const Move move : list)
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);
			if (is_known(next_ix, opp))
			{
				const auto entry = read_dtc<DTC_Final_Entry>(next_ix, opp);
				if (!entry.is_loss_or_draw<ORDER>())
					continue;

				if (long_check)
				{
					long_check = false;
					tt.clear_flag(DTC_FLAG_CHECK_LOSE);
				}

				if (long_chase)
					find_no_chase = true;
			}
			else
			{
				const auto entry = read_dtc<DTC_Intermediate_Entry>(next_ix, opp);

				if (   long_check
					&& !entry.has_flag(DTC_FLAG_CHECK_WIN))
				{
					long_check = false;
					tt.clear_flag(DTC_FLAG_CHECK_LOSE);
				}

				if (long_chase)
				{
					if (!entry.has_flag(DTC_FLAG_CHASE_WIN))
					{
						find_no_chase = true;
						if (   TypeV == Remove_Fake_Step::STEP_3
							&& !other_chase_win
							&& entry.has_flag(DTC_FLAG_CHASE_LOSE)
							&& !entry.has_flag(DTC_FLAG_CHECK_WIN)
							&& board.is_move_evasion(move))
						{
							other_chase_win = true;
						}
					}
					else
						chase_list.add(move);

					if (   !find_chase
						&& entry.has_flag(DTC_FLAG_CHASE_WIN)
						&& entry.has_flag(DTC_FLAG_CHASE_LOSE)
						&& board.is_move_evasion(move)
						&& board.has_attack_after_quiet_move(move))
					{
						find_chase = true;
					}
				}
			}

			if (!long_chase && !long_check)
				break;
		}

		if (find_chase && !other_chase_win)
			return tt;

		if (find_no_chase)
		{
			tt.clear_flag(DTC_FLAG_CHASE_LOSE);
			return tt;
		}

		resolve_long_chase(chase_list);
	}
	else
	{
		Move_List chase_list;

		bool check_list_empty = true;
		for (const Move move : list)
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);
			if (is_known(next_ix, opp))
			{
				const auto entry = read_dtc<DTC_Final_Entry>(next_ix, opp);
				if (!entry.is_loss_or_draw<ORDER>())
					continue;

				if (long_check)
				{
					long_check = false;
					tt.clear_flag(DTC_FLAG_CHECK_LOSE);
				}

				if (long_chase)
				{
					long_chase = false;
					tt.clear_flag(DTC_FLAG_CHASE_LOSE);
				}
			}
			else
			{
				const auto entry = read_dtc<DTC_Intermediate_Entry>(next_ix, opp);

				if (long_check)
				{
					if (!entry.has_flag(DTC_FLAG_CHECK_WIN))
					{
						long_check = false;
						tt.clear_flag(DTC_FLAG_CHECK_LOSE);
					}
					else
						check_list_empty = false;
				}

				if (long_chase)
				{
					if (!entry.has_flag(DTC_FLAG_CHASE_WIN))
					{
						long_chase = false;
						tt.clear_flag(DTC_FLAG_CHASE_LOSE);
					}
					else
						chase_list.add(move);
				}
			}

			if (!long_chase && !long_check)
				break;
		}

		if (long_chase && long_check)
		{
			printf("find check and chase pos %zu is wrong\n", current_pos);
			board.display();
			abort();
		}

		if (long_check && check_list_empty)
			on_check_list_empty();
		else if (long_chase)
			resolve_long_chase(chase_list);
	}

	return tt;
}

template <DTC_Entry_Order ORDER, DTC_Generator::Remove_Fake_Step TypeV>
DTC_Intermediate_Entry DTC_Generator::check_remove_win(
	Position_For_Gen& pos_gen,
	DTC_Intermediate_Entry tt
) const
{
	Position& board = pos_gen.board();
	const Board_Index current_pos = pos_gen.board_index();
	const Color me = board.turn();
	const Color opp = color_opp(me);

	if (tt.has_flag(DTC_FLAG_CHECK_WIN))
	{
		ASSERT(tt.has_flag(DTC_FLAG_IN_CHECK));

		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			const Board_Index next_ix = next_quiet_index(pos_gen, move);
			if (is_known(next_ix, opp))
				continue;

			const auto entry = read_dtc<DTC_Intermediate_Entry>(next_ix, opp);

			if (entry.has_flag(DTC_FLAG_CHECK_LOSE))
				return tt;
		}

		tt.clear_flag(DTC_FLAG_CHECK_WIN);
		return tt;
	}

	ASSERT(tt.has_flag(DTC_FLAG_CHASE_WIN));
	ASSERT(!tt.has_flag(DTC_FLAG_CHECK_WIN));

	auto is_actually_long_chase = [&](Position& next_board, const Move_List& chase_list2, const Bitboard& evt_piecebb)
	{
		if (chase_list2.empty())
		{
			printf("find chase pos 2 %zu is wrong\n", current_pos);
			next_board.display();
			abort();
		}

		return next_board.always_has_attack_after_quiet_moves(chase_list2, evt_piecebb);
	};

	if (tt.has_flag(DTC_FLAG_CHASE_WIN) && tt.has_flag(DTC_FLAG_CHASE_LOSE))
	{
		bool find_draw_moves = tt.has_flag(DTC_FLAG_CAP_DRAW);

		constexpr size_t MAX_CHASE_WIN_CNT = 64;
		Fixed_Vector<std::tuple<Bitboard, Board_Index, Move, bool>, MAX_CHASE_WIN_CNT> chase_win_tb;

		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			bool mirr;
			const Board_Index next_ix = next_quiet_index(pos_gen, move, out_param(mirr));
			if (is_known(next_ix, opp))
			{
				const auto entry = read_dtc<DTC_Final_Entry>(next_ix, opp);
				if (   !find_draw_moves 
					&& entry.is_loss_or_draw<ORDER>())
					find_draw_moves = true;
			}
			else
			{
				const auto entry = read_dtc<DTC_Intermediate_Entry>(next_ix, opp);

				Bitboard evt_piecebb;
				if (   entry.has_flag(DTC_FLAG_CHASE_LOSE)
					&& board.is_move_evasion(move, out_param(evt_piecebb)))
				{
					if (TypeV == Remove_Fake_Step::STEP_1)
						return tt;

					chase_win_tb.emplace_back(evt_piecebb, next_ix, move, mirr);
				}

				if (   !find_draw_moves
					&& !entry.has_flag(DTC_FLAG_CHASE_WIN))
				{
					find_draw_moves = true;
				}
			}
		}

		if (chase_win_tb.empty())
		{
			tt.clear_flag(DTC_FLAG_CHASE_WIN);
			return tt;
		}

		const bool consider_double_chase = !(find_draw_moves && TypeV <= Remove_Fake_Step::STEP_2);

		for (const auto& [evt_piecebb, next_ix, move, mirr] : chase_win_tb)
		{
			ASSERT(is_unknown(next_ix, opp));
			const auto entry = read_dtc<DTC_Intermediate_Entry>(next_ix, opp);

			const bool double_chase =    consider_double_chase 
						              && entry.has_flag(DTC_FLAG_CHASE_WIN)
						              && board.has_attack_after_quiet_move(move);

			Position_For_Gen next_pos_gen(pos_gen, move, next_ix, mirr);
			auto& next_board = next_pos_gen.board();

			Move_List chase_list2;
			Move_List chase_list3;

			bool long_chase = true;
			bool find_chase = false;
			bool other_chase = false;

			for (const Move move2 : next_board.gen_pseudo_legal_quiets())
			{
				const Board_Index next_ix2 = next_quiet_index(next_pos_gen, move2);
				if (is_known(next_ix2, me))
				{
					const auto entry2 = read_dtc<DTC_Final_Entry>(next_ix2, me);
					if (entry2.is_loss_or_draw<ORDER>())
					{
						long_chase = false;
						if (!double_chase)
							break;
					}
				}
				else
				{
					const auto entry2 = read_dtc<DTC_Intermediate_Entry>(next_ix2, me);

					if (!entry2.has_flag(DTC_FLAG_CHASE_WIN))
					{
						long_chase = false;
						if (!double_chase)
							break;

						if (   TypeV == Remove_Fake_Step::STEP_3
							&& !other_chase
							&& entry2.has_flag(DTC_FLAG_CHASE_LOSE)
							&& !entry2.has_flag(DTC_FLAG_CHECK_WIN)
							&& next_board.is_move_evasion(move2))
						{
							other_chase = true;
						}
					}
					else
						chase_list2.add(move2);

					if (   double_chase
						&& entry2.has_flag(DTC_FLAG_CHASE_WIN)
						&& entry2.has_flag(DTC_FLAG_CHASE_LOSE)
						&& next_board.is_move_evasion(move2))
					{
						find_chase = true;
						chase_list3.add(move2);
					}
				}
			}

			const Bitboard adjusted_evt_piecebb = evt_piecebb.maybe_mirror_files(mirr);

			if (   find_chase 
				&& (!other_chase || TypeV <= Remove_Fake_Step::STEP_2))
			{
				for (const Move move3 : chase_list3)
					if (next_board.has_attack_after_quiet_move(move3, adjusted_evt_piecebb))
						return tt;
			}

			if (   long_chase
				&& is_actually_long_chase(next_board, chase_list2, adjusted_evt_piecebb))
				return tt;
		}
	}
	else
	{
		auto is_long_chase = [&](const Position_For_Gen& next_pos_gen, In_Out_Param<Move_List> chase_list2)
		{
			auto& next_board = next_pos_gen.board();

			for (const Move move2 : next_board.gen_pseudo_legal_quiets())
			{
				const Board_Index next_ix2 = next_quiet_index(next_pos_gen, move2);
				if (is_known(next_ix2, me))
				{
					const auto entry2 = read_dtc<DTC_Final_Entry>(next_ix2, me);
					if (entry2.is_loss_or_draw<ORDER>())
						return false;
				}
				else
				{
					const auto entry2 = read_dtc<DTC_Intermediate_Entry>(next_ix2, me);

					if (entry2.has_flag(DTC_FLAG_CHASE_WIN))
						chase_list2->add(move2);
					else
						return false;
				}
			}

			return true;
		};

		for (const Move move : board.gen_pseudo_legal_quiets())
		{
			bool mirr;
			const Board_Index next_ix = next_quiet_index(pos_gen, move, out_param(mirr));
			if (is_known(next_ix, opp))
				continue;

			const auto entry = read_dtc<DTC_Intermediate_Entry>(next_ix, opp);
			if (!entry.has_flag(DTC_FLAG_CHASE_LOSE))
				continue;

			Bitboard evt_piecebb;
			if (!board.is_move_evasion(move, out_param(evt_piecebb)))
				continue;

			if constexpr (TypeV == Remove_Fake_Step::STEP_1)
				return tt;

			Position_For_Gen next_pos_gen(pos_gen, move, next_ix, mirr);
			Move_List chase_list2;

			if (   is_long_chase(next_pos_gen, inout_param(chase_list2))
				&& is_actually_long_chase(next_pos_gen.board(), chase_list2, evt_piecebb.maybe_mirror_files(mirr)))
				return tt;
		}
	}

	tt.clear_flag(DTC_FLAG_CHASE_WIN);
	return tt;
}

bool DTC_Generator::remove_fake(
	In_Out_Param<Thread_Pool> thread_pool, 
	DTC_Score n,
	Remove_Fake_Step type, 
	In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
)
{
	bool rmv_new = false;

	for (const Color me : { WHITE, BLACK })
	{
		auto gen_iterator = make_gen_iterator();
		const auto ret = thread_pool->run_sync_task_on_all_threads(
			[&](size_t thread_id) {
				return TEMPLATE_DISPATCH(
					std::make_tuple(
						EGTB_Order_Template_Dispatch(m_entry_order),
						Remove_Fake_Template_Dispatch(type)
					),
					sp_remove_fake, inout_param(gen_iterator), me, n, inout_param(rule_bits[me])
				);
			}
		);

		rmv_new = rmv_new || std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
	}

	return rmv_new;
}

bool DTC_Generator::remove_fake_step4(
	In_Out_Param<Thread_Pool> thread_pool,
	const EGTB_Bits rule_bits[COLOR_NB]
)
{
	bool rmv_new = false;

	for (const Color me : { WHITE, BLACK })
	{
		auto gen_iterator = make_gen_iterator();
		const auto ret = thread_pool->run_sync_task_on_all_threads(
			[&](size_t thread_id) {
				return sp_remove_fake_step4(inout_param(gen_iterator), me, rule_bits[me]);
			}
		);

		rmv_new = rmv_new || std::any_of(ret.begin(), ret.end(), [](const bool ret) { return ret; });
	}

	return rmv_new;
}

void DTC_Generator::remove_fake_check_chase(
	In_Out_Param<Thread_Pool> thread_pool, 
	In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
)
{
	bool find = false;
	size_t i = 0;

	for (auto type : { Remove_Fake_Step::STEP_1, Remove_Fake_Step::STEP_2, Remove_Fake_Step::STEP_3 })
		for (DTC_Score n = static_cast<DTC_Score>(1); n <= 2 || find; ++n)
		{
			printf("remove_fake %zu\r", ++i);
			fflush(stdout);
			find = remove_fake(thread_pool, n, type, rule_bits);
		}

	if (remove_fake_step4(thread_pool, *rule_bits))
		for (DTC_Score n = static_cast<DTC_Score>(1); n <= 2 || find; ++n)
		{
			printf("remove_fake %zu\r", ++i);
			fflush(stdout);
			find = remove_fake(thread_pool, n, Remove_Fake_Step::STEP_2, rule_bits);
		}

	printf("remove_fake finished in %zu steps\n", i);
}

bool DTC_Generator::build_check_chase(
	In_Out_Param<Thread_Pool> thread_pool, 
	Color root_color, 
	In_Out_Param<EGTB_Bits_Pool> tmp_bits
)
{
	EGTB_Bits gen_bits = tmp_bits->acquire_dirty();
	if (!label_real_check_chase(thread_pool, color_opp(root_color), out_param(gen_bits)))
	{
		tmp_bits->release(std::move(gen_bits));
		return false;
	}

	const auto start_time = std::chrono::steady_clock::now();

	EGTB_Bits win_bits = tmp_bits->acquire_dirty();
	EGTB_Bits pre_bits = tmp_bits->acquire_dirty();

	load_win_bits(thread_pool, root_color, out_param(win_bits));

	Color me = root_color;
	Color opp = color_opp(root_color);
	DTC_Score n = static_cast<DTC_Score>(1);
	for (;; ++n)
	{
		printf("build conv %zu\r", static_cast<size_t>(n));
		fflush(stdout);

		if (!gen_pre_bits(thread_pool, opp, gen_bits, out_param(pre_bits)))
			break;
		
		const bool ok =
			me == root_color
			? save_win(thread_pool, me, n + 1, pre_bits, out_param(gen_bits), inout_param(win_bits))
			: prove_lose(thread_pool, me, n + 1, pre_bits, out_param(gen_bits), win_bits);

		if (!ok)
			break;

		std::swap(me, opp);
	}

	tmp_bits->release(std::move(pre_bits));
	tmp_bits->release(std::move(win_bits));
	tmp_bits->release(std::move(gen_bits));

	const auto end_time = std::chrono::steady_clock::now();

	printf(
		"%s max conv %zu. Done in %s\n", 
		root_color == WHITE ? "white" : "black", 
		static_cast<size_t>(n),
		format_elapsed_time(start_time, end_time).c_str()
	);

	if (m_entry_order != DTC_Entry_Order::ORDER_128 && n > m_max_conv)
		m_max_conv = n;

	return true;
}

void DTC_Generator::loop_build_check_chase(In_Out_Param<Thread_Pool> thread_pool, In_Out_Param<EGTB_Bits_Pool> tmp_bits)
{
	if (!m_epsi.both_sides_have_free_attackers())
		return;

	bool build_finish[COLOR_NB] = { false, false };

	if (!init_check_chase(thread_pool))
		return;

	const auto start_time = std::chrono::steady_clock::now();

	m_max_order = static_cast<DTC_Order>(1);
	for (;;)
	{
		printf("order = %zu\n", static_cast<size_t>(m_max_order));

		EGTB_Bits rule_bits[COLOR_NB] = { tmp_bits->acquire_dirty(), tmp_bits->acquire_dirty() };

		label_may_check_chase(thread_pool, out_param(rule_bits));
		remove_fake_check_chase(thread_pool, inout_param(rule_bits));

		tmp_bits->release(std::move(rule_bits[WHITE]));
		tmp_bits->release(std::move(rule_bits[BLACK]));

		for (const Color me : { WHITE, BLACK })
			build_finish[me] = build_finish[me] || !build_check_chase(thread_pool, me, tmp_bits);

		if (build_finish[WHITE] && build_finish[BLACK])
			break;

		++m_max_order;

		if (m_max_order > DTC_ORDER_MAX_ORDER_64)
		{
			if (m_entry_order != DTC_Entry_Order::ORDER_128)
			{
				if (!DTC_Final_Entry::is_value_ambiguous_with_order_128(m_max_conv))
				{
					m_entry_order = DTC_Entry_Order::ORDER_128;
					printf("order over 63 expand to 127.\n");
				}
				else
					printf("order over 63, cap_score will be not exact...\n");
			}
			else if (m_max_order > DTC_ORDER_MAX_ORDER_128)
				printf("order over 127, cap_score will be not exact...\n");
		}
	}

	const auto end_time = std::chrono::steady_clock::now();

	printf(
		"max order %zu. Done in %s\n", 
		static_cast<size_t>(m_max_order),
		format_elapsed_time(start_time, end_time).c_str()
	);
}

WDL_Entry DTC_Generator::read_sub_tb(
	const Position_For_Gen& pos_gen,
	Move move
) const
{
	const Position& pos = pos_gen.board();
	const Square to = move.to();
	const Piece piece = pos.piece_on(to);

	// The only unpopulated EGTBs are the ones without free attackers, so drawn.
	if (m_sub_wdl_by_capture[piece] == nullptr)
		return WDL_Entry::DRAW;

	const Board_Index next_ix = next_cap_index(pos_gen, move);
	return m_sub_wdl_by_capture[piece]->read(m_sub_read_color_by_capture[piece], next_ix);
}
