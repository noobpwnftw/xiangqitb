#pragma once

#include "egtb.h"
#include "egtb_gen.h"

#include "chess/chess.h"
#include "chess/move.h"
#include "chess/position.h"
#include "chess/piece_config.h"

#include "util/defines.h"
#include "util/utility.h"
#include "util/enum.h"
#include "util/progress_bar.h"

#include <atomic>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>

struct DTM_Generator : public EGTB_Generator
{
	enum struct Load_Bits_Type
	{
		LOAD_LOSE_CHANGE,
		CHANGE_LOSE_POS
	};
	
	enum struct Gen_Pre_Bits_Type {
		NORMAL,
		RULE
	};

	enum struct Change_Win_Pos_Step {
		STEP_1, STEP_2
	};

	NODISCARD static std::optional<EGTB_Generation_Info> dtm_generation_info(const Piece_Config& ps)
	{
		EGTB_Generation_Info info;
		const auto maybe_num_positions = Piece_Config_For_Gen::num_positions_safe(ps);
		if (!maybe_num_positions.has_value())
			return std::nullopt;

		info.num_positions = *maybe_num_positions;

		info.memory_required_for_generation =
			  info.num_positions * (sizeof(DTM_Final_Entry) * 2)
			+ info.num_positions * 5 / 8; // EGTB_Bits

		info.uncompressed_size = info.num_positions * (sizeof(DTM_Final_Entry) * 2);

		info.uncompressed_sub_tb_size = 0;
		for (const auto& [cap, sub_ps] : ps.sub_configs_by_capture())
			if (sub_ps.has_any_free_attackers())
				info.uncompressed_sub_tb_size += 
					Piece_Config_For_Gen(sub_ps).num_positions() * (sizeof(DTM_Final_Entry) * 2);

		return info;
	}

	DTM_Generator(
		const Piece_Config& ps, 
		bool srb,
		const EGTB_Paths& egtb_files
	);

	void gen(In_Out_Param<Thread_Pool> thread_pool);

protected:
	WDL_File_For_Probe m_wdl_file;
	DTM_File_For_Gen m_dtm_file[COLOR_NB];

	std::map<Material_Key, DTM_File_For_Probe> m_sub_dtm_by_material;
	DTM_File_For_Probe* m_sub_dtm_by_capture[PIECE_NB];

	std::atomic<DTM_Score> m_max_step;
	std::atomic<DTM_Score> m_max_build_step[COLOR_NB];

	EGTB_Paths m_egtb_files;
	Temporary_File_Tracker m_tmp_files;
	bool m_save_rule_bits;

	EGTB_Bits m_unknown_bits[COLOR_NB];

	NODISCARD inline bool is_known(const Board_Index pos, const Color me) const
	{
		return !m_unknown_bits[me].bit_is_set(pos);
	}

	NODISCARD inline bool is_unknown(const Board_Index pos, const Color me) const
	{
		return m_unknown_bits[me].bit_is_set(pos);
	}

	template <typename EntryT>
	NODISCARD inline EntryT read_dtm(const Board_Index pos, const Color me) const
	{
		return m_dtm_file[me].read<EntryT>(pos);
	}

	template <typename EntryT>
	inline void write_dtm(const Board_Index pos, const Color me, const EntryT entry)
	{
		m_dtm_file[me].write(entry, pos);
	}

	inline void lock_or_dtm(const Board_Index pos, const Color me, DTM_Rule_Flag flag)
	{
		m_dtm_file[me].lock_add_flags(pos, flag);
	}

	inline void or_dtm(const Board_Index pos, const Color me, DTM_Rule_Flag flag)
	{
		m_dtm_file[me].add_flags(pos, flag);
	}

	void open_sub_egtb();
	void close_sub_egtb();

	void save_egtb(In_Out_Param<Thread_Pool> thread_pool, const EGTB_Info& info);

	void init_entries(In_Out_Param<Thread_Pool> thread_pool);
	void sp_init_entries(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator, 
		In_Out_Param<Concurrent_Progress_Bar> progress_bar
	);

	std::pair<DTM_Any_Entry, WDL_Entry> make_initial_entry(const Position_For_Gen& pos_gen) const;

	NODISCARD DTM_Score search_cap_win_score(const Position_For_Gen& pos_gen) const;

	NODISCARD bool check_double_chase_win(
		Position_For_Gen& pos_gen,
		Move move, 
		Board_Index next_idx,
		const Color me, 
		const bool mirr, 
		const DTM_Score max_step
	) const;

	NODISCARD bool change_lose_pos(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		DTM_Score n,
		Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& pre_bits
	);

	template <Load_Bits_Type TypeV>
	NODISCARD bool sp_load_bits(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTM_Score n,
		In_Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits* pre_bits
	);

	NODISCARD bool load_lose_change(
		In_Out_Param<Thread_Pool> thread_pool,
		const Color me,
		const DTM_Score n,
		Out_Param<EGTB_Bits> gen_bits
	);

	template <Gen_Pre_Bits_Type TypeV>
	NODISCARD bool sp_gen_pre_bits(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTM_Score n,
		const EGTB_Bits& gen_bits,
		In_Out_Param<EGTB_Bits> pre_bits,
		Optional_In_Out_Param<EGTB_Bits> win_bits
	);

	NODISCARD bool gen_pre_bits_normal(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		DTM_Score n,
		const EGTB_Bits& gen_bits,
		Out_Param<EGTB_Bits> pre_bits,
		In_Out_Param<EGTB_Bits> win_bits
	);

	NODISCARD bool gen_pre_bits_rule(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		DTM_Score n,
		const EGTB_Bits& gen_bits,
		Out_Param<EGTB_Bits> pre_bits
	);

	NODISCARD bool sp_save_win(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTM_Score n,
		In_Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& pre_bits,
		In_Out_Param<EGTB_Bits> win_bits
	);

	NODISCARD bool save_win(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		DTM_Score n,
		In_Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& pre_bits,
		In_Out_Param<EGTB_Bits> win_bits
	);

	NODISCARD bool sp_prove_lose(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTM_Score n,
		In_Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& pre_bits,
		const EGTB_Bits& win_bits
	);

	NODISCARD bool prove_lose(In_Out_Param<Thread_Pool> thread_pool, 
		Color me, 
		DTM_Score n,
		In_Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& pre_bits,
		const EGTB_Bits& win_bits
	);

	void sp_second_init(In_Out_Param<Shared_Board_Index_Iterator> gen_iterator, Color root_color);
	void second_init(In_Out_Param<Thread_Pool> thread_pool, Color root_color);

	void sp_init_check_chase(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits,
		In_Out_Param<Concurrent_Progress_Bar> progress_bar
	);

	void init_check_chase(
		In_Out_Param<Thread_Pool> thread_pool, 
		In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
	);

	void sp_load_direct(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		Color me,
		In_Out_Param<EGTB_Bits> gen_bits
	);

	void load_direct(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		Out_Param<EGTB_Bits> gen_bits
	);

	void sp_find_rule_lose(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		Color me,
		In_Out_Param<EGTB_Bits> me_bits,
		In_Out_Param<EGTB_Bits> opp_bits
	);

	void find_rule_lose(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		Out_Param<EGTB_Bits> me_bits, 
		Out_Param<EGTB_Bits> opp_bits
	);

	void sp_save_rule_lose(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		Color me,
		const EGTB_Bits& me_bits
	);

	void save_rule_lose(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		const EGTB_Bits& me_bits
	);

	NODISCARD bool sp_remove_rule_lose(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		Color base_me,
		Color me,
		In_Out_Param<EGTB_Bits> me_bits, 
		const EGTB_Bits& opp_bits
	);

	NODISCARD bool remove_rule_lose(
		In_Out_Param<Thread_Pool> thread_pool,
		Color base_me,
		Color me,
		In_Out_Param<EGTB_Bits> me_bits, 
		const EGTB_Bits& opp_bits
	);

	template <Change_Win_Pos_Step TypeV>
	NODISCARD bool sp_change_win_pos(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		Color me,
		DTM_Score n,
		In_Out_Param<EGTB_Bits> gen_bits,
		Optional_In_Out_Param<EGTB_Bits> win_bits,
		const EGTB_Bits& pre_bits
	);

	NODISCARD bool change_win_pos_step1(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		DTM_Score n,
		Out_Param<EGTB_Bits> gen_bits,
		Out_Param<EGTB_Bits> win_bits,
		const EGTB_Bits& pre_bits
	);

	bool change_win_pos_step2(
		In_Out_Param<Thread_Pool> thread_pool,
		Color me,
		DTM_Score n,
		In_Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& win_bits
	);

	template <WDL_Entry TypeV>
	NODISCARD bool sp_remove_fake(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		In_Out_Param<EGTB_Bits> rule_bits
	);

	NODISCARD bool remove_fake(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		WDL_Entry type,
		In_Out_Param<EGTB_Bits> rule_bits
	);

	NODISCARD EGTB_Info check_dtm_egtb(In_Out_Param<Thread_Pool> thread_pool);
	NODISCARD EGTB_Info sp_check_dtm_egtb(In_Out_Param<Shared_Board_Index_Iterator> gen_iterator);

	void build_steps(In_Out_Param<Thread_Pool> thread_pool, Color root_color, In_Out_Param<EGTB_Bits_Pool> tmp_bits);

	void gen_rule_lose(In_Out_Param<Thread_Pool> thread_pool, In_Out_Param<EGTB_Bits_Pool> tmp_bits);
	void loop_build_check_chase(In_Out_Param<Thread_Pool> thread_pool, In_Out_Param<EGTB_Bits_Pool> tmp_bits);
	void loop_init_check_chase(In_Out_Param<Thread_Pool> thread_pool, In_Out_Param<EGTB_Bits_Pool> tmp_bits);

	NODISCARD DTM_Intermediate_Entry check_remove_lose(Position_For_Gen& pos_gen, DTM_Intermediate_Entry tt) const;
	NODISCARD DTM_Intermediate_Entry check_remove_win(Position_For_Gen& pos_gen, DTM_Intermediate_Entry tt) const;

	NODISCARD DTM_Final_Entry read_sub_tb_dtm(
		const Position_For_Gen& pos,
		Move move
	) const;
};
