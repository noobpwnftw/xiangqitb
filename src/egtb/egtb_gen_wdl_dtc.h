#pragma once

#include "egtb.h"
#include "egtb_gen.h"

#include "chess/chess.h"
#include "chess/position.h"
#include "chess/piece_config.h"

#include "util/defines.h"
#include "util/enum.h"
#include "util/filesystem.h"
#include "util/utility.h"
#include "util/progress_bar.h"
#include "util/compress.h"

#include <vector>
#include <string>
#include <map>
#include <cstdlib>

struct DTC_Generator : public EGTB_Generator
{
	enum struct Load_Bits_Type
	{
		LOAD_WIN = 1, LOAD_GEN
	};

	enum struct Remove_Fake_Step
	{
		STEP_1, STEP_2, STEP_3
	};

	NODISCARD static std::optional<EGTB_Generation_Info> wdl_generation_info(const Piece_Config& ps)
	{
		EGTB_Generation_Info info;
		const auto maybe_num_positions = Piece_Config_For_Gen::num_positions_safe(ps);
		if (!maybe_num_positions.has_value())
			return std::nullopt;

		info.num_positions = *maybe_num_positions;

		info.memory_required_for_generation =
			  info.num_positions * (sizeof(DTC_Final_Entry) * 2)
			+ info.num_positions * 5 / 8; // EGTB_Bits

		info.uncompressed_size = info.num_positions * sizeof(WDL_Entry) * 2 / WDL_ENTRY_PACK_RATIO;

		info.uncompressed_sub_tb_size = 0;
		for (const auto& [cap, sub_ps] : ps.sub_configs_by_capture())
			if (sub_ps.has_any_free_attackers())
				info.uncompressed_sub_tb_size += 
					Piece_Config_For_Gen(sub_ps).num_positions() * sizeof(WDL_Entry) * 2 / WDL_ENTRY_PACK_RATIO;

		return info;
	}

	NODISCARD static std::optional<EGTB_Generation_Info> dtc_generation_info(const Piece_Config& ps)
	{
		EGTB_Generation_Info info;
		const auto maybe_num_positions = Piece_Config_For_Gen::num_positions_safe(ps);
		if (!maybe_num_positions.has_value())
			return std::nullopt;

		info.num_positions = *maybe_num_positions;

		info.memory_required_for_generation =
			  info.num_positions * (sizeof(DTC_Final_Entry) * 2)
			+ info.num_positions * 5 / 8; // EGTB_Bits

		info.uncompressed_size = info.num_positions * (sizeof(DTC_Final_Entry) * 2);

		info.uncompressed_sub_tb_size = 0;
		for (const auto& [cap, sub_ps] : ps.sub_configs_by_capture())
			if (sub_ps.has_any_free_attackers())
				info.uncompressed_sub_tb_size += 
					Piece_Config_For_Gen(sub_ps).num_positions() * (sizeof(DTC_Final_Entry) * 2);

		return info;
	}

	DTC_Generator(
		const Piece_Config& ps, 
		bool save_wdl,
		bool save_dtc,
		const EGTB_Paths& egtb_files
	);

	void gen(In_Out_Param<Thread_Pool> thread_pool);

protected:
	WDL_File_For_Gen m_wdl_file[COLOR_NB];
	DTC_File_For_Gen m_dtc_file[COLOR_NB];

	std::map<Material_Key, WDL_File_For_Probe> m_sub_wdl_by_material;
	WDL_File_For_Probe* m_sub_wdl_by_capture[PIECE_NB];

	DTC_Order m_max_order;
	DTC_Score m_max_conv;

	EGTB_Paths m_egtb_files;
	Temporary_File_Tracker m_tmp_files;
	bool m_save_wdl;
	bool m_save_dtc;

	EGTB_Bits m_unknown_bits[COLOR_NB];

	alignas(64) volatile DTC_Entry_Order m_entry_order;

	NODISCARD inline bool is_known(const Board_Index pos, const Color me) const
	{
		return !m_unknown_bits[me].bit_is_set(pos);
	}

	NODISCARD inline bool is_unknown(const Board_Index pos, const Color me) const
	{
		return m_unknown_bits[me].bit_is_set(pos);
	}

	template <typename EntryT>
	NODISCARD inline EntryT read_dtc(const Board_Index pos, const Color me) const
	{
		return m_dtc_file[me].read<EntryT>(pos);
	}

	template <typename EntryT>
	inline void write_dtc(const Board_Index pos, const Color me, const EntryT entry)
	{
		m_dtc_file[me].write(entry, pos);
	}

	template <typename FlagT>
	inline void lock_or_dtc(const Board_Index pos, const Color me, FlagT flag)
	{
		m_dtc_file[me].lock_add_flags(pos, flag);
	}

	template <typename FlagT>
	inline void or_dtc(const Board_Index pos, const Color me, FlagT flag)
	{
		m_dtc_file[me].add_flags(pos, flag);
	}

	template <DTC_Entry_Order ORDER>
	NODISCARD inline bool is_win(const Board_Index pos, const Color me) const
	{
		if (is_unknown(pos, me))
			return false;
		const auto entry = read_dtc<DTC_Final_Entry>(pos, me);
		return entry.is_win<ORDER>();
	}

	void open_sub_evtb();
	void close_sub_evtb();

	void save_egtb(In_Out_Param<Thread_Pool> thread_pool);

	void init_entries(In_Out_Param<Thread_Pool> thread_pool);
	void sp_init_entries(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		In_Out_Param<Concurrent_Progress_Bar> progress_bar
	);

	NODISCARD DTC_Any_Entry make_initial_entry(const Position_For_Gen& pos_gen) const;

	NODISCARD bool init_check_chase(In_Out_Param<Thread_Pool> thread_pool);
	NODISCARD bool sp_init_check_chase(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		In_Out_Param<Concurrent_Progress_Bar> progress_bar
	);

	template <DTC_Entry_Order ORDER>
	void sp_load_win_bits(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me, 
		In_Out_Param<EGTB_Bits> win_bits
	);

	void sp_load_gen_bits(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTC_Score n,
		In_Out_Param<EGTB_Bits> gen_bits
	);

	void load_win_bits(In_Out_Param<Thread_Pool> thread_pool, 
		Color me, 
		Out_Param<EGTB_Bits> win_bits
	);

	void load_gen_bits(In_Out_Param<Thread_Pool> thread_pool,
		Color me,
		DTC_Score n,
		Out_Param<EGTB_Bits> gen_bits
	);

	NODISCARD bool sp_gen_pre_bits(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const EGTB_Bits& gen_bits,
		In_Out_Param<EGTB_Bits> dst_bits
	);

	NODISCARD bool gen_pre_bits(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		const EGTB_Bits& gen_bits,
		Out_Param<EGTB_Bits> dst_bits
	);

	template <DTC_Entry_Order ORDER>
	NODISCARD bool sp_save_win(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTC_Score n,
		const EGTB_Bits& pre_bits,
		In_Out_Param<EGTB_Bits> gen_bits,
		In_Out_Param<EGTB_Bits> win_bits
	);

	NODISCARD bool save_win(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		DTC_Score n,
		const EGTB_Bits& pre_bits,
		Out_Param<EGTB_Bits> gen_bits,
		In_Out_Param<EGTB_Bits> win_bits
	);

	template <DTC_Entry_Order ORDER>
	NODISCARD bool sp_prove_lose(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTC_Score n,
		const EGTB_Bits& pre_bits,
		In_Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& win_bits
	);

	NODISCARD bool prove_lose(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		DTC_Score n,
		const EGTB_Bits& pre_bits,
		Out_Param<EGTB_Bits> gen_bits,
		const EGTB_Bits& win_bits
	);

	void build_steps(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color root_color,
		In_Out_Param<EGTB_Bits_Pool> tmp_bits
	);

	void loop_build_check_chase(
		In_Out_Param<Thread_Pool> thread_pool, 
		In_Out_Param<EGTB_Bits_Pool> tmp_bits
	);
	NODISCARD bool build_check_chase(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color root_color, 
		In_Out_Param<EGTB_Bits_Pool> tmp_bits
	);

	template <DTC_Entry_Order ORDER>
	NODISCARD EGTB_Info sp_gen_evtb(In_Out_Param<Shared_Board_Index_Iterator> gen_iterator);
	NODISCARD EGTB_Info gen_evtb(In_Out_Param<Thread_Pool> thread_pool);

	void remove_fake_check_chase(
		In_Out_Param<Thread_Pool> thread_pool, 
		In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
	);

	NODISCARD bool remove_fake(
		In_Out_Param<Thread_Pool> thread_pool, 
		DTC_Score n, 
		Remove_Fake_Step step, 
		In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
	);

	template <DTC_Entry_Order ORDER, Remove_Fake_Step TypeV>
	NODISCARD bool sp_remove_fake(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const DTC_Score n,
		In_Out_Param<EGTB_Bits> rule_bits
	);

	NODISCARD bool remove_fake_step4(
		In_Out_Param<Thread_Pool> thread_pool,
		const EGTB_Bits rule_bits[COLOR_NB]
	);

	NODISCARD bool sp_remove_fake_step4(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		const EGTB_Bits& rule_bits
	);

	template <DTC_Entry_Order ORDER, Remove_Fake_Step TypeV>
	NODISCARD DTC_Intermediate_Entry check_remove_lose(
		Position_For_Gen& pos_gen,
		DTC_Intermediate_Entry entry
	) const;

	template <DTC_Entry_Order ORDER, Remove_Fake_Step TypeV>
	NODISCARD DTC_Intermediate_Entry check_remove_win(
		Position_For_Gen& pos_gen,
		DTC_Intermediate_Entry entry
	) const;

	void sp_label_may_check_chase(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		In_Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
	);

	void label_may_check_chase(
		In_Out_Param<Thread_Pool> thread_pool, 
		Out_Param<EGTB_Bits[COLOR_NB]> rule_bits
	);

	template <DTC_Entry_Order ORDER>
	NODISCARD bool sp_label_real_check_chase(
		In_Out_Param<Shared_Board_Index_Iterator> gen_iterator,
		const Color me,
		In_Out_Param<EGTB_Bits> gen_bits
	);

	NODISCARD bool label_real_check_chase(
		In_Out_Param<Thread_Pool> thread_pool, 
		Color me,
		Out_Param<EGTB_Bits> gen_bits
	);

	NODISCARD WDL_Entry read_sub_tb(
		const Position_For_Gen& pos_gen,
		Move move
	) const;
};
