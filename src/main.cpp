#include "chess/chess.h"
#include "chess/attack.h"
#include "chess/piece_config.h"

#include "util/utility.h"
#include "util/algo.h"
#include "util/endian.h"
#include "util/compress.h"

#include "egtb/egtb_gen_wdl_dtc.h"
#include "egtb/egtb_gen_dtm.h"

#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <optional>
#include <filesystem>
#include <functional>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace std::string_view_literals;

static const std::filesystem::path ADDITIONAL_OPTIONS_FILE_PATH = "./option.ini";

constexpr size_t kiB = 1024;
constexpr size_t MiB = 1024 * kiB;
constexpr size_t GiB = 1024 * MiB;

struct Program_Options
{
	Program_Options(const std::filesystem::path& path);

	EGTB_Paths egtb_files;

	bool save_wdl = true;
	bool save_dtc = true;
	bool save_dtm = true;
	bool save_rule_bits = false;

	size_t num_threads = 1;
	size_t max_pieces = 20;
	size_t memory_size = GiB;

	bool generate_run_list = true;
	bool generate_tablebases = true;

	std::filesystem::path egtb_gen_list_file_path = "autoList.txt";
	std::filesystem::path egtb_gen_info_file_path = "egtb_gen_info.csv";
	std::filesystem::path egtb_full_gen_info_file_path = "egtb_full_gen_info.csv";
};

struct Gen_List_Candidate
{
	explicit Gen_List_Candidate(const Piece_Config& ps) :
		piece_set(ps),
		wdl_info(DTC_Generator::wdl_generation_info(ps)),
		dtc_info(DTC_Generator::dtc_generation_info(ps)),
		dtm_info(DTM_Generator::dtm_generation_info(ps))
	{
	}

	NODISCARD bool is_too_large() const
	{
		return !wdl_info.has_value() || !dtc_info.has_value() || !dtm_info.has_value();
	}

	NODISCARD bool requires_more_memory_than(size_t memory) const
	{
		return 
			   !wdl_info.has_value() || wdl_info->memory_required_for_generation > memory 
			|| !dtc_info.has_value() || dtc_info->memory_required_for_generation > memory
			|| !dtm_info.has_value() || dtm_info->memory_required_for_generation > memory;
	}

	NODISCARD inline friend bool operator<(const Gen_List_Candidate& lhs, const Gen_List_Candidate& rhs) noexcept
	{
		if (lhs.wdl_info.has_value() && !rhs.wdl_info.has_value())
			return true;

		if (rhs.wdl_info.has_value() && !lhs.wdl_info.has_value())
			return false;

		if (   lhs.wdl_info.has_value()
			&& rhs.wdl_info.has_value()
			&& lhs.wdl_info->num_positions != rhs.wdl_info->num_positions)
			return lhs.wdl_info->num_positions < rhs.wdl_info->num_positions;

		if (lhs.piece_set.num_pieces() != rhs.piece_set.num_pieces())
			return lhs.piece_set.num_pieces() < rhs.piece_set.num_pieces();

		return lhs.piece_set.name() < rhs.piece_set.name();
	}

	Piece_Config piece_set;
	std::optional<EGTB_Generation_Info> wdl_info;
	std::optional<EGTB_Generation_Info> dtc_info;
	std::optional<EGTB_Generation_Info> dtm_info;
};

struct Gen_List_Entry : Gen_List_Candidate
{
	explicit Gen_List_Entry(const Piece_Config& ps, const Program_Options& options) :
		Gen_List_Candidate(ps)
	{
		fill_generation_needs(options);
	}

	explicit Gen_List_Entry(const Gen_List_Candidate& entry, const Program_Options& options) :
		Gen_List_Candidate(entry)
	{
		fill_generation_needs(options);
	}

	NODISCARD bool needs_any_generation() const
	{
		return generate_wdl || generate_dtc || generate_dtm;
	}

	NODISCARD size_t required_memory() const
	{
		size_t m = 0;
		if (generate_wdl && wdl_info.has_value())
			update_max(m, wdl_info->memory_required_for_generation);
		if (generate_dtc && dtc_info.has_value())
			update_max(m, dtc_info->memory_required_for_generation);
		if (generate_dtm && dtm_info.has_value())
			update_max(m, dtm_info->memory_required_for_generation);
		return m;
	}

	bool generate_wdl;
	bool generate_dtc;
	bool generate_dtm;

private:
	void fill_generation_needs(const Program_Options& options)
	{
		generate_dtm = options.save_dtm && !options.egtb_files.find_dtm_file(piece_set);
		generate_dtc = options.save_dtc && !options.egtb_files.find_dtc_file(piece_set);
		generate_wdl = (generate_dtm || options.save_wdl) && !options.egtb_files.find_wdl_file(piece_set);
	}
};

void gen_tablebases(const std::vector<Gen_List_Entry>& gen_list, const Program_Options& options);

using PieceFilterFunc = std::function<bool(Const_Span<size_t>)>;
NODISCARD std::vector<Gen_List_Candidate> gen_man_piece_sets(size_t max_man_cnt, PieceFilterFunc filter = nullptr);

NODISCARD std::vector<Gen_List_Entry> make_gen_list(const Unique_Piece_Configs& piece_sets, const Program_Options& options);

void save_gen_info(const std::vector<Gen_List_Candidate>& infos, std::filesystem::path path);
void save_gen_list(const std::vector<Gen_List_Candidate>& list, std::filesystem::path path);

Unique_Piece_Configs read_gen_list(std::filesystem::path path);

NODISCARD bool pieces_filter(Const_Span<size_t> counts);

int main(int argc, char* argv[])
{
	if (!is_little_endian())
	{
		std::cerr << "Byte orderings other than little-endian are not supported. Exiting.\n";
		return 1;
	}

	init_possible();
	attack_init();

	const std::vector<std::string> args(argv + 1, argv + argc);
	const Program_Options options(ADDITIONAL_OPTIONS_FILE_PATH);

	if (args.size() >= 1 && args[0] == "compute_egtb_gen_info")
	{
		std::cout << "Gathering all piece configurations...\n";
		const auto& list = gen_man_piece_sets(MAX_MAN);
		std::cout << "Gathered total of " << list.size() << " piece configurations. Saving info...\n";
		save_gen_info(list, options.egtb_full_gen_info_file_path);
		std::cout << "Info saved to " << options.egtb_full_gen_info_file_path << '\n';
		return 0;
	}

	options.egtb_files.init_directories();

	if (options.generate_run_list)
	{
		std::cout << "Gathering configurations with <=" << options.max_pieces << " pieces...\n";
		const auto& list = gen_man_piece_sets(options.max_pieces, pieces_filter);
		std::cout << "Gathered total of " << list.size() << " candidate piece configurations. Saving...\n";
		save_gen_info(list, options.egtb_gen_info_file_path);
		std::cout << "Info saved to " << options.egtb_gen_info_file_path << '\n';
		save_gen_list(list, options.egtb_gen_list_file_path);
		std::cout << "List saved to " << options.egtb_gen_list_file_path << '\n';

	}

	if (options.generate_tablebases)
	{
		std::cout << "Reading desired piece configurations from " << options.egtb_gen_list_file_path << "...\n";
		const auto& base_list = read_gen_list(options.egtb_gen_list_file_path);
		std::cout << "Finished reading resired piece configurations.\n";

		std::cout << "Preparing tablebase generation list.\n";
		const auto& gen_list = make_gen_list(base_list, options);
		std::cout << "Finished preparing tablebase generation list.\n";

		if (gen_list.empty())
			std::cout << "Nothing to do.\n";
		else
		{
			std::cout << gen_list.size() << " piece configurations will be processed further:\n";
			printf("--%s--------------------\n", std::string(options.max_pieces, '-').c_str());
			printf("| %s | WDL | DTC | DTM | Mem\n", std::string(options.max_pieces, ' ').c_str());
			for (const auto& entry : gen_list)
			{
				printf("| %*s |  %c  |  %c  |  %c  | %zuMiB\n",
					static_cast<int>(options.max_pieces), entry.piece_set.name().c_str(),
					entry.generate_wdl ? '+' : ' ',
					entry.generate_dtc ? '+' : ' ',
					entry.generate_dtm ? '+' : ' ',
					entry.required_memory() / MiB
				);
			}
			printf("--%s--------------------\n", std::string(options.max_pieces, '-').c_str());
		}

		gen_tablebases(gen_list, options);
	}
}

void gen_tablebases(const std::vector<Gen_List_Entry>& gen_list, const Program_Options& options)
{
	auto start_time = std::chrono::steady_clock::now();

	Thread_Pool thread_pool(options.num_threads);

	size_t current_processed = 0;
	for (const auto& entry : gen_list)
	{
		current_processed += 1;
		std::cout << "Processing piece configuration " << current_processed << " out of " << gen_list.size() << ": " << entry.piece_set.name() << "\n";
		std::cout << "=====================\n";

		if (entry.generate_wdl || entry.generate_dtc)
		{
			try
			{
				const auto start_time = std::chrono::steady_clock::now();
				DTC_Generator input(entry.piece_set, entry.generate_wdl, entry.generate_dtc, options.egtb_files);
				input.gen(inout_param(thread_pool));
				const auto end_time = std::chrono::steady_clock::now();
				printf("WDL/DTC generation took %s\n", format_elapsed_time(start_time, end_time).c_str());
			}
			catch (std::runtime_error& e)
			{
				std::cout << "Error during generation of " << entry.piece_set.name() << " WDL/DTC TB: " << e.what() << '\n';
				throw;
			}
		}

		if (entry.generate_dtm)
		{
			try
			{
				const auto start_time = std::chrono::steady_clock::now();
				DTM_Generator input(entry.piece_set, options.save_rule_bits, options.egtb_files);
				input.gen(inout_param(thread_pool));
				const auto end_time = std::chrono::steady_clock::now();
				printf("DTM generation took %s\n", format_elapsed_time(start_time, end_time).c_str());
			}
			catch (std::runtime_error& e)
			{
				std::cout << "Error during generation of " << entry.piece_set.name() << " DTM TB: " << e.what() << '\n';
				throw;
			}
		}

		printf("=====================\n");
	}

	auto end_time = std::chrono::steady_clock::now();
	printf("Generating tablebases finished in %s\n", format_elapsed_time(start_time, end_time).c_str());
}

NODISCARD Unique_Piece_Configs read_gen_list(std::filesystem::path path)
{
	std::ifstream fp_list(path);
	if (!fp_list)
		throw std::runtime_error("无法打开生成列表\n");

	Unique_Piece_Configs piece_sets;
	for (std::string name; std::getline(fp_list, name);)
	{
		if (!Piece_Config::is_constructible_from(name))
		{
			std::cout << "ERROR: Omitting " << name << " generation. Not a valid piece configuration.\n";
			continue;
		}

		piece_sets.add_unique(Piece_Config(name));
	}

	return piece_sets;
}

NODISCARD std::vector<Gen_List_Entry> make_gen_list(const Unique_Piece_Configs& piece_sets, const Program_Options& options)
{
	Unique_Piece_Configs closured_piece_sets;

	for (const Piece_Config& ps : piece_sets)
		ps.add_closure_in_dependency_order_to(closured_piece_sets, true);

	const size_t safe_amount_of_memory_bytes = (options.memory_size * MiB) * 4 / 5;

	std::vector<Gen_List_Entry> gen_list;
	for (const Piece_Config& ps : closured_piece_sets)
	{
		if (!ps.has_any_free_attackers())
		{
			std::cout << "INFO: Omitting " << ps.name() << " generation. No free attackers.\n";
			continue;
		}

		const Gen_List_Candidate candidate(ps);
		if (candidate.requires_more_memory_than(safe_amount_of_memory_bytes))
		{
			std::cout << "WARN: Omitting " << ps.name() << " generation. Size exceeds available memory.\n";
			continue;
		}

		const Gen_List_Entry entry(candidate, options);
		if (!entry.needs_any_generation())
		{
			std::cout << "INFO: Omitting " << ps.name() << " generation. All required files already exist.\n";
			continue;
		}

		gen_list.emplace_back(entry);
	}
	
	return gen_list;
}

NODISCARD bool parse_line(const std::string& line, std::string& name, std::string& value);

Program_Options::Program_Options(const std::filesystem::path& path)
{
	std::ifstream fp(path);
	if (fp)
	{
		std::string name, value;
		for (std::string line; std::getline(fp, line);)
		{
			if (!parse_line(line, name, value))
				continue;

			if (name == "dtmtb"sv)
			{
				egtb_files.add_dtm_path(value);
			}
			else if (name == "dtctb"sv)
			{
				egtb_files.add_dtc_path(value);
			}
			else if (name == "wdltb"sv)
			{
				egtb_files.add_wdl_path(value);
			}
			else if (name == "tmpdir"sv)
			{
				egtb_files.set_tmp_path(value);
			}
			else if (name == "MaxMem"sv)
			{
				memory_size = atoi(value.c_str());
			}
			else if (name == "GenerateRunList"sv)
			{
				generate_run_list = atoi(value.c_str());
			}
			else if (name == "GenerateTablebases"sv)
			{
				generate_tablebases = atoi(value.c_str());
			}
			else if (name == "SaveWDL"sv)
			{
				save_wdl = atoi(value.c_str());
			}
			else if (name == "SaveDTC"sv)
			{
				save_dtc = atoi(value.c_str());
			}
			else if (name == "SaveDTM"sv)
			{
				save_dtm = atoi(value.c_str());
			}
			else if (name == "SaveRuleBits"sv)
			{
				save_rule_bits = atoi(value.c_str());
			}
			else if (name == "Threads"sv)
			{
				num_threads = atoi(value.c_str());
			}
			else if (name == "MaxPieces"sv)
			{
				max_pieces = atoi(value.c_str());
			}
		}
	}
}

NODISCARD bool pieces_filter(Const_Span<size_t> counts)
{
	size_t att[COLOR_NB] = { 0, 0 };

	for (const Piece piece : ALL_FREE_ATTACKING_PIECES)
		att[piece_color(piece)] += counts[piece];

	if (att[WHITE] + att[BLACK] == 0)
		return false;

	if (counts[WHITE_PAWN] >= 4 || counts[BLACK_PAWN] >= 4)
		return false;

	if (att[WHITE] + att[BLACK] > 8)
		return false;
	
	if (att[WHITE] >= 5 || att[BLACK] >= 5)
		return false;

	const size_t major[2] = {
		counts[WHITE_ROOK] + counts[WHITE_KNIGHT] + counts[WHITE_CANNON], 
		counts[BLACK_ROOK] + counts[BLACK_KNIGHT] + counts[BLACK_CANNON] 
	};

	if (major[WHITE] + major[BLACK] > 6)
		return false;

	if (major[WHITE] >= 3 || major[BLACK] >= 3)
		return false;

	return true;
}

NODISCARD std::array<std::pair<size_t, size_t>, PIECE_NB> supported_piece_count_ranges()
{
	// Max pawn count depends on the size of the index type.
	// For >3 pawns there are more than 2**16 configurations
	// for them on the board.
	// With 5 pawns there are 2,125,764 piece configurations to be checked.
	constexpr size_t MAX_PAWNS = sizeof(Piece_Group::Placement_Index) > 2 ? 5 : 3;

	std::array<std::pair<size_t, size_t>, PIECE_NB> piece_count_ranges;

	piece_count_ranges[WHITE_OCCUPY] = piece_count_ranges[BLACK_OCCUPY] = { 0, 0 };
	piece_count_ranges[WHITE_KING] = piece_count_ranges[BLACK_KING] = { 1, 1 };
	piece_count_ranges[WHITE_ROOK] = piece_count_ranges[BLACK_ROOK] = { 0, 2 };
	piece_count_ranges[WHITE_KNIGHT] = piece_count_ranges[BLACK_KNIGHT] = { 0, 2 };
	piece_count_ranges[WHITE_CANNON] = piece_count_ranges[BLACK_CANNON] = { 0, 2 };
	piece_count_ranges[WHITE_ADVISOR] = piece_count_ranges[BLACK_ADVISOR] = { 0, 2 };
	piece_count_ranges[WHITE_BISHOP] = piece_count_ranges[BLACK_BISHOP] = { 0, 2 };
	piece_count_ranges[WHITE_PAWN] = piece_count_ranges[BLACK_PAWN] = { 0, MAX_PAWNS };

	return piece_count_ranges;
}

NODISCARD std::vector<Gen_List_Candidate> gen_man_piece_sets(size_t max_man_cnt, PieceFilterFunc filter)
{
	const auto piece_count_ranges = supported_piece_count_ranges();

	Unique_Piece_Configs piece_sets;

	for (const auto& piece_counts : Mixed_Radix<size_t>::with_inclusive_ranges(Const_Span(piece_count_ranges)))
	{
		if (std::accumulate(piece_counts.begin(), piece_counts.end(), static_cast<size_t>(0)) > max_man_cnt)
			continue;

		if (filter && !filter(Const_Span(piece_counts)))
			continue;

		Fixed_Vector<Piece, MAX_MAN> pieces;

		for (const Piece pc : ALL_PIECES)
			for (size_t i = 0; i < piece_counts[pc]; ++i)
				pieces.emplace_back(pc);

		piece_sets.add_unique(Piece_Config(Const_Span(pieces)));
	}

	std::vector<Gen_List_Candidate> res(piece_sets.begin(), piece_sets.end());
	std::sort(res.begin(), res.end());
	return res;
}

void save_gen_list(const std::vector<Gen_List_Candidate>& infos, std::filesystem::path path)
{
	std::ofstream fp(path);
	for (const auto& entry : infos)
		fp << entry.piece_set.name() << '\n';
}

bool parse_line(const std::string& line, std::string& name, std::string& value)
{
	// remove comments
	auto end = line.find_first_of(";#");
	if (end == std::string::npos)
		end = line.size();

	// split at '='
	const auto eq_sign = line.find_first_of('=');
	if (eq_sign == std::string::npos)
		return false;

	// cleanup
	name = strip(line.substr(0, eq_sign));
	value = strip(line.substr(eq_sign + 1));

	return !name.empty() && !value.empty();
}

void save_gen_info(const std::vector<Gen_List_Candidate>& infos, std::filesystem::path path)
{
	std::ofstream out_file(path);
	out_file << "Piece configuration;Num positions;WDL uncompressed size;DTC uncompressed size;DTM uncompressed size;WDL generation memory;DTC generation memory;DTM generation memory;WDL sub EGTB size;DTC sub EGTB size;DTM sub EGTB size\n";
	for (size_t i = 0; i < infos.size(); ++i)
	{
		const auto& entry = infos[i];
		char buf[1024];
		if (entry.is_too_large())
		{
			std::snprintf(buf, sizeof(buf),
				"%-32s;TOO LARGE;TOO LARGE;TOO LARGE;TOO LARGE;TOO LARGE;TOO LARGE;TOO LARGE;TOO LARGE;TOO LARGE;TOO LARGE\n",
				entry.piece_set.name().c_str()
			);
		}
		else
		{
			ASSERT(entry.wdl_info.has_value());
			ASSERT(entry.dtc_info.has_value());
			ASSERT(entry.dtm_info.has_value());

			std::snprintf(buf, sizeof(buf),
				"%-32s;%016zu;%010zuMiB;%010zuMiB;%010zuMiB;%010zuMiB;%010zuMiB;%010zuMiB;%010zuMiB;%010zuMiB;%010zuMiB\n",
				entry.piece_set.name().c_str(), entry.wdl_info->num_positions,
				entry.wdl_info->uncompressed_size / MiB, entry.dtc_info->uncompressed_size / MiB, entry.dtm_info->uncompressed_size / MiB,
				entry.wdl_info->memory_required_for_generation / MiB, entry.dtc_info->memory_required_for_generation / MiB, entry.dtm_info->memory_required_for_generation / MiB,
				entry.wdl_info->uncompressed_sub_tb_size / MiB, entry.dtc_info->uncompressed_sub_tb_size / MiB, entry.dtm_info->uncompressed_sub_tb_size / MiB
			);
		}
		out_file << std::string(buf);
		if ((i + 1) % 10000 == 0 || (i + 1) == infos.size())
			std::cout << "Saved " << (i + 1) << " out of " << infos.size() << " egtb generation infos.\n";
	}
}