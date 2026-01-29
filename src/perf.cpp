
#include "chess/perf.hpp"

#include "chess/board.hpp"
#include "chess/cli.hpp"
#include "chess/defaults.hpp"
#include "chess/engine.hpp"
#include "chess/search_stats.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace chess {
namespace {

struct PerfCase {
  std::string name;
  std::string fen;
};

SearchLimits to_search_limits(const TimeControlOptions& options) {
  SearchLimits limits{};
  if (!options.enabled) {
    return limits;
  }
  limits.use_time = true;
  limits.per_move = options.per_move;
  limits.move_time_ms = options.move_time_ms;
  limits.white_time_ms = options.white_time_ms;
  limits.black_time_ms = options.black_time_ms;
  limits.white_increment_ms = options.white_increment_ms;
  limits.black_increment_ms = options.black_increment_ms;
  limits.moves_to_go = options.moves_to_go;
  return limits;
}

void trim_in_place(std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    value.clear();
    return;
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  value = value.substr(first, last - first + 1);
}

PerfCase parse_suite_entry(std::string line, int line_number,
                           std::string& error) {
  trim_in_place(line);
  if (line.empty() || line.front() == '#') {
    return PerfCase{};
  }

  std::string name;
  std::string fen;
  const auto pipe_pos = line.find('|');
  if (pipe_pos != std::string::npos) {
    name = line.substr(0, pipe_pos);
    trim_in_place(name);
    fen = line.substr(pipe_pos + 1);
    trim_in_place(fen);
  } else {
    name = "case" + std::to_string(line_number);
    fen = std::move(line);
  }

  if (fen.empty()) {
    error = "Empty FEN in perf suite at line " + std::to_string(line_number);
    return PerfCase{};
  }

  return PerfCase{std::move(name), std::move(fen)};
}

bool load_suite_file(const std::string& path, std::vector<PerfCase>& out,
                     std::string& error) {
  std::ifstream file(path);
  if (!file) {
    error = "Unable to open perf suite file: " + path;
    return false;
  }

  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    auto entry = parse_suite_entry(line, line_number, error);
    if (!error.empty()) {
      return false;
    }
    if (!entry.fen.empty()) {
      out.push_back(std::move(entry));
    }
  }

  if (out.empty()) {
    error = "Perf suite file contains no FEN entries: " + path;
    return false;
  }

  return true;
}

std::vector<PerfCase> builtin_bench_suite() {
  return std::vector<PerfCase>{
      {"startpos", std::string(kStartFEN)},
      {"balanced_midgame",
       "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 0 9"},
      {"tactical_midgame",
       "r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P1B2/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 0 8"},
      {"queenside_attack", "3r1rk1/1bqn1pbp/pp1p1np1/2pPp3/P1P1P3/2N1BN1P/"
                           "1P1QBPP1/2R2RK1 w - - 0 14"},
      {"imbalanced_endgame",
       "2r1r1k1/1b1n1pbp/p2p2p1/1ppPp3/4P3/1P1B1N1P/PB1N1PP1/2RR2K1 w - - 0 23"},
      {"simplified_endgame", "8/5k2/3p4/4p3/4P3/3P4/5K2/8 w - - 0 1"}};
}

std::vector<PerfCase> resolve_perf_suite(const CliOptions& options,
                                         std::string& label,
                                         std::string& error) {
  if (options.perf_suite == "file") {
    std::vector<PerfCase> cases;
    if (!load_suite_file(options.perf_suite_file, cases, error)) {
      return {};
    }
    label = "file";
    return cases;
  }

  if (options.perf_suite == "bench" || options.perf_suite == "search") {
    label = "bench";
    return builtin_bench_suite();
  }

  PerfCase single_case;
  if (options.use_custom_fen) {
    single_case = PerfCase{"custom", options.fen};
    label = "custom";
  } else {
    single_case = PerfCase{"startpos", std::string(kStartFEN)};
    label = "startpos";
  }
  return std::vector<PerfCase>{std::move(single_case)};
}

} // namespace

int run_perf_mode(Engine& engine, const chess::CliOptions& options) {
  std::string suite_label;
  std::string suite_error;
  auto suites = resolve_perf_suite(options, suite_label, suite_error);
  if (suites.empty()) {
    std::cerr << "Perf suite error: " << suite_error << "\n";
    return EXIT_FAILURE;
  }

  SearchParameters base_params{};
  base_params.alpha = -INF;
  base_params.beta = INF;

  const bool use_time = options.time_control.enabled;
  if (use_time) {
    base_params.depth = static_cast<int>(MAX_PLY) - 1;
    base_params.limits = to_search_limits(options.time_control);
  } else {
    base_params.depth = options.search_depth;
  }

  const int iterations = std::max(options.perf_iterations, 1);

  std::cout << "[perf] suite=" << suite_label << " cases=" << suites.size()
            << " iterations=" << iterations
            << " threads=" << engine.thread_count();
  if (use_time) {
    if (options.time_control.per_move) {
      std::cout << " move_time_ms=" << options.time_control.move_time_ms;
    } else {
      std::cout << " wtime=" << options.time_control.white_time_ms
                << " btime=" << options.time_control.black_time_ms
                << " winc=" << options.time_control.white_increment_ms
                << " binc=" << options.time_control.black_increment_ms
                << " moves_to_go=" << options.time_control.moves_to_go;
    }
  } else {
    std::cout << " depth=" << base_params.depth;
  }
  std::cout << "\n";

  std::uint64_t suite_nodes = 0;
  std::uint64_t suite_ms = 0;

  for (const auto& perf_case : suites) {
    std::cout << "[perf] case=" << perf_case.name << " fen=\"" << perf_case.fen
              << "\"\n";

    std::uint64_t case_nodes = 0;
    std::uint64_t case_ms = 0;

    for (int iter = 0; iter < iterations; ++iter) {
      Board board{};
      try {
        board = initial_board(perf_case.fen);
      } catch (const std::exception& ex) {
        std::cerr << "Failed to load FEN for case '" << perf_case.name
                  << "': " << ex.what() << "\n";
        return EXIT_FAILURE;
      }

      engine.reset_history(board);
      engine.clear_transposition_table();
      if (search_stats_enabled()) {
        reset_search_stats();
      }

      auto session = engine.create_session(board);
      SearchParameters params = base_params;
      auto result = session.run(params);

      case_nodes += result.nodes;
      case_ms += result.elapsed_ms;

      const auto iter_ms = result.elapsed_ms;
      const auto iter_nps =
          (iter_ms == 0) ? 0 : (result.nodes * 1000ULL) / iter_ms;

      std::cout << "[perf] case=" << perf_case.name << " iter=" << (iter + 1)
                << "/" << iterations << " nodes=" << result.nodes
                << " elapsed_ms=" << iter_ms << " nps=" << iter_nps << "\n";
    }

    const auto clamped_ms = (case_ms == 0) ? 1 : case_ms;
    const auto total_nps =
        (case_ms == 0) ? 0 : (case_nodes * 1000ULL) / clamped_ms;
    const auto avg_nodes = case_nodes / static_cast<std::uint64_t>(iterations);
    const auto avg_ms = clamped_ms / static_cast<std::uint64_t>(iterations);

    std::cout << "[perf] case=" << perf_case.name
              << " total_nodes=" << case_nodes << " total_ms=" << case_ms
              << " total_nps=" << total_nps << " avg_nodes=" << avg_nodes
              << " avg_ms=" << avg_ms << "\n";

    suite_nodes += case_nodes;
    suite_ms += case_ms;
  }

  const auto suite_clamped_ms = (suite_ms == 0) ? 1 : suite_ms;
  const auto suite_total_nps =
      (suite_ms == 0) ? 0 : (suite_nodes * 1000ULL) / suite_clamped_ms;

  std::cout << "[perf] suite=" << suite_label << " total_nodes=" << suite_nodes
            << " total_ms=" << suite_ms << " total_nps=" << suite_total_nps
            << "\n";

  return EXIT_SUCCESS;
}

} // namespace chess
