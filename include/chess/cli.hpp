#pragma once

#include "chess/defaults.hpp"

#include <cstdint>
#include <string>

namespace chess {

struct TimeControlOptions {
  bool enabled = false;
  bool per_move = false;
  std::uint64_t move_time_ms = 0;
  std::uint64_t white_time_ms = 0;
  std::uint64_t black_time_ms = 0;
  std::uint64_t white_increment_ms = 0;
  std::uint64_t black_increment_ms = 0;
  std::uint32_t moves_to_go = 0;
};

struct CliOptions {
  int search_depth = 4;
  int max_full_moves = kMaxMovementCount;
  std::string fen;
  bool use_custom_fen = false;
  bool enable_profile = false;
  bool use_uci = true;
  bool self_play = false;
  bool only_fen = false;
  bool polyglot = false;
  std::string polyglot_book_path;
  bool polyglot_book_override = false;
  bool params_override = false;
  std::string params_path;
  bool perf_mode = false;
  int perf_iterations = 3;
  std::string perf_suite;
  std::string perf_suite_file;
  int version_requests = 0;
  bool show_version = false;
  bool show_extended_version = false;
  int thread_count = 0; // 0 = auto
  bool static_eval = false;
  std::string executable_path;
  bool best_move = false;
  bool arena_mode = false;
  int arena_games = 20;
  TimeControlOptions time_control;
};

struct CliParseResult {
  CliOptions options;
  bool show_help = false;
  bool parse_error = false;
  std::string message;
};

CliParseResult parse_cli(int argc, char** argv);

} // namespace chess
