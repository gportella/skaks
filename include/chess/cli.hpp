#pragma once

#include "chess/defaults.hpp"

#include <string>

namespace chess {

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
  bool perf_mode = false;
  int perf_iterations = 3;
  int version_requests = 0;
  bool show_version = false;
  bool show_extended_version = false;
  std::string executable_path;
  bool best_move = false;
};

struct CliParseResult {
  CliOptions options;
  bool show_help = false;
  bool parse_error = false;
  std::string message;
};

CliParseResult parse_cli(int argc, char** argv);

} // namespace chess
