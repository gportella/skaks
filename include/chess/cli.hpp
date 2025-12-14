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
};

struct CliParseResult {
  CliOptions options;
  bool show_help = false;
  bool parse_error = false;
  std::string message;
};

CliParseResult parse_cli(int argc, char** argv);

} // namespace chess
