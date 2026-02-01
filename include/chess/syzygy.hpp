#pragma once

#include "chess/board.hpp"
#include "chess/moves.hpp"

#include <optional>
#include <string_view>

namespace chess::syzygy {

struct ProbeResult {
  bool available = false;
  int wdl = 0;
  int score = 0;
  std::optional<Move> best_move;
};

bool init(std::string_view path);
void free();
bool available();
int max_pieces();
std::string_view path();
/**
 * @brief Probe Syzygy WDL tables for non-root search usage.
 *
 * Uses the thread-safe WDL probe and returns a score suitable for search
 * cutoffs when the position is tablebase-eligible. The probe does not
 * provide a best move.
 */
ProbeResult probe_wdl(const Board& board);
ProbeResult probe_root_wdl(const Board& board);
ProbeResult probe_root_dtz(const Board& board);
void init_from_env();

} // namespace chess::syzygy
