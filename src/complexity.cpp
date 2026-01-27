// SPDX-License-Identifier: GPL-3.0-or-later
#include "chess/complexity.hpp"

#include "chess/moves.hpp"

#include <algorithm>
#include <cmath>

namespace chess {
namespace {

inline double safe_ratio(double numerator, double denominator) {
  if (denominator <= 0.0) {
    return 0.0;
  }
  return numerator / denominator;
}

} // namespace

ComplexityMetrics compute_complexity(const Board& board, SideToMove stm) {
  Board working = board;
  uint16_t move_count = 0;
  auto moves = generate_legal_moves(working, stm, move_count);

  if (move_count == 0) {
    return ComplexityMetrics{};
  }

  double capture_moves = 0.0;
  double checking_moves = 0.0;
  double quiet_moves = 0.0;
  double unique_targets = 0.0;

  Board analysis = board;

  bool target_map[64] = {false};

  for (uint16_t idx = 0; idx < move_count; ++idx) {
    const Move move = decode_move(moves[idx]);

    if (move.captured_pc != OccupancyType::empty) {
      capture_moves += 1.0;
    }

    target_map[static_cast<std::size_t>(move.to)] = true;

    const Undo undo = make_move(analysis, move);
    const bool gives_check = is_check(analysis, flip_side(stm));
    if (gives_check) {
      checking_moves += 1.0;
    }
    if (move.captured_pc == OccupancyType::empty && !gives_check) {
      quiet_moves += 1.0;
    }
    undo_move(analysis, undo);
  }

  for (bool target : target_map) {
    if (target) {
      unique_targets += 1.0;
    }
  }

  ComplexityMetrics metrics;
  metrics.mobility = std::log1p(static_cast<double>(move_count));
  metrics.capture_ratio =
      safe_ratio(capture_moves, static_cast<double>(move_count));
  metrics.checking_ratio =
      safe_ratio(checking_moves, static_cast<double>(move_count));
  metrics.tension = std::log1p(quiet_moves + capture_moves + checking_moves);

  const double target_density =
      safe_ratio(unique_targets, static_cast<double>(move_count));
  metrics.value = metrics.mobility + 0.6 * metrics.tension +
                  0.4 * metrics.checking_ratio + 0.5 * metrics.capture_ratio +
                  0.3 * target_density;

  return metrics;
}

double normalize_complexity(const ComplexityMetrics& metrics) {
  constexpr double kReference = 6.5;
  const double normalized = metrics.value / kReference;
  return std::clamp(normalized, 0.0, 1.5);
}

} // namespace chess
