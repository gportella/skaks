#pragma once

#include "chess/board.hpp"
#include "chess/types.hpp"

namespace chess {

struct ComplexityMetrics {
  double mobility = 0.0;
  double capture_ratio = 0.0;
  double checking_ratio = 0.0;
  double tension = 0.0;
  double value = 0.0;
};

ComplexityMetrics compute_complexity(const Board& board, SideToMove stm);

double normalize_complexity(const ComplexityMetrics& metrics);

} // namespace chess
