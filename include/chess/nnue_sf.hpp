#pragma once
#include "chess/board.hpp"

namespace chess {
int evaluate_nnue_stockfish(const Board& board);
} // namespace chess