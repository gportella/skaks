#pragma once

#include "chess/board.hpp"

namespace chess {

int evaluate_board(const Board& board);
int evaluate_attacking_pieces(const Board& board);

}