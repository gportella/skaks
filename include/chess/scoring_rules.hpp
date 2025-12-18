#pragma once

#include "chess/board.hpp"

namespace chess {

constexpr int KNIGHT_DEV_BONUS = 12;
constexpr int BISHOP_DEV_BONUS = 10;
constexpr int CONNECT_ROOKS_BONUS = 10;
constexpr int CENTRAL_PAWN_BONUS = 12;
constexpr int CASTLE_URGENCY = 20;
constexpr int EARLY_QUEEN_PENALTY = 16;
constexpr int FLANK_PAWN_PENALTY = 8;
int evaluate_board(const Board& board);
int evaluate_attacking_pieces(const Board& board);

} // namespace chess