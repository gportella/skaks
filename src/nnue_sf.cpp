#include "chess/nnue_sf.hpp"

#include "chess/board.hpp"
#include "probe.h"

#include <algorithm>
#include <array>

namespace chess {

// namespace {

int to_sf_piece(OccupancyType occ) {
  switch (occ) {
  case OccupancyType::wP:
    return 1;
  case OccupancyType::wN:
    return 2;
  case OccupancyType::wB:
    return 3;
  case OccupancyType::wR:
    return 4;
  case OccupancyType::wQ:
    return 5;
  case OccupancyType::wK:
    return 6;
  case OccupancyType::bP:
    return 9;
  case OccupancyType::bN:
    return 10;
  case OccupancyType::bB:
    return 11;
  case OccupancyType::bR:
    return 12;
  case OccupancyType::bQ:
    return 13;
  case OccupancyType::bK:
    return 14;
  case OccupancyType::empty:
    return 0;
  }
  return 0;
}

// } // namespace chess

int evaluate_nnue_stockfish(const Board& board) {
  std::array<int, 64> piece_board{};
  for (std::size_t sq = 0; sq < piece_board.size(); ++sq) {
    piece_board[sq] = to_sf_piece(board.pieces[sq]);
  }

  const bool side = board.side_to_move == SideToMove::White;
  int rule50 = board.fifty_move_counter;
  if (rule50 < 0) {
    rule50 = 0;
  } else if (rule50 > 100) {
    rule50 = 100;
  }
  int score = Stockfish::Probe::eval(piece_board.data(), side, rule50);
  if (board.side_to_move == SideToMove::Black) {
    score = -score;
  }
  return score;
}

} // namespace chess