#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/pins.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

namespace chess {
Board parse_fen_string(std::string_view fen);
} // namespace chess

namespace {

chess::Board make_board(std::string_view fen) {
  chess::Board board = chess::parse_fen_string(fen);

  board.occupancy[static_cast<std::size_t>(chess::PieceColor::White)] =
      chess::calculate_occupancy(board, chess::PieceColor::White);
  board.occupancy[static_cast<std::size_t>(chess::PieceColor::Black)] =
      chess::calculate_occupancy(board, chess::PieceColor::Black);
  board.occupancy[static_cast<std::size_t>(chess::PieceColor::Both)] =
      chess::calculate_occupancy(board, chess::PieceColor::Both);

  board.pieces_bb.fill(0);
  for (std::size_t sq = 0; sq < 64; ++sq) {
    const chess::OccupancyType occ = board.pieces[sq];
    if (occ == chess::OccupancyType::empty) {
      continue;
    }
    const std::size_t piece_idx = static_cast<std::size_t>(occ) - 1;
    board.pieces_bb[piece_idx] |= (Bitboard(1) << sq);
  }

  return board;
}

} // namespace

TEST(KingInCheckTest, DetectsCheck) {
  const chess::Board board =
      make_board("rnbqk1nr/ppppp1pp/5p2/8/7b/8/PPPP2PP/RNBQKBNR w KQkq - 0 1");

  EXPECT_TRUE(chess::is_check(board, chess::SideToMove::White));
}

TEST(KingNotInCheckTest, DetectsNoCheck) {
  const chess::Board board =
      make_board("rnbqk1nr/ppppp1pp/5p2/8/7b/6P1/PPPP3P/RNBQKBNR w KQkq - 0 1");

  EXPECT_FALSE(chess::is_check(board, chess::SideToMove::White));
}
