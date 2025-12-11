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
}

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

TEST(BishopPinTest, BlocksBishopWhenPinned) {
  const chess::Board board =
      make_board("rnbqk1nr/ppp4p/6p1/3ppp2/1b1PP3/7N/PPPB1PPP/RN1QKB1R w KQkq - 2 6");

  const auto pins = chess::build_pinned_map(board, chess::SideToMove::White);
  const auto [d2_pinned_mask, dir] = pins.bishop_pins[chess::to_index(chess::Square::D2)];
  const auto [f1_pinned_mask, f1_dir] = pins.bishop_pins[chess::to_index(chess::Square::F1)];

  // The bishop on d2 is pinned to the king on e1 by the black bishop on b4
  EXPECT_TRUE(d2_pinned_mask != 0);
  EXPECT_EQ(d2_pinned_mask, (Bitboard(1) << chess::to_index(chess::Square::D2) |
                             Bitboard(1) << chess::to_index(chess::Square::C3) |
                             Bitboard(1) << chess::to_index(chess::Square::B4)));
  EXPECT_TRUE(f1_pinned_mask == 0);
}

TEST(KnightPinTest, KnightCannotMove) {
  const chess::Board board =
      make_board("rnb1k1nr/pp6/8/q1pppppp/2PPPBP1/b6N/PP1N1P1P/R2QKB1R b KQkq - 0 11");

  const auto pins = chess::build_pinned_map(board, chess::SideToMove::White);
  const auto [d2_pinned_mask, dir] = pins.knight_pins[chess::to_index(chess::Square::D2)];

  // The knight on g3 is pinned to the king on e1 by the black bishop on b4
  EXPECT_TRUE(d2_pinned_mask != ~Bitboard{0}); // Knights cannot move
}

TEST(BishopPinTest, QueenPins) {
  const chess::Board board =
      make_board("rnb1k1nr/p7/1p6/q1ppppp1/2PPP1Pp/b7/PP1B1P1P/RN1QKBNR w KQkq - 2 15");

  const auto pins = chess::build_pinned_map(board, chess::SideToMove::White);
  const auto [d2_pinned_mask, dir] = pins.queen_pins[chess::to_index(chess::Square::D2)];

  // The bishop on D2 is pinned to the king on e8 by the queen on A5
  EXPECT_TRUE(d2_pinned_mask != 0);
  EXPECT_EQ(d2_pinned_mask, (Bitboard(1) << chess::to_index(chess::Square::A5) |
                             Bitboard(1) << chess::to_index(chess::Square::B4) |
                             Bitboard(1) << chess::to_index(chess::Square::C3) |
                             Bitboard(1) << chess::to_index(chess::Square::D2)));
}