#include "chess/board.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/types_io.hpp"

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

TEST(ScoringRules, HeavierPiecesYieldLargerThreatPenalty) {
  auto rook_target = make_board("4k3/8/3r4/8/4N3/8/8/4K3 w - - 0 1");
  auto pawn_target = make_board("4k3/8/3p4/8/4N3/8/8/4K3 w - - 0 1");

  const int rook_score = chess::evaluate_attacking_pieces(rook_target);
  const int pawn_score = chess::evaluate_attacking_pieces(pawn_target);

  EXPECT_GT(rook_score, pawn_score);
}

TEST(ScoringRules, AdvancedTargetsArePenalizedMore) {
  auto forward_target = make_board("4k3/8/8/8/4r3/8/8/1B4K1 w - - 0 1");
  auto home_target = make_board("4k3/7r/8/8/8/8/8/1B4K1 w - - 0 1");

  const int forward_score = chess::evaluate_attacking_pieces(forward_target);
  const int home_score = chess::evaluate_attacking_pieces(home_target);

  EXPECT_GT(forward_score, home_score);
}
