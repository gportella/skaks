#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/exchange.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <gtest/gtest.h>
#include <string_view>

namespace {

chess::Board parse_board(std::string_view fen) {
  return chess::initial_board(fen);
}

} // namespace

TEST(ExchangeFindSmallestAttacker, PrefersLeastValuableAttacker) {
  const auto board = parse_board("4k3/8/4p3/3P4/5n2/8/4K3/8 b - - 0 1");
  const int target_sq = chess::to_index(chess::Square::D5);

  const auto attacker =
      chess::find_smallest_attacker(board, target_sq, chess::SideToMove::Black);

  ASSERT_TRUE(attacker.has_value());
  EXPECT_EQ(attacker->piece, chess::Piece::bP);
  EXPECT_EQ(attacker->square, chess::to_index(chess::Square::E6));
}

TEST(ExchangeFindSmallestAttacker, ReturnsNulloptWhenNoAttacker) {
  const auto board = parse_board("4k3/8/8/3P4/8/8/4K3/8 b - - 0 1");
  const int target_sq = chess::to_index(chess::Square::D5);

  const auto attacker =
      chess::find_smallest_attacker(board, target_sq, chess::SideToMove::Black);

  EXPECT_FALSE(attacker.has_value());
}

TEST(ExchangeFindSmallestAttacker, FindsKnightAttacker) {
  const auto board = parse_board("4k3/8/8/3P4/5n2/8/4K3/8 b - - 0 1");
  const int target_sq = chess::to_index(chess::Square::D5);

  const auto attacker =
      chess::find_smallest_attacker(board, target_sq, chess::SideToMove::Black);

  ASSERT_TRUE(attacker.has_value());
  EXPECT_EQ(attacker->piece, chess::Piece::bN);
  EXPECT_EQ(attacker->square, chess::to_index(chess::Square::F4));
}

TEST(StaticExchangeEval, ReturnsPositiveGainForSimpleCapture) {
  auto board = parse_board("4k3/8/4p3/3P4/8/8/4K3/8 w - - 0 1");
  const chess::Move capture{
      static_cast<std::uint16_t>(chess::to_index(chess::Square::D5)),
      static_cast<std::uint16_t>(chess::to_index(chess::Square::E6)),
      chess::OccupancyType::wP,
      chess::OccupancyType::bP,
      chess::OccupancyType::empty,
      0};

  const int value = chess::static_exchange_eval(board, capture);

  EXPECT_GT(value, 0);
}

TEST(StaticExchangeEval, ReturnsNegativeWhenCaptureIsLosing) {
  auto board = parse_board("4k3/1p6/p7/1B6/8/8/4K3/8 w - - 0 1");
  const chess::Move capture{
      static_cast<std::uint16_t>(chess::to_index(chess::Square::B5)),
      static_cast<std::uint16_t>(chess::to_index(chess::Square::A6)),
      chess::OccupancyType::wB,
      chess::OccupancyType::bP,
      chess::OccupancyType::empty,
      0};

  const int value = chess::static_exchange_eval(board, capture);

  EXPECT_LT(value, 0);
}

TEST(ExchangeFindSmallestAttacker, FindsBishopAttacker) {
  const auto board = parse_board("b3k3/8/8/3P4/8/8/4K3/8 b - - 0 1");
  const int target_sq = chess::to_index(chess::Square::D5);

  const auto attacker =
      chess::find_smallest_attacker(board, target_sq, chess::SideToMove::Black);

  ASSERT_TRUE(attacker.has_value());
  EXPECT_EQ(attacker->piece, chess::Piece::bB);
  EXPECT_EQ(attacker->square, chess::to_index(chess::Square::A8));
}

TEST(ExchangeFindSmallestAttacker, FindsRookAttacker) {
  const auto board = parse_board("3rk3/8/8/3P4/8/8/4K3/8 b - - 0 1");
  const int target_sq = chess::to_index(chess::Square::D5);

  const auto attacker =
      chess::find_smallest_attacker(board, target_sq, chess::SideToMove::Black);

  ASSERT_TRUE(attacker.has_value());
  EXPECT_EQ(attacker->piece, chess::Piece::bR);
  EXPECT_EQ(attacker->square, chess::to_index(chess::Square::D8));
}

TEST(ExchangeFindSmallestAttacker, FindsQueenAttacker) {
  const auto board = parse_board("3qk3/8/8/3P4/8/8/4K3/8 b - - 0 1");
  const int target_sq = chess::to_index(chess::Square::D5);

  const auto attacker =
      chess::find_smallest_attacker(board, target_sq, chess::SideToMove::Black);

  ASSERT_TRUE(attacker.has_value());
  EXPECT_EQ(attacker->piece, chess::Piece::bQ);
  EXPECT_EQ(attacker->square, chess::to_index(chess::Square::D8));
}

TEST(ExchangeFindSmallestAttacker, FindsKingAttacker) {
  const auto board = parse_board("8/8/4k3/3P4/8/8/4K3/8 b - - 0 1");
  const int target_sq = chess::to_index(chess::Square::D5);

  const auto attacker =
      chess::find_smallest_attacker(board, target_sq, chess::SideToMove::Black);

  ASSERT_TRUE(attacker.has_value());
  EXPECT_EQ(attacker->piece, chess::Piece::bK);
  EXPECT_EQ(attacker->square, chess::to_index(chess::Square::E6));
}
