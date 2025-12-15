#include "chess/board.hpp"
#include "chess/defaults.hpp"

#include <gtest/gtest.h>
#include <string>

namespace chess {
Board parse_fen_string(std::string_view fen);
}

namespace {

chess::Board parse_board(std::string_view fen) {
  chess::Board board = chess::parse_fen_string(fen);
  return board;
}

} // namespace

TEST(BoardFen, RoundTripStartPosition) {
  const auto board = chess::initial_board(chess::kStartFEN);
  EXPECT_EQ(chess::board_to_fen(board), std::string(chess::kStartFEN));
}

TEST(BoardFen, RoundTripWithEnPassantAndNoCastling) {
  const std::string fen = "rnbqkbnr/ppp1pppp/8/3p4/3Pp3/8/PPP1PPPP/RNBQKBNR w KQkq e3 0 3";
  const auto board = parse_board(fen);
  EXPECT_EQ(chess::board_to_fen(board), fen);
}

TEST(BoardCheck, BlackKingNotInCheckInSamplePosition) {
  const std::string fen = "Bk3r2/n6p/b5p1/p7/3P4/4R1P1/Pq3PQP/RN4K1 b - - 1 28";
  auto board = chess::initial_board(fen);
  EXPECT_FALSE(chess::is_check(board, chess::SideToMove::Black));
}
