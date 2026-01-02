#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/move_ordering.hpp"
#include "chess/moves.hpp"

#include <gtest/gtest.h>

namespace {

chess::Move make_quiet_move(chess::Square from, chess::Square to,
                            chess::OccupancyType piece) {
  chess::Move move{};
  move.from = static_cast<uint16_t>(from);
  move.to = static_cast<uint16_t>(to);
  move.moving_pc = piece;
  move.captured_pc = chess::OccupancyType::empty;
  move.promo_pc = chess::OccupancyType::empty;
  move.flags = chess::kFlagQuiet;
  return move;
}

} // namespace

TEST(MoveOrderingHistory, DecaysAfterThreshold) {
  chess::HistoryTable history;
  const auto move = make_quiet_move(chess::Square::E2, chess::Square::E4,
                                    chess::OccupancyType::wP);

  history.update(move, 750, true);
  const int before_decay = history.score(move);
  ASSERT_GT(before_decay, chess::kHistoryDecayTrigger);

  history.maybe_decay();
  const int after_decay = history.score(move);
  EXPECT_EQ(after_decay, before_decay / chess::kHistoryDecayFactor);
}

TEST(MoveOrderingCounter, StoresQuietRepliesOnly) {
  chess::CounterMoveTable table;
  const auto prev = make_quiet_move(chess::Square::G1, chess::Square::F3,
                                    chess::OccupancyType::wN);
  const auto reply = make_quiet_move(chess::Square::B8, chess::Square::C6,
                                     chess::OccupancyType::bN);

  table.store(prev, reply);
  const uint32_t encoded_reply = table.lookup(prev);
  EXPECT_EQ(encoded_reply,
            chess::encode_move(reply.from, reply.to, reply.moving_pc,
                               reply.captured_pc, reply.promo_pc,
                               reply.flags));

  chess::Move capture = reply;
  capture.captured_pc = chess::OccupancyType::wP;
  capture.flags = 0;
  table.store(prev, capture);
  EXPECT_EQ(table.lookup(prev), encoded_reply);
}

TEST(MoveOrderingSort, BoostsCounterMovePriority) {
  auto board = chess::initial_board(chess::kStartFEN);
  std::array<uint32_t, chess::kMaxMovementCount> moves{};
  moves[0] = chess::encode_move(static_cast<int>(chess::Square::G1),
                                static_cast<int>(chess::Square::F3),
                                chess::OccupancyType::wN,
                                chess::OccupancyType::empty,
                                chess::OccupancyType::empty,
                                chess::kFlagQuiet);
  moves[1] = chess::encode_move(static_cast<int>(chess::Square::B1),
                                static_cast<int>(chess::Square::C3),
                                chess::OccupancyType::wN,
                                chess::OccupancyType::empty,
                                chess::OccupancyType::empty,
                                chess::kFlagQuiet);

  uint16_t move_count = 2;
  chess::sort_moves(board, moves, move_count, 0);
  EXPECT_EQ(moves[0], chess::encode_move(static_cast<int>(chess::Square::G1),
                                         static_cast<int>(chess::Square::F3),
                                         chess::OccupancyType::wN,
                                         chess::OccupancyType::empty,
                                         chess::OccupancyType::empty,
                                         chess::kFlagQuiet));

  moves[0] = chess::encode_move(static_cast<int>(chess::Square::G1),
                                static_cast<int>(chess::Square::F3),
                                chess::OccupancyType::wN,
                                chess::OccupancyType::empty,
                                chess::OccupancyType::empty,
                                chess::kFlagQuiet);
  moves[1] = chess::encode_move(static_cast<int>(chess::Square::B1),
                                static_cast<int>(chess::Square::C3),
                                chess::OccupancyType::wN,
                                chess::OccupancyType::empty,
                                chess::OccupancyType::empty,
                                chess::kFlagQuiet);
  const uint32_t counter = moves[1];
  chess::sort_moves(board, moves, move_count, 0, nullptr, nullptr, 0, counter);
  EXPECT_EQ(moves[0], counter);
}
