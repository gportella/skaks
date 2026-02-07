#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/move_ordering.hpp"
#include "chess/move_picker.hpp"
#include "chess/moves.hpp"
#include "chess/transposition_table.hpp"
#include "chess/types_io.hpp"

#include <gtest/gtest.h>

TEST(MovePickerTests, InitializesFromBoard) {
  auto board = chess::initial_board(chess::kStartFEN);
  using namespace chess;

  TranspositionTable tt;
  TranspositionEntry tt_entry;
  Move sample_move =
      Move{to_index(Square::E2), to_index(Square::E4), OccupancyType::wP,
           OccupancyType::empty, OccupancyType::empty, kFlagQuiet};
  tt.store(board.position_key, 5, 10, TranspositionFlag::Exact, sample_move, 0);

  auto result = tt.probe(board.position_key, tt_entry);
  if (result) {
    EXPECT_EQ(tt_entry.best_move.from, sample_move.from);
    EXPECT_EQ(tt_entry.best_move.to, sample_move.to);
    EXPECT_EQ(tt_entry.best_move.moving_pc, sample_move.moving_pc);
  } else {
    FAIL() << "Transposition table entry not found";
  }

  chess::MovePicker picker(board, chess::SideToMove::White, &tt_entry.best_move,
                           nullptr, nullptr, 0, 0, nullptr, nullptr, false);

  chess::Move next_move = picker.nextMove();
  EXPECT_EQ(next_move.from, sample_move.from);
  EXPECT_EQ(next_move.to, sample_move.to);
  EXPECT_EQ(next_move.moving_pc, sample_move.moving_pc);
}
