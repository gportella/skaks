#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/nnue.hpp"
#include "chess/nnue_incremental.hpp"
#include "chess/types_io.hpp"

#include <gtest/gtest.h>

namespace chess {

TEST(NnueIncremental, QuietMoveDirtyPiece) {
  Board board = initial_board("");
  SfNnueStack stack;
  stack.reset();

  Move move{};
  move.from = to_index(Square::E2);
  move.to = to_index(Square::E4);
  move.moving_pc = OccupancyType::wP;
  move.captured_pc = OccupancyType::empty;
  move.promo_pc = OccupancyType::empty;
  move.flags = kFlagDoublePush | kFlagQuiet;

  Undo undo = make_move(board, move);
  stack.push_move(board, move, undo);

  const DirtyPiece& dirty = stack.current_data().dirtyPiece;
  EXPECT_EQ(dirty.dirtyNum, 1);
  EXPECT_EQ(dirty.pc[0], sf_nnue_piece_code(OccupancyType::wP));
  EXPECT_EQ(dirty.from[0], to_index(Square::E2));
  EXPECT_EQ(dirty.to[0], to_index(Square::E4));

  undo_move(board, undo);
  stack.pop();
}

TEST(NnueIncremental, CaptureDirtyPiece) {
  Board board = initial_board(
      "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
  SfNnueStack stack;
  stack.reset();

  Move move{};
  move.from = to_index(Square::E4);
  move.to = to_index(Square::D5);
  move.moving_pc = OccupancyType::wP;
  move.captured_pc = OccupancyType::bP;
  move.promo_pc = OccupancyType::empty;
  move.flags = 0;

  Undo undo = make_move(board, move);
  stack.push_move(board, move, undo);

  const DirtyPiece& dirty = stack.current_data().dirtyPiece;
  ASSERT_EQ(dirty.dirtyNum, 2);
  EXPECT_EQ(dirty.pc[0], sf_nnue_piece_code(OccupancyType::wP));
  EXPECT_EQ(dirty.from[0], to_index(Square::E4));
  EXPECT_EQ(dirty.to[0], to_index(Square::D5));
  EXPECT_EQ(dirty.pc[1], sf_nnue_piece_code(OccupancyType::bP));
  EXPECT_EQ(dirty.from[1], to_index(Square::D5));
  EXPECT_EQ(dirty.to[1], 64);

  undo_move(board, undo);
  stack.pop();
}

TEST(NnueIncremental, PromotionCaptureDirtyPiece) {
  Board board =
      initial_board("rnbq1rk1/4Pppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQ - 0 1");
  SfNnueStack stack;
  stack.reset();

  Move move{};
  move.from = to_index(Square::E7);
  move.to = to_index(Square::F8);
  move.moving_pc = OccupancyType::wP;
  move.captured_pc = OccupancyType::bR;
  move.promo_pc = OccupancyType::wQ;
  move.flags = 0;

  Undo undo = make_move(board, move);
  stack.push_move(board, move, undo);

  const DirtyPiece& dirty = stack.current_data().dirtyPiece;
  ASSERT_EQ(dirty.dirtyNum, 3);
  EXPECT_EQ(dirty.pc[0], sf_nnue_piece_code(OccupancyType::wP));
  EXPECT_EQ(dirty.from[0], to_index(Square::E7));
  EXPECT_EQ(dirty.to[0], 64);
  EXPECT_EQ(dirty.pc[1], sf_nnue_piece_code(OccupancyType::wQ));
  EXPECT_EQ(dirty.from[1], 64);
  EXPECT_EQ(dirty.to[1], to_index(Square::F8));
  EXPECT_EQ(dirty.pc[2], sf_nnue_piece_code(OccupancyType::bR));
  EXPECT_EQ(dirty.from[2], to_index(Square::F8));
  EXPECT_EQ(dirty.to[2], 64);

  undo_move(board, undo);
  stack.pop();
}

TEST(NnueIncremental, CastlingDirtyPieceOrdering) {
  Board board =
      initial_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1");
  SfNnueStack stack;
  stack.reset();

  Move move{};
  move.from = to_index(Square::E1);
  move.to = to_index(Square::G1);
  move.moving_pc = OccupancyType::wK;
  move.captured_pc = OccupancyType::empty;
  move.promo_pc = OccupancyType::empty;
  move.flags = kFlagCastle;

  Undo undo = make_move(board, move);
  stack.push_move(board, move, undo);

  const DirtyPiece& dirty = stack.current_data().dirtyPiece;
  ASSERT_EQ(dirty.dirtyNum, 2);
  EXPECT_EQ(dirty.pc[0], sf_nnue_piece_code(OccupancyType::wK));
  EXPECT_EQ(dirty.from[0], to_index(Square::E1));
  EXPECT_EQ(dirty.to[0], to_index(Square::G1));
  EXPECT_EQ(dirty.pc[1], sf_nnue_piece_code(OccupancyType::wR));
  EXPECT_EQ(dirty.from[1], to_index(Square::H1));
  EXPECT_EQ(dirty.to[1], to_index(Square::F1));

  undo_move(board, undo);
  stack.pop();
}

} // namespace chess
