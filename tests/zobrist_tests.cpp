#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/types_io.hpp"
#include "chess/zobrist.hpp"

#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace chess {

TEST(ZobristTests, MakeUndoKeepsKeyInSync) {
  Board board = initial_board("");
  const auto original_key = board.position_key;

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, board.side_to_move, move_count);
  ASSERT_GT(move_count, 0);

  for (uint16_t i = 0; i < move_count; ++i) {
    Board copy = board;
    Move move = decode_move(moves[i]);
    Undo undo = make_move(copy, move);

    EXPECT_EQ(copy.position_key, compute_position_key(copy))
        << "FEN: " << board_to_fen(copy);

    undo_move(copy, undo);
    EXPECT_EQ(copy.position_key, original_key) << "FEN: " << board_to_fen(copy);
  }
}

TEST(ZobristTests, RandomWalkMaintainsKey) {
  Board board = initial_board("");
  std::mt19937 rng(1337);
  std::vector<Undo> undo_stack;
  undo_stack.reserve(64);

  for (int depth = 0; depth < 64; ++depth) {
    uint16_t move_count = 0;
    auto moves = generate_legal_moves(board, board.side_to_move, move_count);
    if (move_count == 0) {
      break;
    }

    std::uniform_int_distribution<int> dist(0, move_count - 1);
    Move move = decode_move(moves[static_cast<std::size_t>(dist(rng))]);
    Undo undo = make_move(board, move);

    EXPECT_EQ(board.position_key, compute_position_key(board))
        << "FEN: " << board_to_fen(board);

    undo_stack.push_back(undo);
  }

  while (!undo_stack.empty()) {
    Undo undo = undo_stack.back();
    undo_stack.pop_back();
    undo_move(board, undo);

    EXPECT_EQ(board.position_key, compute_position_key(board))
        << "FEN: " << board_to_fen(board);
  }
}

TEST(ZobristTests, IncrementalUpdateMatchesRecompute) {
  Board board = initial_board("");
  std::mt19937 rng(4242);

  for (int depth = 0; depth < 64; ++depth) {
    uint16_t move_count = 0;
    auto moves = generate_legal_moves(board, board.side_to_move, move_count);
    if (move_count == 0) {
      break;
    }

    std::uniform_int_distribution<int> dist(0, move_count - 1);
    Move move = decode_move(moves[static_cast<std::size_t>(dist(rng))]);

    Board copy = board;
    Undo undo = make_move(copy, move);
    const auto recomputed = compute_position_key(copy);
    const bool ep_hash_after =
        (copy.en_passant >= 0 && ep_capture_available(copy));

    copy.position_key = undo.position_key_before;
    update_key_for_move(copy, undo, to_mask(copy.castling_rights),
                        undo.ep_hash_before, ep_hash_after);

    EXPECT_EQ(copy.position_key, recomputed) << "FEN: " << board_to_fen(copy);

    board = copy;
  }
}

} // namespace chess
