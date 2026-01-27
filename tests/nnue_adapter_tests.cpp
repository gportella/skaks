// SPDX-License-Identifier: GPL-3.0-or-later
#include "chess/board.hpp"
#include "chess/engine.hpp"
#include "chess/moves.hpp"
#include "chess/nnue_sf.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>

namespace chess {
namespace {

void ensure_nnue_init() {
  static bool initialized = false;
  if (!initialized) {
    Engine engine;
    engine.init_nnue();
    initialized = true;
  }
}

void verify_incremental_matches_refresh(const Board& board,
                                        const NnueAdapter& adapter) {
  const int refresh_eval = evaluate_nnue_stockfish(board);
  const int incremental_eval = evaluate_nnue_stockfish_incremental(adapter);
  EXPECT_EQ(incremental_eval, refresh_eval) << "FEN: " << board_to_fen(board);
}

void run_random_walk(Board& board, NnueAdapter& adapter, int depth,
                     std::mt19937& rng) {
  if (depth == 0) {
    return;
  }

  verify_incremental_matches_refresh(board, adapter);

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, board.side_to_move, move_count);
  if (move_count == 0) {
    return;
  }

  std::uniform_int_distribution<int> dist(0, static_cast<int>(move_count) - 1);
  const int branches = std::min<int>(move_count, 3);

  for (int i = 0; i < branches; ++i) {
    Move move = decode_move(moves[static_cast<std::size_t>(dist(rng))]);
    Undo undo = make_move(board, move);
    adapter.push_move(move);

    run_random_walk(board, adapter, depth - 1, rng);

    adapter.pop_move();
    undo_move(board, undo);
  }

  verify_incremental_matches_refresh(board, adapter);
}

} // namespace

TEST(NnueAdapterTest, IncrementalMatchesRefreshStartPos) {
  ensure_nnue_init();

  Board board = initial_board("");
  NnueAdapter adapter(board);

  std::mt19937 rng(12345);
  run_random_walk(board, adapter, 4, rng);
}

TEST(NnueAdapterTest, IncrementalMatchesRefreshSpecialMoves) {
  ensure_nnue_init();

  Board board =
      initial_board("r3k2r/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/R3K2R w KQkq f6 0 3");
  NnueAdapter adapter(board);

  std::mt19937 rng(54321);
  run_random_walk(board, adapter, 4, rng);
}

} // namespace chess
