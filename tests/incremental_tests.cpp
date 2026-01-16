#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/piece_values.hpp"
#include "chess/pst_tables.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <gtest/gtest.h>
#include <random>

namespace chess {

// Helper to calculate material from scratch
int calculate_material_scratch(const Board& board) {
  int score = 0;
  for (int sq = 0; sq < 64; ++sq) {
    const auto piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece != OccupancyType::empty) {
      score += piece_material_value(piece);
    }
  }
  return score;
}

// Helper to calculate PST scores from scratch
struct PstScores {
  int midgame;
  int endgame;
  int phase;
};

PstScores calculate_pst_scratch(const Board& board) {
  int midgame = 0;
  int endgame = 0;
  int phase = 0;

  const auto& mg_tables = midgame_pst();
  const auto& eg_tables = endgame_pst();

  for (int sq = 0; sq < 64; ++sq) {
    const auto piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece == OccupancyType::empty)
      continue;

    const bool white_piece = is_white(piece);
    const int type_index = (static_cast<int>(piece) - 1) % 6;
    const int oriented_sq = white_piece ? sq : mirror_rank(sq);

    const int mg_entry = mg_tables[static_cast<std::size_t>(type_index)]
                                  [static_cast<std::size_t>(oriented_sq)];
    const int eg_entry = eg_tables[static_cast<std::size_t>(type_index)]
                                  [static_cast<std::size_t>(oriented_sq)];

    midgame += white_piece ? mg_entry : -mg_entry;
    endgame += white_piece ? eg_entry : -eg_entry;
    phase += kPstPhaseWeights[static_cast<std::size_t>(type_index)];
  }
  return {midgame, endgame, phase};
}

class IncrementalTest : public ::testing::Test {
protected:
  void VerifyIncrementalScores(const Board& board) {
    // Verify Material
    int expected_material = calculate_material_scratch(board);
    EXPECT_EQ(board.material_score, expected_material)
        << "Material score mismatch. Incremental: " << board.material_score
        << ", Scratch: " << expected_material
        << "\nFEN: " << board_to_fen(board);

    // Verify PST
    PstScores expected_pst = calculate_pst_scratch(board);
    EXPECT_EQ(board.pst_midgame_score, expected_pst.midgame)
        << "PST Midgame mismatch. Incremental: " << board.pst_midgame_score
        << ", Scratch: " << expected_pst.midgame
        << "\nFEN: " << board_to_fen(board);
    EXPECT_EQ(board.pst_endgame_score, expected_pst.endgame)
        << "PST Endgame mismatch. Incremental: " << board.pst_endgame_score
        << ", Scratch: " << expected_pst.endgame
        << "\nFEN: " << board_to_fen(board);
    EXPECT_EQ(board.phase, expected_pst.phase)
        << "Phase mismatch. Incremental: " << board.phase
        << ", Scratch: " << expected_pst.phase
        << "\nFEN: " << board_to_fen(board);
  }

  void RunRandomWalk(Board& board, int depth, int seed) {
    if (depth == 0)
      return;

    VerifyIncrementalScores(board);

    uint16_t move_count = 0;
    auto moves = generate_all_moves(board, board.side_to_move, move_count);

    if (move_count == 0)
      return;

    std::mt19937 gen(static_cast<unsigned int>(seed));
    std::uniform_int_distribution<> dis(0, move_count - 1);

    // Try a few random moves at this node
    int branches = std::min(static_cast<int>(move_count), 3);
    for (int i = 0; i < branches; ++i) {
      int idx = dis(gen);
      Move m = decode_move(moves[static_cast<std::size_t>(idx)]);

      Undo undo = make_move(board, m);
      VerifyIncrementalScores(board);

      RunRandomWalk(board, depth - 1, seed + i);

      undo_move(board, undo);
      VerifyIncrementalScores(board);
    }
  }
};

TEST_F(IncrementalTest, StartPos) {
  Board board = initial_board("");
  VerifyIncrementalScores(board);
}

TEST_F(IncrementalTest, RandomWalkStartPos) {
  Board board = initial_board("");
  RunRandomWalk(board, 4, 12345);
}

TEST_F(IncrementalTest, Kiwipete) {
  Board board = initial_board(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  RunRandomWalk(board, 4, 54321);
}

TEST_F(IncrementalTest, Castling) {
  // Position with castling rights and moves available
  Board board = initial_board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  RunRandomWalk(board, 3, 111);
}

TEST_F(IncrementalTest, Promotion) {
  // Position with promotion potential
  Board board = initial_board("8/P7/8/8/8/8/7p/8 w - - 0 1");
  RunRandomWalk(board, 3, 222);
}

TEST_F(IncrementalTest, EnPassant) {
  // Position with en passant potential
  Board board = initial_board(
      "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
  RunRandomWalk(board, 3, 333);
}

} // namespace chess
