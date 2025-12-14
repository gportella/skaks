#include "chess/search.hpp"

#include "chess/moves.hpp"
#include "chess/scoring_rules.hpp"

namespace chess {

SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta, SideToMove stm) {
  if (depth == 0 || board.is_terminal()) {
    return {evaluate_board(board), Move{}};
  }

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);
  if (move_count == 0) {
    return {evaluate_board(board), Move{}};
  }

  SearchResult best = {(stm == SideToMove::White) ? -INF : INF, Move{}};

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);
    Undo undo = make_move(board, move);
    int score = alphabeta_minimax(board, depth - 1, alpha, beta, flip_side(stm)).score;
    undo_move(board, undo);

    if (stm == SideToMove::White) {
      if (score > alpha) {
        alpha = score;
        best = {score, move};
      }
      if (alpha >= beta) {
        break;
      }
    } else {
      if (score < beta) {
        beta = score;
        best = {score, move};
      }
      if (beta <= alpha) {
        break;
      }
    }
  }

  return best;
}

SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params) {
  return alphabeta_minimax(board, params.depth, params.alpha, params.beta, stm);
}

} // namespace chess
