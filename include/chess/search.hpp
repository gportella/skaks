#pragma once

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/history.hpp"
#include "chess/moves.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/types.hpp"

#include <algorithm>
#include <cstddef>

namespace chess {

struct SearchParameters {
  int depth = 0;
  int alpha = -INF;
  int beta = INF;
};

struct SearchResult {
  int score;
  Move best_move;
};

struct DefaultEvaluator {
  int operator()(const Board& board) const {
    return evaluate_board(board);
  }
};

namespace detail {

template <typename Evaluator>
SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta, SideToMove stm,
                               Evaluator& evaluator, MoveHistory* history, int ply,
                               int repetition_start) {
  if (history) {
    if (ply < MAX_PLY) {
      history->key_history[static_cast<std::size_t>(ply)] = board.position_key;
      history->ply_count = std::max(history->ply_count, ply + 1);
    }
    const int repeat_begin = std::max(repetition_start, 0);
    for (int i = repeat_begin; i < ply; ++i) {
      if (history->key_history[static_cast<std::size_t>(i)] == board.position_key) {
        return {0, Move{}};
      }
    }
  }

  if (depth == 0 || board.is_terminal()) {
    return {evaluator(static_cast<const Board&>(board)), Move{}};
  }

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);
  if (move_count == 0) {
    return {evaluator(static_cast<const Board&>(board)), Move{}};
  }

  SearchResult best = {(stm == SideToMove::White) ? -INF : INF, Move{}};

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);
    Undo undo = make_move(board, move);
    const bool irreversible = move_is_irreversible(move);
    const int next_repetition_start = irreversible ? (ply + 1) : repetition_start;
    int score = alphabeta_minimax(board, depth - 1, alpha, beta, flip_side(stm), evaluator, history,
                                  ply + 1, next_repetition_start)
                    .score;
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

} // namespace detail

template <typename Evaluator>
SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params,
                             Evaluator evaluator, MoveHistory* history = nullptr,
                             int repetition_start = 0) {
  int base_ply = 0;
  int start_ply = 0;

  if (history) {
    if (history->ply_count == 0) {
      history->key_history[0] = board.position_key;
      history->ply_count = 1;
    } else if (history->ply_count > 0) {
      const auto idx = static_cast<std::size_t>(history->ply_count - 1);
      history->key_history[idx] = board.position_key;
    }

    base_ply = history->ply_count;
    start_ply = std::max(history->ply_count - 1, 0);
    repetition_start = std::max(0, std::min(repetition_start, start_ply + 1));
  }

  auto result = detail::alphabeta_minimax(board, params.depth, params.alpha, params.beta, stm,
                                          evaluator, history, start_ply, repetition_start);

  if (history) {
    history->ply_count = base_ply;
  }

  return result;
}

inline SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params) {
  return search_position(board, stm, params, DefaultEvaluator{});
}

} // namespace chess
