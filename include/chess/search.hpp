#pragma once

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/history.hpp"
#include "chess/moves.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/transposition_table.hpp"
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
  enum class Outcome { InProgress, Mate, DrawByStalemate, DrawByRepetition };

  int score;
  Move best_move;
  Outcome outcome = Outcome::InProgress;
};

struct DefaultEvaluator {
  int operator()(const Board& board) const {
    return evaluate_board(board);
  }
};

namespace detail {

template <typename Evaluator>
SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta, SideToMove stm,
                               Evaluator& evaluator, MoveHistory* history, TranspositionTable* tt,
                               int ply, int repetition_start) {
  if (history) {
    if (ply < MAX_PLY) {
      history->key_history[static_cast<std::size_t>(ply)] = board.position_key;
      history->ply_count = std::max(history->ply_count, ply + 1);
      uint8_t repeats = 0;
      const int repeat_begin = std::max(0, std::min(repetition_start, ply));
      for (int i = repeat_begin; i < ply; ++i) {
        if (history->key_history[static_cast<std::size_t>(i)] == board.position_key) {
          ++repeats;
        }
      }
      history->repetition_counts[static_cast<std::size_t>(ply)] = repeats;
      if (repeats >= 2) {
        const int repetition_score =
            (stm == SideToMove::White) ? -REPETITION_PENALTY : REPETITION_PENALTY;
        return {repetition_score, Move{}, SearchResult::Outcome::DrawByRepetition};
      }
    }
  }

  int alpha_base = alpha;
  int beta_base = beta;

  TranspositionEntry cached_entry;
  bool has_cached_move = false;
  Move cached_move{};
  if (tt && tt->probe(board.position_key, cached_entry)) {
    const int cached_score = TranspositionTable::decode_score(cached_entry.score, ply);
    if (cached_entry.depth >= depth) {
      switch (cached_entry.flag) {
      case TranspositionFlag::Exact:
        return {cached_score, cached_entry.best_move, SearchResult::Outcome::InProgress};
      case TranspositionFlag::LowerBound:
        alpha = std::max(alpha, cached_score);
        break;
      case TranspositionFlag::UpperBound:
        beta = std::min(beta, cached_score);
        break;
      case TranspositionFlag::None:
        break;
      }
      if (alpha >= beta) {
        return {cached_score, cached_entry.best_move, SearchResult::Outcome::InProgress};
      }
    }
    if (cached_entry.best_move.moving_pc != OccupancyType::empty) {
      has_cached_move = true;
      cached_move = cached_entry.best_move;
    }
  }

  alpha_base = alpha;
  beta_base = beta;

  if (depth == 0 || board.is_terminal()) {
    const int eval = evaluator(static_cast<const Board&>(board));
    if (tt) {
      tt->store(board.position_key, depth, eval, TranspositionFlag::Exact, Move{}, ply);
    }
    return {eval, Move{}, SearchResult::Outcome::InProgress};
  }

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);
  if (move_count == 0) {
    if (is_check(board, stm)) {
      const int mate_score = (stm == SideToMove::White) ? (-INF + ply) : (INF - ply);
      if (tt) {
        tt->store(board.position_key, depth, mate_score, TranspositionFlag::Exact, Move{}, ply);
      }
      std::cout << "Mate detected at ply " << ply << " score=" << mate_score << "\n";
      return {mate_score, Move{}, SearchResult::Outcome::Mate};
    }
    constexpr int draw_score = 0;
    if (tt) {
      tt->store(board.position_key, depth, draw_score, TranspositionFlag::Exact, Move{}, ply);
    }
    return {draw_score, Move{}, SearchResult::Outcome::DrawByStalemate};
  }

  if (has_cached_move) {
    const auto cached_code =
        encode_move(cached_move.from, cached_move.to, cached_move.moving_pc,
                    cached_move.captured_pc, cached_move.promo_pc, cached_move.flags);
    for (uint16_t i = 0; i < move_count; ++i) {
      if (moves[i] == cached_code) {
        std::swap(moves[0], moves[i]);
        break;
      }
    }
  }

  SearchResult best = {(stm == SideToMove::White) ? -INF : INF, Move{},
                       SearchResult::Outcome::InProgress};

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);
    Undo undo = make_move(board, move);
    const bool irreversible = move_is_irreversible(move);
    const int next_repetition_start = irreversible ? (ply + 1) : repetition_start;
    auto child = alphabeta_minimax(board, depth - 1, alpha, beta, flip_side(stm), evaluator,
                                   history, tt, ply + 1, next_repetition_start);
    int score = child.score;
    undo_move(board, undo);
    if (ply + 1 < MAX_PLY && history) {
      history->repetition_counts[static_cast<std::size_t>(ply + 1)] = 0;
    }

    const bool is_better = (best.best_move.moving_pc == OccupancyType::empty) ? true
                           : (stm == SideToMove::White)                       ? (score > best.score)
                                                        : (score < best.score);
    if (is_better) {
      best.score = score;
      best.best_move = move;
      best.outcome = child.outcome;
    }

    if (stm == SideToMove::White) {
      if (score > alpha) {
        alpha = score;
      }
      if (alpha >= beta) {
        break;
      }
    } else {
      if (score < beta) {
        beta = score;
      }
      if (beta <= alpha) {
        break;
      }
    }
  }

  if (tt) {
    TranspositionFlag flag = TranspositionFlag::Exact;
    if (best.score <= alpha_base) {
      flag = TranspositionFlag::UpperBound;
    } else if (best.score >= beta_base) {
      flag = TranspositionFlag::LowerBound;
    }
    tt->store(board.position_key, depth, best.score, flag, best.best_move, ply);
  }

  return best;
}

} // namespace detail

template <typename Evaluator>
SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params,
                             Evaluator evaluator, MoveHistory* history = nullptr,
                             TranspositionTable* tt = nullptr, int repetition_start = 0) {
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
                                          evaluator, history, tt, start_ply, repetition_start);

  if (history) {
    history->ply_count = base_ply;
  }

  return result;
}

inline SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params) {
  return search_position(board, stm, params, DefaultEvaluator{}, nullptr, nullptr);
}

} // namespace chess
