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
#include <cstdint>

namespace chess {

struct SearchParameters {
  int depth = 0;
  int alpha = -INF;
  int beta = INF;
};

struct SearchResult {
  enum class Outcome { InProgress, Mate, DrawByStalemate, DrawByRepetition };

  int score = 0;
  Move best_move{};
  Outcome outcome = Outcome::InProgress;
  std::uint64_t nodes = 0;
  std::uint64_t elapsed_ms = 0;
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
                               int ply, int repetition_start, std::uint64_t& nodes) {
  ++nodes;

  // Try to avoid repetitions
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

  // Transposition Table Lookup
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
      // return if we can cutoff
      if (alpha >= beta) {
        return {cached_score, cached_entry.best_move, SearchResult::Outcome::InProgress};
      }
    }
    // use cached best move for move ordering
    if (cached_entry.best_move.moving_pc != OccupancyType::empty) {
      has_cached_move = true;
      cached_move = cached_entry.best_move;
    }
  }

  alpha_base = alpha;
  beta_base = beta;

  // Leaf node: evaluate
  if (depth == 0) {
    const int eval = evaluator(static_cast<const Board&>(board));
    // Store in TT
    if (tt) {
      tt->store(board.position_key, depth, eval, TranspositionFlag::Exact, Move{}, ply);
    }
    return {eval, Move{}, SearchResult::Outcome::InProgress};
  }

  // Generate legal moves
  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);

  // No legal moves: checkmate or stalemate, store in TT and return
  if (move_count == 0) {
    if (is_check(board, stm)) {
      const int mate_score = (stm == SideToMove::White) ? (-INF + ply) : (INF - ply);
      if (tt) {
        tt->store(board.position_key, depth, mate_score, TranspositionFlag::Exact, Move{}, ply);
      }
      return {mate_score, Move{}, SearchResult::Outcome::Mate};
    }
    constexpr int draw_score = 0;
    if (tt) {
      tt->store(board.position_key, depth, draw_score, TranspositionFlag::Exact, Move{}, ply);
    }
    return {draw_score, Move{}, SearchResult::Outcome::DrawByStalemate};
  }

  uint32_t tt_code = 0;
  if (has_cached_move) {
    tt_code = encode_move(cached_move.from, cached_move.to, cached_move.moving_pc,
                          cached_move.captured_pc, cached_move.promo_pc, cached_move.flags);
  }
  sort_moves(moves, move_count, tt_code);

  SearchResult best = {(stm == SideToMove::White) ? -INF : INF, Move{},
                       SearchResult::Outcome::InProgress};

  for (uint16_t i = 0; i < move_count; ++i) {
    int check_extension = 0;
    Move move = decode_move(moves[i]);
    Undo undo = make_move(board, move);
    const bool irreversible = move_is_irreversible(move);
    const int next_repetition_start = irreversible ? (ply + 1) : repetition_start;
    if (is_check(board, flip_side(stm)) && depth > 0) {
      // Optional: implement check extension if desired
      check_extension = 1;
    }
    auto child = alphabeta_minimax(board, depth - 1 + check_extension, alpha, beta, flip_side(stm),
                                   evaluator, history, tt, ply + 1, next_repetition_start, nodes);
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

  std::uint64_t nodes = 0;
  auto result =
      detail::alphabeta_minimax(board, params.depth, params.alpha, params.beta, stm, evaluator,
                                history, tt, start_ply, repetition_start, nodes);

  if (history) {
    history->ply_count = base_ply;
  }

  result.nodes = nodes;
  return result;
}

inline SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params) {
  return search_position(board, stm, params, DefaultEvaluator{}, nullptr, nullptr);
}

} // namespace chess
