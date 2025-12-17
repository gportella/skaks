#include "chess/search.hpp"

#include "chess/moves.hpp"
#include "chess/scoring_rules.hpp"

#include <algorithm>
#include <array>

namespace chess {

namespace {

SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta, SideToMove stm,
                               const EvaluatorFn& evaluator, MoveHistory* history,
                               TranspositionTable* tt, int ply, int repetition_start,
                               int ply_from_root, bool is_pv, std::uint64_t& nodes,
                               const uint32_t* excluded_root_moves,
                               std::size_t excluded_root_count) {
  ++nodes;

  const bool apply_root_exclusions =
      excluded_root_moves != nullptr && excluded_root_count > 0 && ply_from_root == 0;

  auto is_excluded_code = [&](uint32_t encoded) -> bool {
    if (!apply_root_exclusions) {
      return false;
    }
    for (std::size_t idx = 0; idx < excluded_root_count; ++idx) {
      if (excluded_root_moves[idx] == encoded) {
        return true;
      }
    }
    return false;
  };

  auto is_excluded_move = [&](const Move& move) -> bool {
    if (!apply_root_exclusions || move.moving_pc == OccupancyType::empty) {
      return false;
    }
    const uint32_t encoded = encode_move(move.from, move.to, move.moving_pc, move.captured_pc,
                                         move.promo_pc, move.flags);
    return is_excluded_code(encoded);
  };

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
    if (!is_excluded_move(cached_entry.best_move)) {
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
  }

  alpha_base = alpha;
  beta_base = beta;

  if (depth == 0) {
    const int eval = evaluator(static_cast<const Board&>(board));
    if (tt) {
      tt->store(board.position_key, depth, eval, TranspositionFlag::Exact, Move{}, ply);
    }
    return {eval, Move{}, SearchResult::Outcome::InProgress};
  }

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);

  if (apply_root_exclusions) {
    uint16_t write_idx = 0;
    for (uint16_t i = 0; i < move_count; ++i) {
      const uint32_t code = moves[i];
      if (is_excluded_code(code)) {
        continue;
      }
      moves[write_idx++] = code;
    }
    move_count = write_idx;
  }

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
    if (is_excluded_move(cached_move)) {
      has_cached_move = false;
    } else {
      tt_code = encode_move(cached_move.from, cached_move.to, cached_move.moving_pc,
                            cached_move.captured_pc, cached_move.promo_pc, cached_move.flags);
    }
  }
  sort_moves(moves, move_count, tt_code);

  SearchResult best = {(stm == SideToMove::White) ? -INF : INF, Move{},
                       SearchResult::Outcome::InProgress};

  int moves_tried = 0;

  for (uint16_t i = 0; i < move_count; ++i) {
    int child_depth = depth - 1;
    moves_tried += 1;
    const bool is_first_move = (moves_tried == 1);

    Move move = decode_move(moves[i]);
    Undo undo = make_move(board, move);

    const bool irreversible = move_is_irreversible(move);
    const int next_repetition_start = irreversible ? (ply + 1) : repetition_start;

    const bool in_check_after_move = is_check(board, flip_side(stm));
    if (child_depth == 0 && in_check_after_move && ply < static_cast<int>(MAX_PLY) - 1) {
      child_depth = 1;
    }

    int reduction = 0;
    const bool is_capture = undo.captured_pc != OccupancyType::empty;
    if (child_depth > 2 && depth >= 0 && moves_tried > 3 && !is_capture && !in_check_after_move) {
      reduction = 1;
    }

    int search_depth = std::max(0, child_depth);

    auto run_search = [&](int depth_to_use, int a_val, int b_val, bool pv_flag) -> SearchResult {
      return alphabeta_minimax(board, depth_to_use, a_val, b_val, flip_side(stm), evaluator,
                               history, tt, ply + 1, next_repetition_start, ply_from_root + 1,
                               pv_flag, nodes, nullptr, 0);
    };

    bool need_full_search = true;
    SearchResult child = {};
    if (reduction > 0) {
      const int reduced_depth = std::max(0, search_depth - reduction);
      child = run_search(reduced_depth, alpha, beta, false);
      if (stm == SideToMove::White) {
        need_full_search = child.score > alpha;
      } else {
        need_full_search = child.score < beta;
      }
    }

    bool use_pvs = false;
    int narrow_alpha = alpha;
    int narrow_beta = beta;

    if (need_full_search) {
      use_pvs = is_pv && !is_first_move && (search_depth >= 0) && (alpha > -MATE_BOUND) &&
                beta < MATE_BOUND && (beta - alpha) > 1;
      if (stm == SideToMove::White) {
        narrow_beta = std::min(beta, alpha + 1);
        use_pvs = use_pvs && narrow_alpha <= narrow_beta;
      } else {
        narrow_alpha = std::max(alpha, beta - 1);
        use_pvs = use_pvs && narrow_beta >= narrow_alpha;
      }
      if (use_pvs) {
        child = run_search(search_depth, narrow_alpha, narrow_beta, true);
        if (child.score >= narrow_beta) {
          child = run_search(search_depth, alpha, beta, is_pv);
        }
      } else {
        child = run_search(search_depth, alpha, beta, is_pv && is_first_move);
      }
    }

    const int score = child.score;

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
  // TODO: clean up transposition table after lots of searches?

  return best;
}

} // namespace

SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params,
                             const EvaluatorFn& evaluator, MoveHistory* history,
                             TranspositionTable* tt, int repetition_start) {
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

  int max_root_moves = 0;
  uint16_t move_count = 0;
  generate_legal_moves(board, stm, move_count);
  if (move_count == 0) {
    return SearchResult{0, Move{}, SearchResult::Outcome::InProgress, 0, 0};
  }
  max_root_moves = move_count;
  auto excluded_moves = std::array<uint32_t, kMaxMovementCount>{};
  std::size_t excluded_count = 0;
  if (params.root_excluded_moves && params.root_excluded_count > 0) {
    const std::size_t copy_count =
        std::min<std::size_t>(params.root_excluded_count, excluded_moves.size());
    std::copy_n(params.root_excluded_moves, copy_count, excluded_moves.begin());
    excluded_count = copy_count;
  }
  const std::size_t static_excluded_count = excluded_count;
  const bool multi_pv = params.pv_count > 1;
  std::uint64_t nodes = 0;
  int aspiration_window = ASPIRATION_WINDOW_INITIAL;
  SearchResult best_result{};
  SearchResult result{};

  int best_score = 0;
  for (int current_depth = 1; current_depth <= params.depth; ++current_depth) {
    excluded_count = static_excluded_count;
    int pv_generated = 0;
    while (true) {
      int alpha = -INF;
      int beta = INF;
      if (current_depth == 1 || best_result.best_move.moving_pc == OccupancyType::empty) {
        alpha = -INF;
        beta = INF;
      } else {
        alpha = std::max(best_score - aspiration_window, -MATE_BOUND);
        beta = std::min(best_score + aspiration_window, MATE_BOUND);
      }

      int window = aspiration_window;
      while (true) {
        const bool is_pv = true;
        result =
            alphabeta_minimax(board, current_depth, alpha, beta, stm, evaluator, history, tt,
                              start_ply, repetition_start, 0, is_pv, nodes,
                              excluded_count ? excluded_moves.data() : nullptr, excluded_count);
        if (result.score <= alpha) {
          if (alpha <= -MATE_BOUND) {
            alpha = -INF;
            beta = INF;
          } else {
            window = std::min(window * 2, ASPIRATION_WINDOW_MAX);
            alpha = std::max(result.score - window, -MATE_BOUND);
            beta = std::min(result.score + window, MATE_BOUND);
            continue;
          }
        }
        if (result.score >= beta) {
          if (beta >= MATE_BOUND) {
            alpha = -INF;
            beta = INF;
          } else {
            window = std::min(window * 2, ASPIRATION_WINDOW_MAX);
            alpha = std::max(result.score - window, -MATE_BOUND);
            beta = std::min(result.score + window, MATE_BOUND);
            continue;
          }
        }
        break;
      }

      if (result.best_move.moving_pc == OccupancyType::empty || !max_root_moves) {
        if (best_result.best_move.moving_pc == OccupancyType::empty) {
          best_result = result;
          best_score = result.score;
        }
        aspiration_window = window;
        break;
      }

      ++pv_generated;
      const bool improving = best_result.best_move.moving_pc == OccupancyType::empty ? true
                             : (stm == SideToMove::White) ? (result.score > best_score)
                                                          : (result.score < best_score);
      if (improving) {
        best_result = result;
        best_score = result.score;
      }

      const bool can_extend = multi_pv && pv_generated < params.pv_count;
      if (can_extend && excluded_count < excluded_moves.size()) {
        excluded_moves[excluded_count++] = encode_move(
            result.best_move.from, result.best_move.to, result.best_move.moving_pc,
            result.best_move.captured_pc, result.best_move.promo_pc, result.best_move.flags);
        aspiration_window = ASPIRATION_WINDOW_INITIAL;
        continue;
      }

      aspiration_window = window;
      break;
    }
  }
  if (history) {
    history->ply_count = base_ply;
  }

  SearchResult final_result =
      best_result.best_move.moving_pc != OccupancyType::empty ? best_result : result;
  final_result.nodes = nodes;
  return final_result;
}

namespace {
int default_evaluator(const Board& board) {
  return evaluate_board(board);
}
} // namespace

SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params) {
  const EvaluatorFn evaluator = default_evaluator;
  return search_position(board, stm, params, evaluator, nullptr, nullptr);
}

} // namespace chess
