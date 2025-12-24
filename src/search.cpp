#include "chess/search.hpp"

#include "chess/moves.hpp"
#include "chess/quiescence.hpp"
#include "chess/score.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/time_manager.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

namespace chess {
namespace {

constexpr int kMaxAspirationAttempts = 24;
constexpr int kHistoryMax = 1'000'000;

struct SearchScratch {
  MoveHistory* history = nullptr;
  KillerTable* killers = nullptr;
  TranspositionTable* tt = nullptr;
  std::array<std::array<int, 64>, 64> history_heuristic = {};
  TimeManager* time_manager = nullptr;
  bool* abort_requested = nullptr;
};

constexpr std::uint64_t kTimeCheckMask = 0x3FFULL;

inline bool should_abort_due_to_time(SearchScratch& scratch,
                                     std::uint64_t nodes) {
  if (scratch.abort_requested && *scratch.abort_requested) {
    return true;
  }
  if (!scratch.time_manager || !scratch.time_manager->enabled()) {
    return false;
  }
  if ((nodes & kTimeCheckMask) != 0) {
    return false;
  }
  if (!scratch.time_manager->hard_limit_reached()) {
    return false;
  }
  if (scratch.abort_requested) {
    *scratch.abort_requested = true;
  }
  return true;
}

inline SearchResult make_aborted_result() {
  SearchResult aborted{};
  aborted.aborted = true;
  return aborted;
}

inline void store_heuristic(SearchScratch& scratch, const Move& move,
                            int depth) {
  const bool is_quiet = (move.captured_pc == OccupancyType::empty) &&
                        (move.promo_pc == OccupancyType::empty) &&
                        !flag_is_ep(move.flags) && !flag_is_castle(move.flags) &&
                        !flag_is_long_castle(move.flags);
  if (!is_quiet) {
    return;
  }

  const std::size_t from = static_cast<std::size_t>(move.from);
  const std::size_t to = static_cast<std::size_t>(move.to);
  int& entry = scratch.history_heuristic[from][to];

  const int bonus = depth * depth;
  entry = std::min(kHistoryMax, entry + bonus);
}

inline void store_killer_move(SearchScratch& scratch, int ply, const Move& move,
                              const Undo& undo) {
  if (scratch.killers == nullptr || ply < 0 || ply >= MAX_PLY) {
    return;
  }
  const bool quiet = (undo.captured_pc == OccupancyType::empty) &&
                     (move.promo_pc == OccupancyType::empty);
  if (!quiet) {
    return;
  }

  const uint32_t code = encode_move(move.from, move.to, move.moving_pc,
                                    move.captured_pc, move.promo_pc, move.flags);

  auto& primary = scratch.killers->primary[static_cast<std::size_t>(ply)];
  auto& secondary = scratch.killers->secondary[static_cast<std::size_t>(ply)];

  if (primary == code) {
    return;
  }
  if (secondary == code) {
    std::swap(primary, secondary);
    return;
  }
  secondary = primary;
  primary = code;
}

SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta,
                               SideToMove stm, const EvaluatorFn& evaluator,
                               SearchScratch& scratch, int ply,
                               int repetition_start, int ply_from_root,
                               bool is_pv, std::uint64_t& nodes,
                               const uint32_t* excluded_root_moves,
                               std::size_t excluded_root_count) {
  ++nodes;

  if (should_abort_due_to_time(scratch, nodes)) {
    return make_aborted_result();
  }

  MoveHistory* history = scratch.history;
  TranspositionTable* tt = scratch.tt;

  const bool apply_root_exclusions = excluded_root_moves != nullptr &&
                                     excluded_root_count > 0 &&
                                     ply_from_root == 0;

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
    const uint32_t encoded =
        encode_move(move.from, move.to, move.moving_pc, move.captured_pc,
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
        if (history->key_history[static_cast<std::size_t>(i)] ==
            board.position_key) {
          ++repeats;
        }
      }
      history->repetition_counts[static_cast<std::size_t>(ply)] = repeats;
      if (repeats >= 2) {
        const int repetition_score = (stm == SideToMove::White)
                                         ? -REPETITION_PENALTY
                                         : REPETITION_PENALTY;
        return {repetition_score, Move{},
                SearchResult::Outcome::DrawByRepetition};
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
      const int cached_score = normalize_mate_score(
          TranspositionTable::decode_score(cached_entry.score, ply), ply);
      if (cached_entry.depth >= depth) {
        switch (cached_entry.flag) {
        case TranspositionFlag::Exact:
          return {cached_score, cached_entry.best_move,
                  SearchResult::Outcome::InProgress};
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
          return {cached_score, cached_entry.best_move,
                  SearchResult::Outcome::InProgress};
        }
      }
      if (cached_entry.best_move.moving_pc != OccupancyType::empty) {
        has_cached_move = true;
        cached_move = cached_entry.best_move;
      }
    }
  }

  if (allow_null_move(board, depth)) {
    UndoNull undo_null = make_null_move(board);
    const int null_move_reduction = NULL_MOVE_REDUCTION;
    SearchResult null_result = alphabeta_minimax(
        board, depth - 1 - null_move_reduction, beta - 1, beta, flip_side(stm),
        evaluator, scratch, ply + 1, repetition_start, ply_from_root + 1, false,
        nodes, nullptr, 0);
    if (null_result.aborted) {
      return null_result;
    }
    const int score_after_null = normalize_mate_score(null_result.score, ply);
    undo_null_move(board, undo_null);
    if (score_after_null >= beta) {
      return {score_after_null, Move{}, SearchResult::Outcome::InProgress};
    }
  }
  alpha_base = alpha;
  beta_base = beta;

  bool do_quiescence = true;
  if (depth == 0) {

    int qs_raw = 0;
    if (do_quiescence) {
      qs_raw = quiescence(board, alpha, beta, stm, evaluator, nodes, tt, ply);
    } else {
      int eval = evaluator(static_cast<const Board&>(board));
      qs_raw = eval;
    }
    const int qs = normalize_mate_score(qs_raw, ply);
    if (tt) {
      TranspositionEntry entry;
      if (tt->probe(board.position_key, entry)) {
        if (entry.depth <= 0) {
          tt->store(board.position_key, 0, qs, entry.flag, entry.best_move, ply);
        }
      } else {
        tt->store(board.position_key, 0, qs, TranspositionFlag::Exact, Move{},
                  ply);
      }
    }
    return {qs, Move{}, SearchResult::Outcome::InProgress};
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
      const int mate_score = normalize_mate_score(
          (stm == SideToMove::White) ? -MATE_VALUE : MATE_VALUE, ply);
      if (tt) {
        tt->store(board.position_key, depth, mate_score,
                  TranspositionFlag::Exact, Move{}, ply);
      }
      return {mate_score, Move{}, SearchResult::Outcome::Mate};
    }
    constexpr int draw_score = 0;
    if (tt) {
      tt->store(board.position_key, depth, draw_score, TranspositionFlag::Exact,
                Move{}, ply);
    }
    return {draw_score, Move{}, SearchResult::Outcome::DrawByStalemate};
  }

  uint32_t tt_code = 0;
  if (has_cached_move) {
    if (is_excluded_move(cached_move)) {
      has_cached_move = false;
    } else {
      tt_code = encode_move(cached_move.from, cached_move.to,
                            cached_move.moving_pc, cached_move.captured_pc,
                            cached_move.promo_pc, cached_move.flags);
    }
  }
  sort_moves(board, moves, move_count, tt_code, scratch.killers,
             &scratch.history_heuristic, ply);

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
    const int next_repetition_start =
        irreversible ? (ply + 1) : repetition_start;

    const bool in_check_after_move = is_check(board, flip_side(stm));
    if (child_depth == 0 && in_check_after_move &&
        ply < static_cast<int>(MAX_PLY) - 1) {
      child_depth = 1;
    }

    int reduction = 0;
    const bool is_capture = undo.captured_pc != OccupancyType::empty;
    if (child_depth > 2 && depth >= 0 && moves_tried > 3 && !is_capture &&
        !in_check_after_move) {
      reduction = 1;
    }

    int search_depth = std::max(0, child_depth);

    auto run_search = [&](int depth_to_use, int a_val, int b_val,
                          bool pv_flag) -> SearchResult {
      SearchResult res =
          alphabeta_minimax(board, depth_to_use, a_val, b_val, flip_side(stm),
                            evaluator, scratch, ply + 1, next_repetition_start,
                            ply_from_root + 1, pv_flag, nodes, nullptr, 0);
      res.score = normalize_mate_score(res.score, ply);
      return res;
    };

    bool need_full_search = true;
    SearchResult child = {};
    if (reduction > 0) {
      const int reduced_depth = std::max(0, search_depth - reduction);
      child = run_search(reduced_depth, alpha, beta, false);
      if (child.aborted) {
        undo_move(board, undo);
        return child;
      }
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
      use_pvs = is_pv && !is_first_move && (search_depth >= 0) &&
                (alpha > -MATE_BOUND) && beta < MATE_BOUND && (beta - alpha) > 1;
      if (stm == SideToMove::White) {
        narrow_beta = std::min(beta, alpha + 1);
        use_pvs = use_pvs && narrow_alpha <= narrow_beta;
      } else {
        narrow_alpha = std::max(alpha, beta - 1);
        use_pvs = use_pvs && narrow_beta >= narrow_alpha;
      }
      if (use_pvs) {
        child = run_search(search_depth, narrow_alpha, narrow_beta, true);
        if (child.aborted) {
          undo_move(board, undo);
          return child;
        }
        if (child.score >= narrow_beta) {
          child = run_search(search_depth, alpha, beta, is_pv);
          if (child.aborted) {
            undo_move(board, undo);
            return child;
          }
        }
      } else {
        child = run_search(search_depth, alpha, beta, is_pv && is_first_move);
        if (child.aborted) {
          undo_move(board, undo);
          return child;
        }
      }
    }

    const int score = child.score;

    undo_move(board, undo);

    if (ply + 1 < MAX_PLY && history) {
      history->repetition_counts[static_cast<std::size_t>(ply + 1)] = 0;
    }

    const bool is_better = (best.best_move.moving_pc == OccupancyType::empty)
                               ? true
                           : (stm == SideToMove::White) ? (score > best.score)
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
        store_killer_move(scratch, ply, move, undo);
        store_heuristic(scratch, move, depth);
        break;
      }
    } else {
      if (score < beta) {
        beta = score;
      }
      if (beta <= alpha) {
        store_killer_move(scratch, ply, move, undo);
        store_heuristic(scratch, move, depth);
        break;
      }
    }
  }

  if (best.aborted) {
    return best;
  }

  if (tt) {
    const int normalized_best = normalize_mate_score(best.score, ply);
    best.score = normalized_best;

    TranspositionFlag flag = TranspositionFlag::Exact;
    if (normalized_best <= alpha_base) {
      flag = TranspositionFlag::UpperBound;
    } else if (normalized_best >= beta_base) {
      flag = TranspositionFlag::LowerBound;
    }
    tt->store(board.position_key, depth, normalized_best, flag, best.best_move,
              ply);
  }
  // TODO: clean up transposition table after lots of searches?

  return best;
}

} // namespace

SearchResult search_position(Board& board, SideToMove stm,
                             const SearchParameters& params,
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

  KillerTable killers{};
  bool abort_requested = false;
  SearchScratch scratch{};
  scratch.history = history;
  scratch.killers = &killers;
  scratch.tt = tt;
  scratch.time_manager = params.time_manager;
  scratch.abort_requested = &abort_requested;

  int max_root_moves = 0;
  uint16_t move_count = 0;

  if (history) {
    const int repeat_begin = std::max(0, std::min(repetition_start, start_ply));
    int repeats = 0;
    for (int idx = repeat_begin; idx < start_ply; ++idx) {
      if (history->key_history[static_cast<std::size_t>(idx)] ==
          board.position_key) {
        ++repeats;
      }
    }
    if (repeats >= 2) {
      SearchResult draw{};
      draw.score = 0;
      draw.best_move = Move{};
      draw.outcome = SearchResult::Outcome::DrawByRepetition;
      draw.nodes = 0;
      draw.elapsed_ms = 0;
      return draw;
    }
  }

  generate_legal_moves(board, stm, move_count);
  if (move_count == 0) {
    const bool side_in_check = is_check(board, stm);
    SearchResult terminal{};
    terminal.best_move = Move{};
    terminal.nodes = 0;
    terminal.elapsed_ms = 0;
    if (side_in_check) {
      const int mate_score = normalize_mate_score(
          (stm == SideToMove::White) ? -MATE_VALUE : MATE_VALUE, start_ply);
      terminal.score = mate_score;
      terminal.outcome = SearchResult::Outcome::Mate;
    } else {
      terminal.score = 0;
      terminal.outcome = SearchResult::Outcome::DrawByStalemate;
    }
    return terminal;
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
      if (current_depth == 1 ||
          best_result.best_move.moving_pc == OccupancyType::empty) {
        alpha = -INF;
        beta = INF;
      } else {
        if (is_mate_score(best_score)) {
          alpha = -INF;
          beta = INF;
          aspiration_window = ASPIRATION_WINDOW_INITIAL;
        } else {
          const int window_low = best_score - aspiration_window;
          const int window_high = best_score + aspiration_window;
          alpha = std::max(window_low, -MATE_BOUND);
          alpha = std::min(alpha, MATE_BOUND);
          beta = std::min(window_high, MATE_BOUND);
          beta = std::max(beta, -MATE_BOUND);
          if (alpha >= beta) {
            alpha = window_low;
            beta = window_high;
            if (alpha >= beta) {
              alpha = -INF;
              beta = INF;
            }
          }
        }
      }

      int window = aspiration_window;
      int attempts = 0;
      bool forced_full_window = false;
      while (true) {
        ++attempts;
        const bool is_pv = true;
        result = alphabeta_minimax(
            board, current_depth, alpha, beta, stm, evaluator, scratch,
            start_ply, repetition_start, 0, is_pv, nodes,
            excluded_count ? excluded_moves.data() : nullptr, excluded_count);
        result.searched_depth = current_depth;
        result.selective_depth = current_depth;
        if (result.aborted) {
          abort_requested = true;
          break;
        }
        if (result.score <= alpha) {
          if (!forced_full_window && attempts >= kMaxAspirationAttempts) {
            alpha = -INF;
            beta = INF;
            window = ASPIRATION_WINDOW_INITIAL;
            forced_full_window = true;
            attempts = 0;
            continue;
          }
          if (is_mate_score(result.score)) {
            if (!forced_full_window) {
              alpha = -INF;
              beta = INF;
              window = ASPIRATION_WINDOW_INITIAL;
              forced_full_window = true;
              attempts = 0;
              continue;
            }
            break;
          }
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
          if (!forced_full_window && attempts >= kMaxAspirationAttempts) {
            alpha = -INF;
            beta = INF;
            window = ASPIRATION_WINDOW_INITIAL;
            forced_full_window = true;
            attempts = 0;
            continue;
          }
          if (is_mate_score(result.score)) {
            if (!forced_full_window) {
              alpha = -INF;
              beta = INF;
              window = ASPIRATION_WINDOW_INITIAL;
              forced_full_window = true;
              attempts = 0;
              continue;
            }
            break;
          }
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

      if (result.best_move.moving_pc == OccupancyType::empty ||
          !max_root_moves) {
        if (best_result.best_move.moving_pc == OccupancyType::empty) {
          best_result = result;
          best_score = result.score;
        }
        aspiration_window = window;
        break;
      }

      ++pv_generated;
      const bool improving =
          best_result.best_move.moving_pc == OccupancyType::empty ? true
          : (stm == SideToMove::White) ? (result.score > best_score)
                                       : (result.score < best_score);
      if (improving) {
        best_result = result;
        best_score = result.score;
      }

      const bool can_extend = multi_pv && pv_generated < params.pv_count;
      if (can_extend && excluded_count < excluded_moves.size()) {
        excluded_moves[excluded_count++] =
            encode_move(result.best_move.from, result.best_move.to,
                        result.best_move.moving_pc, result.best_move.captured_pc,
                        result.best_move.promo_pc, result.best_move.flags);
        aspiration_window = ASPIRATION_WINDOW_INITIAL;
        continue;
      }

      aspiration_window = window;
      break;
    }

    if (abort_requested) {
      break;
    }

    if (scratch.time_manager && scratch.time_manager->soft_limit_reached()) {
      break;
    }
  }
  if (history) {
    history->ply_count = base_ply;
  }

  SearchResult final_result =
      best_result.best_move.moving_pc != OccupancyType::empty ? best_result
                                                              : result;
  if (final_result.best_move.moving_pc == OccupancyType::empty) {
    uint16_t fallback_count = 0;
    auto fallback_moves = generate_legal_moves(board, stm, fallback_count);
    if (fallback_count > 0) {
      final_result.best_move = decode_move(fallback_moves[0]);
    }
  }
  final_result.score = normalize_mate_score(final_result.score, start_ply);
  final_result.nodes = nodes;
  final_result.aborted = abort_requested;
  return final_result;
}

namespace {
int default_evaluator(const Board& board) {
  return evaluate_board(board);
}
} // namespace

SearchResult search_position(Board& board, SideToMove stm,
                             const SearchParameters& params) {
  const EvaluatorFn evaluator = default_evaluator;
  return search_position(board, stm, params, evaluator, nullptr, nullptr);
}

} // namespace chess
