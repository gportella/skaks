#include "chess/quiescence.hpp"

#include "chess/moves.hpp"
#include "chess/piece_values.hpp"
#include "chess/score.hpp"
#include "chess/search_params.hpp"
#include "chess/search_stats.hpp"

#include <algorithm>

namespace chess {

namespace {

inline int capture_gain(const Move& move) {
  int gain = 0;
  if (move.captured_pc != OccupancyType::empty) {
    gain += piece_material_magnitude(move.captured_pc);
  }
  if (move.promo_pc != OccupancyType::empty) {
    const int promo_gain = piece_material_magnitude(move.promo_pc) -
                           piece_material_magnitude(move.moving_pc);
    gain += std::max(promo_gain, 0);
  }
  return gain;
}

uint16_t collect_quiet_checks(Board& board, SideToMove stm,
                              std::array<uint32_t, kMaxMovementCount>& out,
                              NnueAdapter* nnue_adapter, uint16_t max_checks) {
  uint16_t pseudo_count = 0;
  auto pseudo = generate_all_moves(board, stm, pseudo_count);
  uint16_t kept = 0;
  for (uint16_t i = 0; i < pseudo_count && kept < max_checks; ++i) {
    Move move = decode_move(pseudo[i]);
    const bool is_capture = move.captured_pc != OccupancyType::empty;
    const bool is_promo = move.promo_pc != OccupancyType::empty;
    if (is_capture || is_promo) {
      continue;
    }

    Undo undo = make_move(board, move);
    if (nnue_adapter) {
      nnue_adapter->push_move(move);
    }
    const bool legal = !is_check(board, flip_side(stm));
    bool gives_check = false;
    if (legal) {
      gives_check = is_check(board, board.side_to_move);
    }
    undo_move(board, undo);
    if (nnue_adapter) {
      nnue_adapter->pop_move();
    }

    if (legal && gives_check) {
      out[kept++] = pseudo[i];
    }
  }
  return kept;
}

} // namespace

// The quiescence search function:
// extends search in "noisy" positions to avoid horizon effect
int quiescence(Board& board, int alpha, int beta, SideToMove stm,
               const EvaluatorFn& evaluator, NnueAdapter* nnue_adapter,
               std::uint64_t& nodes, TranspositionTable* tt, int ply) {
  ++nodes;

  if (search_stats_enabled()) {
    search_stats().quiescence_nodes.fetch_add(1, std::memory_order_relaxed);
  }

  const auto& sparams = search_params();

  if (ply >= sparams.quiescence_max_ply) {
    return evaluator(static_cast<const Board&>(board), nnue_adapter);
  }

  // TT probe: tighten window or return cached cutoff/score
  if (tt) {
    TranspositionEntry e;
    if (tt->probe(board.position_key, e)) {
      const int tscore = TranspositionTable::decode_score(e.score, ply);
      if (e.flag == TranspositionFlag::Exact) {
        return tscore;
      } else if (e.flag == TranspositionFlag::LowerBound) {
        if (tscore >= beta)
          return tscore;
        alpha = std::max(alpha, tscore);
      } else if (e.flag == TranspositionFlag::UpperBound) {
        if (tscore <= alpha)
          return tscore;
        beta = std::min(beta, tscore);
      }
      if (alpha >= beta)
        return tscore;
    }
  }

  // Track original bounds for final TT flag selection
  const int alpha_origin = alpha;
  const int beta_origin = beta;
  const bool in_check = is_check(board, stm);
  const int stand_eval =
      evaluator(static_cast<const Board&>(board), nnue_adapter);
  const int stand_pat = stand_eval;
  const bool maximizing = (stm == SideToMove::White);

  // Stand-pat evaluation and immediate cutoff if safe (not in check)
  if (!in_check) {
    if (maximizing) {
      if (stand_pat >= beta) {
        if (tt)
          tt->store(board.position_key, 0, stand_pat,
                    TranspositionFlag::LowerBound, Move{}, ply);
        if (search_stats_enabled()) {
          search_stats().quiescence_stand_pat_cutoffs.fetch_add(
              1, std::memory_order_relaxed);
        }
        return stand_pat;
      }
      if (stand_pat > alpha) {
        alpha = stand_pat;
      }
    } else {
      if (stand_pat <= alpha) {
        if (tt)
          tt->store(board.position_key, 0, stand_pat,
                    TranspositionFlag::UpperBound, Move{}, ply);
        if (search_stats_enabled()) {
          search_stats().quiescence_stand_pat_cutoffs.fetch_add(
              1, std::memory_order_relaxed);
        }
        return stand_pat;
      }
      if (stand_pat < beta) {
        beta = stand_pat;
      }
    }
  }

  // Generate candidate moves (all if in check, otherwise noisy only)
  uint16_t move_count = 0;
  std::array<uint32_t, kMaxMovementCount> moves{};
  if (in_check) {
    moves = generate_legal_moves(board, stm, move_count);
  } else {
    moves = generate_all_moves(board, stm, move_count);
    uint16_t write_idx = 0;
    // filter only legal captures and promotions
    for (uint16_t i = 0; i < move_count; ++i) {
      Move m = decode_move(moves[i]);
      const bool is_capture = m.captured_pc != OccupancyType::empty;
      const bool is_promo = m.promo_pc != OccupancyType::empty;
      if (!(is_capture || is_promo)) {
        continue;
      }
      Undo u = make_move(board, m);
      if (nnue_adapter) {
        nnue_adapter->push_move(m);
      }
      const bool legal = !is_check(board, flip_side(board.side_to_move));
      undo_move(board, u);
      if (nnue_adapter) {
        nnue_adapter->pop_move();
      }
      if (legal) {
        moves[write_idx++] = moves[i];
      }
    }
    move_count = write_idx;
  }

  // Order captures/promotions for better cutoffs
  sort_moves(board, moves, move_count, 0);

  // Hard cap noisy moves per node when not in check to avoid explosion
  if (!in_check && move_count > sparams.quiescence_max_noisy_moves) {
    move_count = static_cast<uint16_t>(sparams.quiescence_max_noisy_moves);
  }

  // If no moves, return mate/draw/stand-pat depending on check
  if (move_count == 0) {
    if (in_check) {
      const int mate = (stm == SideToMove::White) ? -MATE_VALUE : MATE_VALUE;
      if (tt)
        tt->store(board.position_key, 0, mate, TranspositionFlag::Exact, Move{},
                  ply);
      return mate;
    }
    if (tt)
      tt->store(board.position_key, 0, stand_pat, TranspositionFlag::Exact,
                Move{}, ply);
    return stand_pat;
  }

  Move best_move{};
  bool has_best = false;
  int best_score = in_check ? (maximizing ? -INF : INF) : stand_pat;

  // Apply alpha/beta updates and detect cutoffs
  auto process_score = [&](const Move& current_move, int score_value) -> bool {
    if (maximizing) {
      if (score_value >= beta) {
        if (tt)
          tt->store(board.position_key, 0, score_value,
                    TranspositionFlag::LowerBound, current_move, ply);
        return true;
      }
      if (!has_best || score_value > best_score) {
        best_score = score_value;
        best_move = current_move;
        has_best = true;
      }
      if (score_value > alpha) {
        alpha = score_value;
      }
    } else {
      if (score_value <= alpha) {
        if (tt)
          tt->store(board.position_key, 0, score_value,
                    TranspositionFlag::UpperBound, current_move, ply);
        return true;
      }
      if (!has_best || score_value < best_score) {
        best_score = score_value;
        best_move = current_move;
        has_best = true;
      }
      if (score_value < beta) {
        beta = score_value;
      }
    }
    return false;
  };

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);

    if (!in_check) {
      // Delta and zero-gain pruning for noisy moves
      // found that the simple capture gain helps a lot in pruning
      // more than SEE, and seems like the quality is a tad better too...
      // we don't ignore negative gains to avoid missing tactics
      const int delta_gain = capture_gain(move);

      if (i >= sparams.quiescence_zero_gain_skip_index && delta_gain == 0) {
        if (search_stats_enabled()) {
          search_stats().quiescence_zero_gain_skips.fetch_add(
              1, std::memory_order_relaxed);
        }
        continue;
      }

      const bool non_promo_pawn_capture =
          (move.moving_pc == OccupancyType::wP ||
           move.moving_pc == OccupancyType::bP) &&
          move.promo_pc == OccupancyType::empty;
      if (non_promo_pawn_capture) {
        const int pawn_value = piece_material_magnitude(move.moving_pc);
        if (delta_gain + sparams.quiescence_delta_margin < pawn_value) {
          if (search_stats_enabled()) {
            search_stats().quiescence_delta_prunes.fetch_add(
                1, std::memory_order_relaxed);
          }
          continue;
        }
      }

      if (maximizing) {
        if (stand_pat + delta_gain + sparams.quiescence_delta_margin <= alpha) {
          if (search_stats_enabled()) {
            search_stats().quiescence_delta_prunes.fetch_add(
                1, std::memory_order_relaxed);
          }
          continue;
        }
      } else {
        if (stand_pat - delta_gain - sparams.quiescence_delta_margin >= beta) {
          if (search_stats_enabled()) {
            search_stats().quiescence_delta_prunes.fetch_add(
                1, std::memory_order_relaxed);
          }
          continue;
        }
      }
    }

    // Recurse on tactical move
    Undo undo = make_move(board, move);
    if (nnue_adapter) {
      nnue_adapter->push_move(move);
    }
    const int score = quiescence(board, alpha, beta, flip_side(stm), evaluator,
                                 nnue_adapter, nodes, tt, ply + 1);
    undo_move(board, undo);
    if (nnue_adapter) {
      nnue_adapter->pop_move();
    }
    if (process_score(move, score)) {
      return score;
    }
  }

  // Optional quiet-check extension after noisy moves
  if (!in_check && sparams.quiescence_max_quiet_checks > 0) {
    const int quiet_cap_raw = sparams.quiescence_max_quiet_checks;
    const uint16_t quiet_cap =
        static_cast<uint16_t>(std::min<int>(quiet_cap_raw, kMaxMovementCount));
    if (quiet_cap > 0) {
      std::array<uint32_t, kMaxMovementCount> quiet_checks{};
      // not worth keeping track of quiet moves from before, recompute for now
      const uint16_t quiet_count = collect_quiet_checks(board, stm, quiet_checks,
                                                        nnue_adapter, quiet_cap);
      for (uint16_t i = 0; i < quiet_count; ++i) {
        Move move = decode_move(quiet_checks[i]);
        Undo undo = make_move(board, move);
        if (nnue_adapter) {
          nnue_adapter->push_move(move);
        }
        const int score =
            quiescence(board, alpha, beta, flip_side(stm), evaluator,
                       nnue_adapter, nodes, tt, ply + 1);
        undo_move(board, undo);
        if (nnue_adapter) {
          nnue_adapter->pop_move();
        }
        if (process_score(move, score)) {
          return score;
        }
      }
    }
  }

  // Final TT store with proper bound flag
  if (tt) {
    const int normalized_best = normalize_mate_score(best_score, ply);
    TranspositionFlag flag = TranspositionFlag::Exact;
    if (normalized_best <= alpha_origin) {
      flag = TranspositionFlag::UpperBound;
    } else if (normalized_best >= beta_origin) {
      flag = TranspositionFlag::LowerBound;
    }
    tt->store(board.position_key, 0, normalized_best, flag,
              has_best ? best_move : Move{}, ply);
    return normalized_best;
  }

  return best_score;
}

} // namespace chess