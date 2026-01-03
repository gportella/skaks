#include "chess/quiescence.hpp"

#include "chess/moves.hpp"
#include "chess/nnue_incremental.hpp"
#include "chess/piece_values.hpp"
#include "chess/score.hpp"
#include "chess/search_params.hpp"

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

} // namespace

int quiescence(Board& board, int alpha, int beta, SideToMove stm,
               const EvaluatorFn& evaluator, std::uint64_t& nodes,
               TranspositionTable* tt, int ply) {
  ++nodes;

  const auto& sparams = search_params();

  if (ply >= sparams.quiescence_max_ply) {
    return evaluator(static_cast<const Board&>(board));
  }

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

  const int alpha_origin = alpha;
  const int beta_origin = beta;
  const bool in_check = is_check(board, stm);
  const int stand_eval = evaluator(static_cast<const Board&>(board));
  const int stand_pat = stand_eval;
  const bool maximizing = (stm == SideToMove::White);

  if (!in_check) {
    if (maximizing) {
      if (stand_pat >= beta) {
        if (tt)
          tt->store(board.position_key, 0, stand_pat,
                    TranspositionFlag::LowerBound, Move{}, ply);
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
        return stand_pat;
      }
      if (stand_pat < beta) {
        beta = stand_pat;
      }
    }
  }

  uint16_t move_count = 0;
  std::array<uint32_t, kMaxMovementCount> moves{};
  if (in_check) {
    moves = generate_legal_moves(board, stm, move_count);
  } else {
    moves = generate_all_moves(board, stm, move_count);
    uint16_t write_idx = 0;
    for (uint16_t i = 0; i < move_count; ++i) {
      Move m = decode_move(moves[i]);
      const bool is_capture = m.captured_pc != OccupancyType::empty;
      const bool is_promo = m.promo_pc != OccupancyType::empty;
      if (!(is_capture || is_promo)) {
        continue;
      }
      Undo u = make_move(board, m);
      const bool legal = !is_check(board, flip_side(board.side_to_move));
      undo_move(board, u);
      if (legal) {
        moves[write_idx++] = moves[i];
      }
    }
    move_count = write_idx;
  }

  sort_moves(board, moves, move_count, 0);

  // Hard cap noisy moves per node when not in check to avoid explosion
  if (!in_check && move_count > sparams.quiescence_max_noisy_moves) {
    move_count = static_cast<uint16_t>(sparams.quiescence_max_noisy_moves);
  }

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

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);

    if (!in_check) {
      // found that the simple capture gain helps a lot in pruning
      // more that SEE, and seems like the quality is a tad better too...
      const int delta_gain = capture_gain(move);

      if (i >= sparams.quiescence_zero_gain_skip_index && delta_gain == 0) {
        continue;
      }

      const bool non_promo_pawn_capture =
          (move.moving_pc == OccupancyType::wP ||
           move.moving_pc == OccupancyType::bP) &&
          move.promo_pc == OccupancyType::empty;
      if (non_promo_pawn_capture) {
        const int pawn_value = piece_material_magnitude(move.moving_pc);
        if (delta_gain + sparams.quiescence_delta_margin < pawn_value) {
          continue;
        }
      }

      if (maximizing) {
        if (stand_pat + delta_gain + sparams.quiescence_delta_margin <= alpha) {
          continue;
        }
      } else {
        if (stand_pat - delta_gain - sparams.quiescence_delta_margin >= beta) {
          continue;
        }
      }
    }

    Undo undo = make_move(board, move);
    SfNnueStack* nnue_stack = current_thread_nnue_stack();
    if (nnue_stack) {
      nnue_stack->push_move(board, move, undo);
    }
    const int score = quiescence(board, alpha, beta, flip_side(stm), evaluator,
                                 nodes, tt, ply + 1);
    undo_move(board, undo);
    if (nnue_stack) {
      nnue_stack->pop();
    }

    if (maximizing) {
      if (score >= beta) {
        if (tt)
          tt->store(board.position_key, 0, score, TranspositionFlag::LowerBound,
                    move, ply);
        return score;
      }
      if (!has_best || score > best_score) {
        best_score = score;
        best_move = move;
        has_best = true;
      }
      if (score > alpha) {
        alpha = score;
      }
    } else {
      if (score <= alpha) {
        if (tt)
          tt->store(board.position_key, 0, score, TranspositionFlag::UpperBound,
                    move, ply);
        return score;
      }
      if (!has_best || score < best_score) {
        best_score = score;
        best_move = move;
        has_best = true;
      }
      if (score < beta) {
        beta = score;
      }
    }
  }

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