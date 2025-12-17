#include "chess/quiescence.hpp"

#include "chess/moves.hpp"
#include "chess/piece_values.hpp"
#include "chess/score.hpp"

#include <algorithm>

namespace chess {

namespace {

inline int capture_gain(const Move& move) {
  int gain = 0;
  if (move.captured_pc != OccupancyType::empty) {
    gain += piece_material_magnitude(move.captured_pc);
  }
  if (move.promo_pc != OccupancyType::empty) {
    const int promo_gain =
        piece_material_magnitude(move.promo_pc) - piece_material_magnitude(move.moving_pc);
    gain += std::max(promo_gain, 0);
  }
  return gain;
}

} // namespace

int quiescence(Board& board, int alpha, int beta, SideToMove stm, const EvaluatorFn& evaluator,
               std::uint64_t& nodes, TranspositionTable* tt, int ply) {
  ++nodes;

  if (ply >= QUIESCENCE_MAX_PLY) {
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

  const bool in_check = is_check(board, stm);
  const int stand_pat = evaluator(static_cast<const Board&>(board));

  if (!in_check) {
    if (stand_pat >= beta) {
      if (tt)
        tt->store(board.position_key, 0, stand_pat, TranspositionFlag::LowerBound, Move{}, ply);
      return stand_pat;
    }
    if (stand_pat > alpha) {
      alpha = stand_pat;
    }
  }

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);

  // Filter to noisy moves when not in check
  if (!in_check) {
    uint16_t w = 0;
    for (uint16_t i = 0; i < move_count; ++i) {
      Move m = decode_move(moves[i]);
      const bool is_capture = m.captured_pc != OccupancyType::empty;
      const bool is_promo = m.promo_pc != OccupancyType::empty;
      if (is_capture || is_promo) {
        moves[w++] = moves[i];
      }
    }
    move_count = w;
  }

  // Sort captures/promotions; you can use MVV-LVA inside sort_moves
  sort_moves(moves, move_count, 0);

  // Hard cap noisy moves per node when not in check to avoid explosion
  if (!in_check && move_count > QUIESCENCE_MAX_NOISY_MOVES) {
    move_count = static_cast<uint16_t>(QUIESCENCE_MAX_NOISY_MOVES);
  }

  if (move_count == 0) {
    if (in_check) {
      const int mate = (stm == SideToMove::White) ? -MATE_VALUE : MATE_VALUE;
      if (tt)
        tt->store(board.position_key, 0, mate, TranspositionFlag::Exact, Move{}, ply);
      return mate;
    }
    if (tt)
      tt->store(board.position_key, 0, stand_pat, TranspositionFlag::Exact, Move{}, ply);
    return stand_pat;
  }

  Move best_move{};
  bool has_best = false;

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);

    if (!in_check) {
      const int delta_gain = capture_gain(move);

      if (i >= QUIESCENCE_ZERO_GAIN_SKIP_INDEX && delta_gain == 0) {
        continue;
      }

      const int futility_limit =
          (stm == SideToMove::White) ? (alpha - stand_pat - QUIESCENCE_DELTA_MARGIN)
                                     : (stand_pat - beta - QUIESCENCE_DELTA_MARGIN);

      if (futility_limit >= 0 && delta_gain <= futility_limit) {
        continue;
      }
    }

    Undo undo = make_move(board, move);
    const int score =
        -quiescence(board, -beta, -alpha, flip_side(stm), evaluator, nodes, tt, ply + 1);
    undo_move(board, undo);

    if (score >= beta) {
      if (tt)
        tt->store(board.position_key, 0, score, TranspositionFlag::LowerBound, move, ply);
      return score;
    }
    if (score > alpha) {
      alpha = score;
      best_move = move;
      has_best = true;
    }
  }

  TranspositionFlag flag =
      (!in_check && has_best) ? TranspositionFlag::Exact : TranspositionFlag::UpperBound;
  if (tt)
    tt->store(board.position_key, 0, alpha, flag, has_best ? best_move : Move{}, ply);
  return alpha;
}

} // namespace chess