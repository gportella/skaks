#include "chess/quiescence.hpp"

#include "chess/moves.hpp"
#include "chess/piece_values.hpp"

#include <algorithm>

namespace chess {

int quiescence(Board& board, int alpha, int beta, SideToMove stm, const EvaluatorFn& evaluator,
               std::uint64_t& nodes, TranspositionTable* tt, int ply) {
  ++nodes;

  const int alpha_origin = alpha;

  if (tt) {
    TranspositionEntry cached_entry;
    if (tt->probe(board.position_key, cached_entry)) {
      const int cached_score = TranspositionTable::decode_score(cached_entry.score, ply);
      switch (cached_entry.flag) {
      case TranspositionFlag::Exact:
        return cached_score;
      case TranspositionFlag::LowerBound:
        if (cached_score >= beta) {
          return cached_score;
        }
        alpha = std::max(alpha, cached_score);
        break;
      case TranspositionFlag::UpperBound:
        if (cached_score <= alpha) {
          return cached_score;
        }
        beta = std::min(beta, cached_score);
        break;
      case TranspositionFlag::None:
        break;
      }
      if (alpha >= beta) {
        return cached_score;
      }
    }
  }

  const bool in_check = is_check(board, stm);
  const int stand_pat = evaluator(static_cast<const Board&>(board));

  if (!in_check) {
    if (stand_pat >= beta) {
      if (tt) {
        tt->store(board.position_key, 0, stand_pat, TranspositionFlag::LowerBound, Move{}, ply);
      }
      return stand_pat;
    }
    if (stand_pat > alpha) {
      alpha = stand_pat;
    }
  }

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);

  if (!in_check) {
    uint16_t write_idx = 0;
    for (uint16_t i = 0; i < move_count; ++i) {
      Move m = decode_move(moves[i]);
      const bool is_capture = m.captured_pc != OccupancyType::empty;
      const bool is_promo = m.promo_pc != OccupancyType::empty;
      if (is_capture || is_promo) {
        moves[write_idx++] = moves[i];
      }
    }
    move_count = write_idx;
  }

  sort_moves(moves, move_count, 0);

  if (move_count == 0) {
    int terminal_score = stand_pat;
    if (in_check) {
      terminal_score = (stm == SideToMove::White) ? (-INF + ply) : (INF - ply);
    }
    if (tt) {
      tt->store(board.position_key, 0, terminal_score, TranspositionFlag::Exact, Move{}, ply);
    }
    return terminal_score;
  }

  Move best_move{};
  bool has_best = false;

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);

    if (!in_check) {
      int gain = 0;
      if (move.captured_pc != OccupancyType::empty) {
        gain += piece_material_magnitude(move.captured_pc);
      }
      if (move.promo_pc != OccupancyType::empty) {
        const int promo_gain =
            piece_material_magnitude(move.promo_pc) - piece_material_magnitude(move.moving_pc);
        gain += std::max(promo_gain, 0);
      }

      if (gain > 0) {
        if (stm == SideToMove::White) {
          if (stand_pat + gain + QUIESCENCE_DELTA_MARGIN <= alpha) {
            continue;
          }
        } else {
          if (stand_pat - gain - QUIESCENCE_DELTA_MARGIN >= beta) {
            continue;
          }
        }
      }
    }

    Undo undo = make_move(board, move);

    const int score =
        -quiescence(board, -beta, -alpha, flip_side(stm), evaluator, nodes, tt, ply + 1);

    undo_move(board, undo);

    if (score >= beta) {
      if (tt) {
        tt->store(board.position_key, 0, score, TranspositionFlag::LowerBound, move, ply);
      }
      return score;
    }
    if (score > alpha) {
      alpha = score;
      best_move = move;
      has_best = true;
    }
  }

  if (tt) {
    TranspositionFlag flag =
        (alpha > alpha_origin) ? TranspositionFlag::Exact : TranspositionFlag::UpperBound;
    tt->store(board.position_key, 0, alpha, flag, has_best ? best_move : Move{}, ply);
  }

  return alpha;
}

} // namespace chess