
#include "chess/exchange.hpp"

#include "chess/attack_masks.hpp"
#include "chess/piece_values.hpp"

namespace chess {
inline int SEE_sq(Board& b, int sq) {
  int value = 0;
  auto attacker =
      find_smallest_attacker(b, static_cast<u_int8_t>(sq), b.side_to_move);
  if (attacker.has_value()) {
    UndoSEE undo = make_see_move(
        b,
        Move{static_cast<uint16_t>(attacker->square), static_cast<uint16_t>(sq),
             b.pieces[static_cast<std::size_t>(attacker->square)],
             b.pieces[static_cast<std::size_t>(sq)], OccupancyType::empty, 0});
    b.side_to_move = flip_side(b.side_to_move);
    value = piece_material_magnitude(undo.captured_pc) - SEE_sq(b, sq);
    undo_see_move(b, undo);
    b.side_to_move = flip_side(b.side_to_move);
  }
  return value;
}

int static_exchange_eval(Board b, const Move& move) {
  const bool is_capture = move.captured_pc != OccupancyType::empty;
  const bool is_promotion = move.promo_pc != OccupancyType::empty;
  if (!is_capture && !is_promotion) {
    return 0;
  }

  const int target_sq = static_cast<int>(move.to);
  int initial_gain = 0;
  if (is_capture) {
    initial_gain = piece_material_magnitude(move.captured_pc);
  }
  if (is_promotion) {
    const int promo_delta = piece_material_magnitude(move.promo_pc) -
                            piece_material_magnitude(move.moving_pc);
    initial_gain += std::max(promo_delta, 0);
  }

  UndoSEE undo = make_see_move(b, move);
  b.side_to_move = flip_side(b.side_to_move);
  const int reply_gain = SEE_sq(b, target_sq);
  b.side_to_move = flip_side(b.side_to_move);
  undo_see_move(b, undo);

  return initial_gain - reply_gain;
}
} // namespace chess