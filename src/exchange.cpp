
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
    value =
        std::max(0, piece_material_magnitude(undo.captured_pc) - SEE_sq(b, sq));
    undo_see_move(b, undo);
    b.side_to_move = flip_side(b.side_to_move);
  }
  return value;
}

int static_exchange_eval(Board b, const Move& move) {
  // we are going to skip unless we capture or promote
  if (move.captured_pc == OccupancyType::empty ||
      move.promo_pc == OccupancyType::empty) {
    auto target_sq = move.to;
    return SEE_sq(b, target_sq);
  }
  return 0;
}
} // namespace chess