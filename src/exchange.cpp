
#include "chess/exchange.hpp"

#include "chess/attack_masks.hpp"
#include "chess/piece_values.hpp"

namespace chess {

int static_exchange_eval(const Board& board, const Move& move) {
  int gain[32];
  int d = 0;

  uint64_t occupancy = board.occupancy[to_index(PieceColor::Both)];
  uint64_t attackers = get_all_attackers(board, move.to, occupancy);

  // Initial capture
  gain[d] = piece_material_magnitude(move.captured_pc);

  uint64_t from_bb = 1ULL << move.from;
  SideToMove stm = flip_side(board.side_to_move);

  // Iteratively find the next smallest attacker
  while (from_bb) {
    d++;
    gain[d] = piece_material_magnitude(
                  board.pieces[static_cast<std::size_t>(lsb_index(from_bb))]) -
              gain[d - 1];

    if (std::max(-gain[d - 1], gain[d]) < 0)
      break;

    occupancy ^= from_bb;
    attackers |= get_xray_attackers(board, move.to, occupancy);
    attackers &= occupancy;

    from_bb = get_smallest_attacker_from_mask(board, attackers, stm);
    stm = flip_side(stm);
  }

  // Negamax the swap list back up
  while (--d > 0) {
    gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
  }

  return gain[0];
}

} // namespace chess