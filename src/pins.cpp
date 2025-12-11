/*
 Wonderful world of pins

 Plan of attack
 ================

find_file_rank_pins(Board, ksq, my_occ, their_occ) → returns a list or map of pinned piece squares
with allowed line masks for orthogonal directions (rook/queen)

find_diagonal_pins(Board, ksq, my_occ, their_occ) → same for diagonals (bishop/queen)

Pin aggregation

build_pinned_map(pins_ortho, pins_diag) → a structure mapping pinned piece square → allowed_mask
Pin masks by piece type

allowed_mask_for_bishop(square, pinned_map) → Bitboard (line mask or full if not pinned)
allowed_mask_for_rook(square, pinned_map) → Bitboard
allowed_mask_for_queen(square, pinned_map) → Bitboard
allowed_mask_for_knight(square, pinned_map) → Bitboard (empty if pinned)
allowed_mask_for_pawn(square, pinned_map, SideToMove) → Bitboard with special-case rules:
file pin: allow only forward squares on that file
diagonal pin: allow only the aligned capture square
rank pin: none
Check-resolution mask (optional, separate)

compute_check_mask(Board, SideToMove) → Bitboard legal_target_mask
If double check: only king moves allowed
Else: union of checker square and block squares along the check line
Apply: moves &= legal_target_mask for non-king pieces
 */

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/types_io.hpp"

namespace chess {

/**
 * Find orthogonal pins for the given side to move, solves for rooks and queens
 * @param[in] board struct
 * @param[in] side to move
 * @return array of pairs of pinned square and line mask
 */
std::array<std::pair<Square, Bitboard>, 8> find_orthogonal_pins_unrolled(const Board& board,
                                                                         SideToMove stm) {
  bool white = (stm == SideToMove::White);
  std::array<std::pair<Square, Bitboard>, 8> pins{};
  std::size_t pin_count = 0;

  u_int8_t king_sq =
      static_cast<u_int8_t>(white ? board.king_list[0].squares[0] : board.king_list[1].squares[0]);

  const auto& rays = ROOK_RAYS[king_sq];

  // north
  auto north = rays.north;
  auto north_blockers =
      north & (board.occupancy[to_index(stm)] | board.occupancy[to_index(flip_side(stm))]);

  if (north_blockers) {
    int first_blocker_sq = lsb_index(north_blockers);
    const auto& first_rays = ROOK_RAYS[static_cast<u_int8_t>(first_blocker_sq)];
    Bitboard after_first = first_rays.north;
    Bitboard second_blockers = after_first & board.occupancy[to_index(flip_side(stm))];

    if (second_blockers) {
      int second_blocker_sq = lsb_index(second_blockers);
      // Check if the piece at second_blocker_sq is a rook or queen
      OccupancyType occ = board.pieces[to_index(second_blocker_sq)];
      if ((white && (occ == OccupancyType::bR || occ == OccupancyType::bQ)) ||
          (!white && (occ == OccupancyType::wR || occ == OccupancyType::wQ))) {
        // Found a pin
        const auto& attacker_rays = ROOK_RAYS[static_cast<u_int8_t>(second_blocker_sq)];
        Bitboard beyond_attacker = attacker_rays.north;
        Square pinned_sq = static_cast<Square>(first_blocker_sq);
        Bitboard line_mask = ROOK_RAYS[king_sq].north & beyond_attacker;
        pins[pin_count++] = {pinned_sq, line_mask};
      }
    }
  }

  // south
  auto south = rays.south;
  auto south_blockers =
      south & (board.occupancy[to_index(stm)] | board.occupancy[to_index(flip_side(stm))]);

  if (south_blockers) {
    int first_blocker_sq = msb_index(south_blockers);
    const auto& first_rays = ROOK_RAYS[static_cast<u_int8_t>(first_blocker_sq)];
    Bitboard after_first = first_rays.south;
    Bitboard second_blockers = after_first & board.occupancy[to_index(flip_side(stm))];

    if (second_blockers) {
      int second_blocker_sq = msb_index(second_blockers);
      // Check if the piece at second_blocker_sq is a rook or queen
      OccupancyType occ = board.pieces[to_index(second_blocker_sq)];
      if ((white && (occ == OccupancyType::bR || occ == OccupancyType::bQ)) ||
          (!white && (occ == OccupancyType::wR || occ == OccupancyType::wQ))) {
        // Found a pin
        const auto& attacker_rays = ROOK_RAYS[static_cast<u_int8_t>(second_blocker_sq)];
        Bitboard beyond_attacker = attacker_rays.south;
        Square pinned_sq = static_cast<Square>(first_blocker_sq);
        Bitboard line_mask = ROOK_RAYS[king_sq].south & beyond_attacker;
        pins[pin_count++] = {pinned_sq, line_mask};
      }
    }
  }
  // east
  auto east = rays.east;
  auto east_blockers =
      east & (board.occupancy[to_index(stm)] | board.occupancy[to_index(flip_side(stm))]);

  if (east_blockers) {
    int first_blocker_sq = lsb_index(east_blockers);
    const auto& first_rays = ROOK_RAYS[static_cast<u_int8_t>(first_blocker_sq)];
    Bitboard after_first = first_rays.east;
    Bitboard second_blockers = after_first & board.occupancy[to_index(flip_side(stm))];

    if (second_blockers) {
      int second_blocker_sq = lsb_index(second_blockers);
      // Check if the piece at second_blocker_sq is a rook or queen
      OccupancyType occ = board.pieces[to_index(second_blocker_sq)];
      if ((white && (occ == OccupancyType::bR || occ == OccupancyType::bQ)) ||
          (!white && (occ == OccupancyType::wR || occ == OccupancyType::wQ))) {
        // Found a pin
        const auto& attacker_rays = ROOK_RAYS[static_cast<u_int8_t>(second_blocker_sq)];
        Bitboard beyond_attacker = attacker_rays.east;
        Square pinned_sq = static_cast<Square>(first_blocker_sq);
        Bitboard line_mask = ROOK_RAYS[king_sq].east & beyond_attacker;
        pins[pin_count++] = {pinned_sq, line_mask};
      }
    }
  }
  // west
  auto west = rays.west;
  auto west_blockers =
      west & (board.occupancy[to_index(stm)] | board.occupancy[to_index(flip_side(stm))]);

  if (west_blockers) {
    int first_blocker_sq = msb_index(west_blockers);
    const auto& first_rays = ROOK_RAYS[static_cast<u_int8_t>(first_blocker_sq)];
    Bitboard after_first = first_rays.west;
    Bitboard second_blockers = after_first & board.occupancy[to_index(flip_side(stm))];

    if (second_blockers) {
      int second_blocker_sq = msb_index(second_blockers);
      // Check if the piece at second_blocker_sq is a rook or queen
      OccupancyType occ = board.pieces[to_index(second_blocker_sq)];
      if ((white && (occ == OccupancyType::bR || occ == OccupancyType::bQ)) ||
          (!white && (occ == OccupancyType::wR || occ == OccupancyType::wQ))) {
        // Found a pin
        const auto& attacker_rays = ROOK_RAYS[static_cast<u_int8_t>(second_blocker_sq)];
        Bitboard beyond_attacker = attacker_rays.west;
        Square pinned_sq = static_cast<Square>(first_blocker_sq);
        Bitboard line_mask = ROOK_RAYS[king_sq].west & beyond_attacker;
        pins[pin_count++] = {pinned_sq, line_mask};
      }
    }
  }

  return pins;
}

enum class Edge { Forward, Backward };

struct OrthoDir {
  Bitboard chess::RookRays::* ray; // king → direction
  Edge first_blocker_edge;         // do we walk “up” (lsb) or “down” (msb)?
};

constexpr std::array<OrthoDir, 4> kOrthoDirs{{
    {&chess::RookRays::north, Edge::Forward},  // squares increase as you move north → lsb_index
    {&chess::RookRays::south, Edge::Backward}, // squares decrease → msb_index
    {&chess::RookRays::east, Edge::Forward},   // increasing file → lsb_index
    {&chess::RookRays::west, Edge::Backward},  // decreasing file → msb_index
}};

bool is_rook_or_queen_of_enemy(OccupancyType occ, SideToMove stm) {
  bool white = (stm == SideToMove::White);
  if (white) {
    return occ == OccupancyType::bR || occ == OccupancyType::bQ;
  } else {
    return occ == OccupancyType::wR || occ == OccupancyType::wQ;
  }
}

std::array<std::pair<Square, Bitboard>, 8> find_orthogonal_pins(const Board& board,
                                                                SideToMove stm) {

  auto pick_index = [&](Bitboard mask, Edge edge) {
    return edge == Edge::Forward ? to_index(lsb_index(mask)) : to_index(msb_index(mask));
  };

  bool white = (stm == SideToMove::White);
  std::array<std::pair<Square, Bitboard>, 8> pins{};
  std::size_t pin_count = 0;

  u_int8_t king_sq =
      static_cast<u_int8_t>(white ? board.king_list[0].squares[0] : board.king_list[1].squares[0]);

  const auto& king_rays = ROOK_RAYS[king_sq];
  const Bitboard my_occ = board.occupancy[to_index(stm)];
  const Bitboard their_occ = board.occupancy[to_index(flip_side(stm))];

  for (const auto& dir : kOrthoDirs) {
    const Bitboard ray = king_rays.*(dir.ray);
    Bitboard blockers = ray & (my_occ | their_occ);
    if (!blockers)
      continue;

    auto first_sq = pick_index(blockers, dir.first_blocker_edge);
    const auto& first_rays = ROOK_RAYS[first_sq];

    const Bitboard beyond_first = first_rays.*(dir.ray);
    if (!beyond_first)
      continue;

    const Bitboard next = beyond_first & (my_occ | their_occ);
    if (!next)
      continue;

    auto next_sq = pick_index(next, dir.first_blocker_edge);
    const OccupancyType occ = board.pieces[next_sq];
    if (!is_rook_or_queen_of_enemy(occ, stm))
      continue;

    Bitboard beyond_attacker = ROOK_RAYS[next_sq].*(dir.ray);
    Bitboard line_mask = ray & ~beyond_attacker;
    Square pinned_sq = static_cast<Square>(first_sq);
    if (pin_count < pins.size()) { // guard against overflow when multiple pins appear
      pins[pin_count++] = {pinned_sq, line_mask};
    } else {
      std::cerr << "Warning: exceeded pin storage capacity\n" << std::endl;
    }
  }

  return pins;
}
// std::array<std::pair<Square, Bitboard>, 8> find_diagonal_pins(const Board& board, SideToMove stm)
// {

//   bool white = (stm == SideToMove::White);
//   std::array<std::pair<Square, Bitboard>, 8> pins{};
//   std::size_t pin_count = 0;
//   u_int8_t king_sq =
//       static_cast<u_int8_t>(white ? board.king_list[0].squares[0] :
//       board.king_list[1].squares[0]);

//   const auto& rays = BISHOP_RAYS[king_sq];
// }

} // namespace chess