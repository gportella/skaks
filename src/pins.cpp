// SPDX-License-Identifier: GPL-3.0-or-later
/*
 Wonderful world of pins

For move generation, take these into account
Check-resolution mask (optional, separate)
compute_check_mask(Board, SideToMove) → Bitboard legal_target_mask
If double check: only king moves allowed
Else: union of checker square and block squares along the check line
Apply: moves &= legal_target_mask for non-king pieces
 */

#include "chess/pins.hpp"

#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/types_io.hpp"

namespace chess {

namespace {

constexpr std::array<OrthoDir, 4> kOrthoDirs{{
    {&chess::RookRays::north, Edge::Forward,
     MoveDirection::File}, // squares increase → lsb_index
    {&chess::RookRays::south, Edge::Backward,
     MoveDirection::File}, // squares decrease → msb_index
    {&chess::RookRays::east, Edge::Forward,
     MoveDirection::Rank}, // increasing file → lsb_index
    {&chess::RookRays::west, Edge::Backward,
     MoveDirection::Rank}, // decreasing file → msb_index
}};

bool is_rook_or_queen_of_enemy(OccupancyType occ, SideToMove stm) {
  bool white = (stm == SideToMove::White);
  if (white) {
    return occ == OccupancyType::bR || occ == OccupancyType::bQ;
  } else {
    return occ == OccupancyType::wR || occ == OccupancyType::wQ;
  }
}

constexpr std::array<DiagonalDir, 4> kDiagonalDirs{{
    {&chess::BishopRays::northeast, Edge::Forward,
     MoveDirection::NE}, // squares increase as you move northeast → lsb_index
    {&chess::BishopRays::northwest, Edge::Forward,
     MoveDirection::NW}, // squares increase as you move northwest → lsb_index
    {&chess::BishopRays::southeast, Edge::Backward,
     MoveDirection::SE}, // squares decrease as you move southeast → msb_index
    {&chess::BishopRays::southwest, Edge::Backward,
     MoveDirection::SW}, // squares decrease as you move southwest → msb_index
}};

bool is_bishop_or_queen_of_enemy(OccupancyType occ, SideToMove stm) {
  bool white = (stm == SideToMove::White);
  if (white) {
    return occ == OccupancyType::bB || occ == OccupancyType::bQ;
  } else {
    return occ == OccupancyType::wB || occ == OccupancyType::wQ;
  }
}
inline void clear_pin_array(std::array<PinnedBitBoardDirections, 64>& arr,
                            Bitboard mask) {
  for (auto& entry : arr) {
    entry.mask = mask;
    entry.direction = MoveDirection::None;
  }
}

inline void reset_pin_map(PinnedMapByPiece& pinned_map) {
  clear_pin_array(pinned_map.bishop_pins, 0);
  clear_pin_array(pinned_map.rook_pins, 0);
  clear_pin_array(pinned_map.queen_pins, 0);
  clear_pin_array(pinned_map.pawn_pins, 0);
  clear_pin_array(pinned_map.knight_pins, ~Bitboard{0});
}

} // namespace

void build_pinned_map_into(const Board& board, SideToMove stm,
                           PinnedMapByPiece& pinned_map) {
  reset_pin_map(pinned_map);

  const auto side_idx = to_index(stm);
  if (board.king_list[side_idx].count == 0 ||
      board.king_positions[side_idx] < 0) {
    return;
  }

  const auto pick_index = [](Bitboard mask, Edge edge) {
    return static_cast<u_int8_t>(edge == Edge::Forward
                                     ? to_index(lsb_index(mask))
                                     : to_index(msb_index(mask)));
  };

  const bool white = (stm == SideToMove::White);
  const u_int8_t king_sq = static_cast<u_int8_t>(
      white ? board.king_list[0].squares[0] : board.king_list[1].squares[0]);
  const Bitboard my_occ = board.occupancy[to_index(stm)];
  const Bitboard their_occ = board.occupancy[to_index(flip_side(stm))];

  auto record_orthogonal_pin = [&](Square sq,
                                   const PinnedBitBoardDirections& data) {
    const std::size_t idx = to_index(sq);
    pinned_map.rook_pins[idx] = data;
    pinned_map.queen_pins[idx] = data;
    pinned_map.pawn_pins[idx] = data;
    pinned_map.knight_pins[idx] = {0, MoveDirection::None};
  };

  auto record_diagonal_pin = [&](Square sq,
                                 const PinnedBitBoardDirections& data) {
    const std::size_t idx = to_index(sq);
    pinned_map.bishop_pins[idx] = data;
    pinned_map.queen_pins[idx] = data;
    pinned_map.knight_pins[idx] = {0, MoveDirection::None};
    pinned_map.pawn_pins[idx] = data;
  };

  const auto& rook_rays = ROOK_RAYS[king_sq];
  for (const auto& dir : kOrthoDirs) {
    const Bitboard ray = rook_rays.*(dir.ray);
    Bitboard blockers = ray & (my_occ | their_occ);
    if (!blockers)
      continue;

    const u_int8_t first_sq = pick_index(blockers, dir.first_blocker_edge);
    const auto& first_rays = ROOK_RAYS[first_sq];

    const Bitboard beyond_first = first_rays.*(dir.ray);
    if (!beyond_first)
      continue;

    const Bitboard next = beyond_first & (my_occ | their_occ);
    if (!next)
      continue;

    const u_int8_t next_sq = pick_index(next, dir.first_blocker_edge);
    const OccupancyType occ = board.pieces[next_sq];
    if (!is_rook_or_queen_of_enemy(occ, stm))
      continue;

    const Bitboard beyond_attacker = ROOK_RAYS[next_sq].*(dir.ray);
    const Bitboard line_mask = ray & ~beyond_attacker;
    if (!line_mask)
      continue;

    record_orthogonal_pin(static_cast<Square>(first_sq), {line_mask, dir.dir});
  }

  const auto& bishop_rays = BISHOP_RAYS[king_sq];
  for (const auto& dir : kDiagonalDirs) {
    const Bitboard ray = bishop_rays.*(dir.ray);
    Bitboard blockers = ray & (my_occ | their_occ);
    if (!blockers)
      continue;

    const u_int8_t first_sq = pick_index(blockers, dir.first_blocker_edge);
    const auto& first_rays = BISHOP_RAYS[first_sq];

    const Bitboard beyond_first = first_rays.*(dir.ray);
    if (!beyond_first)
      continue;

    const Bitboard next = beyond_first & (my_occ | their_occ);
    if (!next)
      continue;

    const u_int8_t next_sq = pick_index(next, dir.first_blocker_edge);
    const OccupancyType occ = board.pieces[next_sq];
    if (!is_bishop_or_queen_of_enemy(occ, stm))
      continue;

    const Bitboard beyond_attacker = BISHOP_RAYS[next_sq].*(dir.ray);
    const Bitboard line_mask = ray & ~beyond_attacker;
    if (!line_mask)
      continue;

    record_diagonal_pin(static_cast<Square>(first_sq), {line_mask, dir.dir});
  }
}

PinnedMapByPiece build_pinned_map(const Board& board, SideToMove stm) {
  PinnedMapByPiece pinned_map{};
  build_pinned_map_into(board, stm, pinned_map);
  return pinned_map;
}
} // namespace chess