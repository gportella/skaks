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

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/types_io.hpp"

namespace chess {

constexpr std::array<OrthoDir, 4> kOrthoDirs{{
    {&chess::RookRays::north, Edge::Forward, MoveDirection::File},  // squares increase → lsb_index
    {&chess::RookRays::south, Edge::Backward, MoveDirection::File}, // squares decrease → msb_index
    {&chess::RookRays::east, Edge::Forward, MoveDirection::Rank},   // increasing file → lsb_index
    {&chess::RookRays::west, Edge::Backward, MoveDirection::Rank},  // decreasing file → msb_index
}};

bool is_rook_or_queen_of_enemy(OccupancyType occ, SideToMove stm) {
  bool white = (stm == SideToMove::White);
  if (white) {
    return occ == OccupancyType::bR || occ == OccupancyType::bQ;
  } else {
    return occ == OccupancyType::wR || occ == OccupancyType::wQ;
  }
}

std::array<std::pair<Square, PinnedBitBoardDirections>, 8> find_orthogonal_pins(const Board& board,
                                                                                SideToMove stm) {

  auto pick_index = [&](Bitboard mask, Edge edge) {
    return edge == Edge::Forward ? to_index(lsb_index(mask)) : to_index(msb_index(mask));
  };

  bool white = (stm == SideToMove::White);
  std::array<std::pair<Square, PinnedBitBoardDirections>, 8> pins{};
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
      pins[pin_count++] = {pinned_sq, {line_mask, dir.dir}};
    } else {
      std::cerr << "Warning: exceeded pin storage capacity\n" << std::endl;
    }
  }

  return pins;
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

std::array<std::pair<Square, PinnedBitBoardDirections>, 8> find_diagonal_pins(const Board& board,
                                                                              SideToMove stm) {

  auto pick_index = [&](Bitboard mask, Edge edge) {
    return edge == Edge::Forward ? to_index(lsb_index(mask)) : to_index(msb_index(mask));
  };

  bool white = (stm == SideToMove::White);
  std::array<std::pair<Square, PinnedBitBoardDirections>, 8> pins{};
  std::size_t pin_count = 0;

  u_int8_t king_sq =
      static_cast<u_int8_t>(white ? board.king_list[0].squares[0] : board.king_list[1].squares[0]);

  const auto& king_rays = BISHOP_RAYS[king_sq];
  const Bitboard my_occ = board.occupancy[to_index(stm)];
  const Bitboard their_occ = board.occupancy[to_index(flip_side(stm))];

  for (const auto& dir : kDiagonalDirs) {
    const Bitboard ray = king_rays.*(dir.ray);
    Bitboard blockers = ray & (my_occ | their_occ);
    if (!blockers)
      continue;

    auto first_sq = pick_index(blockers, dir.first_blocker_edge);
    const auto& first_rays = BISHOP_RAYS[first_sq];

    const Bitboard beyond_first = first_rays.*(dir.ray);
    if (!beyond_first)
      continue;

    const Bitboard next = beyond_first & (my_occ | their_occ);
    if (!next)
      continue;

    auto next_sq = pick_index(next, dir.first_blocker_edge);
    const OccupancyType occ = board.pieces[next_sq];
    if (!is_bishop_or_queen_of_enemy(occ, stm))
      continue;

    Bitboard beyond_attacker = BISHOP_RAYS[next_sq].*(dir.ray);
    Bitboard line_mask = ray & ~beyond_attacker;
    Square pinned_sq = static_cast<Square>(first_sq);
    if (pin_count < pins.size()) { // guard against overflow when multiple pins appear
      pins[pin_count++] = {pinned_sq, {line_mask, dir.dir}};
    } else {
      std::cerr << "Warning: exceeded pin storage capacity\n" << std::endl;
    }
  }

  return pins;
}

chess::PinnedMapByPiece build_pinned_map(const Board& board, SideToMove stm) {

  const std::array<std::pair<Square, PinnedBitBoardDirections>, 8> ortho_pins =
      chess::find_orthogonal_pins(board, stm);
  const std::array<std::pair<Square, PinnedBitBoardDirections>, 8> diag_pins =
      chess::find_diagonal_pins(board, stm);

  chess::PinnedMapByPiece pinned_map{};
  pinned_map.knight_pins.fill({~Bitboard{0}, MoveDirection::None});

  for (const auto& [sq, mask_dir] : ortho_pins) {
    if (mask_dir.mask != 0) {
      pinned_map.rook_pins[to_index(sq)] = mask_dir;
      pinned_map.queen_pins[to_index(sq)] = mask_dir;
      //  to be dealt with: pawns when rook-pinned can only move forward
      pinned_map.pawn_pins[to_index(sq)] = mask_dir;
      // knights cannot move when rook-pinned
      pinned_map.knight_pins[to_index(sq)] = {0, MoveDirection::None};
    }
  }

  for (const auto& [sq, mask_dir] : diag_pins) {
    if (mask_dir.mask != 0) {
      pinned_map.bishop_pins[to_index(sq)] = mask_dir;
      pinned_map.queen_pins[to_index(sq)] = mask_dir;
      // knights cannot move when bishop-pinned
      pinned_map.knight_pins[to_index(sq)] = {0, MoveDirection::None};

      // pawns have special rules when bishop-pinned, handled elsewhere
      pinned_map.pawn_pins[to_index(sq)] = mask_dir;
    }
  }

  return pinned_map;
}
} // namespace chess