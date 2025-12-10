#pragma once

#include "chess/board_arithmetic.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <array>

namespace chess {

struct BishopRays {
  Bitboard northeast{};
  Bitboard northwest{};
  Bitboard southeast{};
  Bitboard southwest{};
};
struct RookRays {
  Bitboard north{};
  Bitboard south{};
  Bitboard east{};
  Bitboard west{};
};

std::array<Bitboard, 64> build_king_attack_patterns() {
  std::array<Bitboard, 64> attacks{};
  for (std::size_t sq = 0; sq < 64; ++sq) {
    Bitboard bit = Bitboard(1) << sq;
    Bitboard mask = 0;
    mask |= (bit << 8) & U64_MASK;
    mask |= (bit >> 8) & U64_MASK;
    if (bit & NOT_FILE_H) {
      mask |= (bit << 9) & U64_MASK;
      mask |= (bit >> 7) & U64_MASK;
      mask |= (bit << 1) & U64_MASK;
    }
    if (bit & NOT_FILE_A) {
      mask |= (bit << 7) & U64_MASK;
      mask |= (bit >> 9) & U64_MASK;
      mask |= (bit >> 1) & U64_MASK;
    }
    attacks[sq] = Bitboard(mask & U64_MASK);
  }
  return attacks;
}

constexpr std::array<RookRays, 64> build_rook_rays() {
  std::array<RookRays, 64> rays{};
  for (std::size_t sq = 0; sq < 64; ++sq) {
    auto& ray = rays[sq];
    const std::size_t rank = sq / 8;
    const std::size_t file = sq % 8;

    for (std::size_t r = rank + 1; r < 8; ++r) {
      const std::size_t target = (r * 8) + file;
      ray.north |= Bitboard(1) << target;
    }
    for (int r = static_cast<int>(rank) - 1; r >= 0; --r) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + file;
      ray.south |= Bitboard(1) << target;
    }
    for (std::size_t f = file + 1; f < 8; ++f) {
      const std::size_t target = (rank * 8) + f;
      ray.east |= Bitboard(1) << target;
    }
    for (int f = static_cast<int>(file) - 1; f >= 0; --f) {
      const std::size_t target = (rank * 8) + static_cast<std::size_t>(f);
      ray.west |= Bitboard(1) << target;
    }
  }
  return rays;
}

inline constexpr std::array<RookRays, 64> ROOK_RAYS = build_rook_rays();

/**
 * Calculate rooks attacks from a given square
 * We might want to cache this one if possible
 * @param[in] sq The square index in the 64 mailbox
 * @param[in] b_occ The occupancy of the board
 * return A bitboard with the possible locations to attack

**/
Bitboard rook_attacks_from(u_int8_t sq, Bitboard b_occupancy) {

  const auto& rays = ROOK_RAYS[sq];
  //  north
  auto north = rays.north;
  auto blocker = north & b_occupancy;
  if (blocker) {
    int b = lsb_index(blocker);
    Bitboard mask_below = (Bitboard(1) << b) - 1; // bits [0..b-1] set
    north &= mask_below;
  }
  // East
  Bitboard east = rays.east;
  Bitboard east_blockers = east & b_occupancy;
  if (east_blockers) {
    int b = lsb_index(east_blockers); // nearest east blocker
    Bitboard mask_below = (Bitboard(1) << b) - 1;
    east &= mask_below;
  }

  // South
  Bitboard south = rays.south;
  Bitboard south_blockers = south & b_occupancy;
  if (south_blockers) {
    int b = msb_index(south_blockers);
    Bitboard mask_above = ~((Bitboard(1) << (b + 1)) - 1); // bits [b+1..63] set
    south &= mask_above;
  }
  // West
  Bitboard west = rays.west;
  Bitboard west_blockers = west & b_occupancy;
  if (west_blockers) {
    int b = msb_index(west_blockers); // nearest west blocker
    Bitboard mask_above = ~((Bitboard(1) << (b + 1)) - 1);
    west &= mask_above;
  }

  return north | south | east | west;
}

/**
 * Generate rook attack based removing friedly occupancy
 * @params[in] board struct
 * @params[in] square of interest
 * @params[in] what side are we moving
 * @return a bitboard with possible locations to attack
 *
 */
Bitboard generate_rook_attack_bm(Board board, u_int8_t sq, SideToMove sidetm) {
  auto occ = board.occupancy[2];
  auto friendly = board.occupancy[to_index(sidetm)];
  auto attacks = rook_attacks_from(sq, occ);
  return attacks & ~friendly;
}

std::array<BishopRays, 64> build_bishop_rays() {
  std::array<BishopRays, 64> rays{};
  for (std::size_t sq = 0; sq < 64; ++sq) {
    auto& ray = rays[sq];
    const std::size_t rank = sq / 8;
    const std::size_t file = sq % 8;

    // Northeast
    for (std::size_t r = rank + 1, f = file + 1; r < 8 && f < 8; ++r, ++f) {
      const std::size_t target = (r * 8) + f;
      ray.northeast |= Bitboard(1) << target;
    }
    // Northwest
    for (std::size_t r = rank + 1, f = file; r < 8 && f > 0; ++r, --f) {
      const std::size_t target = (r * 8) + (f - 1);
      ray.northwest |= Bitboard(1) << target;
    }
    // Southeast
    for (int r = static_cast<int>(rank) - 1, f = static_cast<int>(file) + 1; r >= 0 && f < 8;
         --r, ++f) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + static_cast<std::size_t>(f);
      ray.southeast |= Bitboard(1) << target;
    }
    // Southwest
    for (int r = static_cast<int>(rank) - 1, f = static_cast<int>(file) - 1; r >= 0 && f >= 0;
         --r, --f) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + static_cast<std::size_t>(f);
      ray.southwest |= Bitboard(1) << target;
    }
  }
  return rays;

} // namespace chess
} // namespace chess