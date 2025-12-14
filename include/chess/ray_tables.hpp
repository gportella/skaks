#pragma once

#include "chess/types.hpp"

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

constexpr std::array<BishopRays, 64> build_bishop_rays() {
  std::array<BishopRays, 64> rays{};
  for (std::size_t sq = 0; sq < 64; ++sq) {
    auto& ray = rays[sq];
    const std::size_t rank = sq / 8;
    const std::size_t file = sq % 8;

    for (std::size_t r = rank + 1, f = file + 1; r < 8 && f < 8; ++r, ++f) {
      const std::size_t target = (r * 8) + f;
      ray.northeast |= Bitboard(1) << target;
    }
    for (std::size_t r = rank + 1, f = file; r < 8 && f > 0; ++r, --f) {
      const std::size_t target = (r * 8) + (f - 1);
      ray.northwest |= Bitboard(1) << target;
    }
    for (int r = static_cast<int>(rank) - 1, f = static_cast<int>(file) + 1; r >= 0 && f < 8;
         --r, ++f) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + static_cast<std::size_t>(f);
      ray.southeast |= Bitboard(1) << target;
    }
    for (int r = static_cast<int>(rank) - 1, f = static_cast<int>(file) - 1; r >= 0 && f >= 0;
         --r, --f) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + static_cast<std::size_t>(f);
      ray.southwest |= Bitboard(1) << target;
    }
  }
  return rays;
}

inline constexpr std::array<BishopRays, 64> BISHOP_RAYS = build_bishop_rays();

} // namespace chess
