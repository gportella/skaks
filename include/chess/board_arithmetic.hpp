#pragma once

#include "chess/types.hpp"

#include <bit>
#include <cassert>

namespace chess {

inline int chebyshev_dist(std::size_t sq1_index, std::size_t sq2_index) {
  int f1 = static_cast<int>(sq1_index) % 8;
  int r1 = static_cast<int>(sq1_index) / 8;
  int f2 = static_cast<int>(sq2_index) % 8;
  int r2 = static_cast<int>(sq2_index) / 8;
  int file_dist = abs(f1 - f2);
  int rank_dist = abs(r1 - r2);

  int distance = std::max(file_dist, rank_dist);
  return distance;
}

constexpr int lsb_index(Bitboard b) noexcept {
  // Precondition: b != 0
  return std::countr_zero(b);
}

constexpr int msb_index(Bitboard b) noexcept {
  // Precondition: b != 0
  return 63 - std::countl_zero(b);
}

inline constexpr Bitboard bit_mask(int sq) noexcept {
  return Bitboard(1) << sq;
}

inline constexpr void set_bit(Bitboard& bb, int sq) noexcept {
  bb |= bit_mask(sq);
}

inline constexpr void clear_bit(Bitboard& bb, int sq) noexcept {
  bb &= ~bit_mask(sq);
}

inline constexpr void toggle_bit(Bitboard& bb, int sq) noexcept {
  bb ^= bit_mask(sq);
}

inline constexpr Bitboard bit_mask(Square sq) noexcept {
  return bit_mask(static_cast<int>(sq));
}

inline constexpr void set_bit(Bitboard& bb, Square sq) noexcept {
  set_bit(bb, static_cast<int>(sq));
}

inline constexpr void clear_bit(Bitboard& bb, Square sq) noexcept {
  clear_bit(bb, static_cast<int>(sq));
}

inline constexpr void toggle_bit(Bitboard& bb, Square sq) noexcept {
  toggle_bit(bb, static_cast<int>(sq));
}

// inline int popcount_bitboard(Bitboard bb) {
// #if defined(_MSC_VER)
//   return static_cast<int>(__popcnt64(bb));
// #else
//   return static_cast<int>(__builtin_popcountll(static_cast<unsigned long long>(bb)));
// #endif
// }
inline int popcount_bitboard(Bitboard bb) noexcept {
  return static_cast<int>(std::popcount(bb));
}
} // namespace chess