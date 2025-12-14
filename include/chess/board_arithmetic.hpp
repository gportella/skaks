#pragma once

#include "chess/types.hpp"

#include <bit>

namespace chess {

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

inline int popcount_bitboard(Bitboard bb) {
#if defined(_MSC_VER)
  return static_cast<int>(__popcnt64(bb));
#else
  return static_cast<int>(__builtin_popcountll(static_cast<unsigned long long>(bb)));
#endif
}
} // namespace chess