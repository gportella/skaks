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
} // namespace chess