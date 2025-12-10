#pragma once
#include "chess/types.hpp"

#include <ostream>
#include <string_view>

namespace chess {

constexpr std::size_t to_index(SideToMove s) noexcept {
  return static_cast<std::size_t>(s);
}

inline constexpr std::string_view to_string(SideToMove side) {
  return side == SideToMove::White ? "White" : "Black";
}
inline constexpr std::string_view to_string(OccupancyType occ) {
  switch (occ) {
  case OccupancyType::empty:
    return "empty";
  case OccupancyType::wP:
    return "P";
  case OccupancyType::wN:
    return "N";
  case OccupancyType::wB:
    return "B";
  case OccupancyType::wR:
    return "R";
  case OccupancyType::wQ:
    return "Q";
  case OccupancyType::wK:
    return "K";
  case OccupancyType::bP:
    return "p";
  case OccupancyType::bN:
    return "n";
  case OccupancyType::bB:
    return "b";
  case OccupancyType::bR:
    return "r";
  case OccupancyType::bQ:
    return "q";
  case OccupancyType::bK:
    return "k";
  default:
    return "unknown";
  }
}

inline std::ostream& operator<<(std::ostream& os, SideToMove side) {
  return os << to_string(side);
}
inline std::ostream& operator<<(std::ostream& os, OccupancyType occ) {
  return os << to_string(occ);
}
} // namespace chess