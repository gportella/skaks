#pragma once
#include "chess/types.hpp"

#include <ostream>
#include <string>
#include <string_view>

namespace chess {

constexpr std::size_t to_index(SideToMove s) noexcept {
  return static_cast<std::size_t>(s);
}

constexpr std::size_t to_index(PieceColor color) noexcept {
  return static_cast<std::size_t>(color);
}

constexpr std::size_t to_index(int sq) noexcept {
  return static_cast<std::size_t>(sq);
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

inline std::string square_to_string(int sq) {
  if (sq < 0 || sq >= 64) {
    return "??";
  }
  const char file = static_cast<char>('a' + (sq % 8));
  const char rank = static_cast<char>('1' + (sq / 8));
  return std::string{file, rank};
}

inline std::string square_to_string(Square sq) {
  return square_to_string(static_cast<int>(sq));
}

inline std::ostream& operator<<(std::ostream& os, SideToMove side) {
  return os << to_string(side);
}
inline std::ostream& operator<<(std::ostream& os, OccupancyType occ) {
  return os << to_string(occ);
}
} // namespace chess