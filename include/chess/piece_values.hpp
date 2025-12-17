#pragma once

#include "chess/types.hpp"

namespace chess {

inline constexpr int piece_material_value(OccupancyType piece) {
  switch (piece) {
  case OccupancyType::wP:
    return 100;
  case OccupancyType::wN:
    return 320;
  case OccupancyType::wB:
    return 330;
  case OccupancyType::wR:
    return 500;
  case OccupancyType::wQ:
    return 900;
  case OccupancyType::bP:
    return -100;
  case OccupancyType::bN:
    return -320;
  case OccupancyType::bB:
    return -330;
  case OccupancyType::bR:
    return -500;
  case OccupancyType::bQ:
    return -900;
  default:
    return 0;
  }
}

inline constexpr int piece_material_magnitude(OccupancyType piece) {
  const int value = piece_material_value(piece);
  return value >= 0 ? value : -value;
}

} // namespace chess
