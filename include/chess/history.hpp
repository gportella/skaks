#pragma once
#include "chess/defaults.hpp"
// #include "chess/moves.hpp"

#include <array>
#include <cstdint>

namespace chess {

struct MoveHistory {
  std::array<std::uint64_t, MAX_PLY> key_history;
  int repetition_start = 0;
  int ply_count = 0;
  // std::array<Move, MAX_PLY> killer_moves;
  // std::array<int, MAX_PLY> killer_move_counts = {0};
  std::array<uint8_t, MAX_PLY> repetition_counts;
};
} // namespace chess