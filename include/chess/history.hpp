#pragma once
#include "chess/defaults.hpp"

#include <array>
#include <cstdint>

namespace chess {

struct MoveHistory {
  std::array<std::uint64_t, MAX_PLY> key_history;
  int repetition_start = 0;
  int ply_count = 0;
};
} // namespace chess