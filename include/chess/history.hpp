#pragma once
#include <cstdint>
#include <vector>

namespace chess {

struct MoveHistory {
  std::array<std::uint64_t, MAX_PLY> key_history;
  int repetition_start = 0;
};
} // namespace chess