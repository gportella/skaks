#pragma once

#include "chess/board.hpp"

#include <array>

namespace chess {

class Engine {
public:
  Engine();

  [[nodiscard]] int sample_evaluation() const;

private:
  std::array<int, 64> squares_;
};

} // namespace chess
