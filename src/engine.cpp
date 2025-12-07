#include "chess/engine.hpp"

#include <numeric>

namespace chess {

Engine::Engine() : squares_{} {}

int Engine::sample_evaluation() const {
  return std::accumulate(squares_.begin(), squares_.end(), 0);
}

} // namespace chess
