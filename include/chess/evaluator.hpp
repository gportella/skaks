#pragma once

#include "chess/board.hpp"

#include <functional>

namespace chess {

using EvaluatorFn = std::function<int(const Board&)>;

} // namespace chess
