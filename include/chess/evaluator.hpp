#pragma once

#include "chess/board.hpp"
#include "chess/nnue_sf.hpp"

#include <functional>

namespace chess {

using EvaluatorFn = std::function<int(const Board&, NnueAdapter*)>;

} // namespace chess
