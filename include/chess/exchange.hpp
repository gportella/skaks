#pragma once

#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"

#include <optional>

namespace chess {

int static_exchange_eval(const Board& board, const Move& move);

} // namespace chess