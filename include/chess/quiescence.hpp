#pragma once

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/evaluator.hpp"
#include "chess/moves.hpp"
#include "chess/nnue_sf.hpp"
#include "chess/transposition_table.hpp"
#include "chess/types.hpp"

namespace chess {

int quiescence(Board& board, int alpha, int beta, SideToMove stm,
               const EvaluatorFn& evaluator, NnueAdapter* nnue_adapter,
               std::uint64_t& nodes, TranspositionTable* tt = nullptr,
               int ply = 0, int* max_ply = nullptr);
}