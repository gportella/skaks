#pragma once

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/evaluator.hpp"
#include "chess/history.hpp"
#include "chess/moves.hpp"
#include "chess/transposition_table.hpp"
#include "chess/types.hpp"

#include <cstddef>
#include <cstdint>

namespace chess {

struct SearchParameters {
  int depth = 0;
  int alpha = -INF;
  int beta = INF;
  int pv_count = 1;
  const uint32_t* root_excluded_moves = nullptr;
  std::size_t root_excluded_count = 0;
};

struct SearchResult {
  enum class Outcome { InProgress, Mate, DrawByStalemate, DrawByRepetition };

  int score = 0;
  Move best_move{};
  Outcome outcome = Outcome::InProgress;
  std::uint64_t nodes = 0;
  std::uint64_t elapsed_ms = 0;
};

SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params,
                             const EvaluatorFn& evaluator, MoveHistory* history = nullptr,
                             TranspositionTable* tt = nullptr, int repetition_start = 0);

SearchResult search_position(Board& board, SideToMove stm, const SearchParameters& params);

} // namespace chess
