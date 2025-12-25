#pragma once

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/evaluator.hpp"
#include "chess/history.hpp"
#include "chess/moves.hpp"
#include "chess/transposition_table.hpp"
#include "chess/types.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

class TimeManager;

struct SearchLimits {
  bool use_time = false;
  bool per_move = false;
  std::uint64_t move_time_ms = 0;
  std::uint64_t white_time_ms = 0;
  std::uint64_t black_time_ms = 0;
  std::uint64_t white_increment_ms = 0;
  std::uint64_t black_increment_ms = 0;
  std::uint32_t moves_to_go = 0;
};

struct SearchParameters {
  int depth = 0;
  int alpha = -INF;
  int beta = INF;
  int pv_count = 1;
  const uint32_t* root_excluded_moves = nullptr;
  std::size_t root_excluded_count = 0;
  SearchLimits limits;
  TimeManager* time_manager = nullptr;
  std::atomic<bool>* abort_flag = nullptr;
};

struct SearchResult {
  enum class Outcome { InProgress, Mate, DrawByStalemate, DrawByRepetition };

  int score = 0;
  Move best_move{};
  Outcome outcome = Outcome::InProgress;
  std::uint64_t nodes = 0;
  std::uint64_t elapsed_ms = 0;
  int searched_depth = 0;
  int selective_depth = 0;
  bool aborted = false;
  std::vector<Move> principal_variation;
  int pv_length = 0;

  SearchResult() = default;
  SearchResult(const SearchResult& other) = default;
  SearchResult(SearchResult&& other) noexcept = default;
  SearchResult& operator=(const SearchResult& other) = default;
  SearchResult& operator=(SearchResult&& other) noexcept = default;
};

SearchResult
search_position(Board& board, SideToMove stm, const SearchParameters& params,
                const EvaluatorFn& evaluator, MoveHistory* history = nullptr,
                TranspositionTable* tt = nullptr, int repetition_start = 0);

SearchResult search_position(Board& board, SideToMove stm,
                             const SearchParameters& params);

} // namespace chess
