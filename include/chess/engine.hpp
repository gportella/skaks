#pragma once

#include "chess/board.hpp"
#include "chess/history.hpp"
#include "chess/search.hpp"
#include "chess/transposition_table.hpp"

#include <array>
#include <cstdint>

namespace chess {

struct EvaluationConfig {
  int material_weight = 1;
  int mobility_weight = 1;
  int king_safety_weight = 1;
};

class Engine {
public:
  Engine();

  [[nodiscard]] SearchResult search(Board& board, const SearchParameters& params);
  [[nodiscard]] int evaluate(const Board& board) const;

  void reset_history(const Board& board);
  void clear_history();
  void clear_transposition_table();

  [[nodiscard]] const MoveHistory& history() const {
    return history_;
  }
  void record_position(std::uint64_t key, bool irreversible);

private:
  EvaluationConfig eval_config_;
  MoveHistory history_;
  TranspositionTable tt_;
};

} // namespace chess
