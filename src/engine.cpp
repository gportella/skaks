#include "chess/engine.hpp"

#include "chess/scoring_rules.hpp"

#include <cstddef>

namespace chess {

Engine::Engine() : eval_config_{}, history_{} {
  clear_history();
}

SearchResult Engine::search(Board& board, const SearchParameters& params) {
  if (history_.ply_count == 0 ||
      history_.key_history[static_cast<std::size_t>(history_.ply_count - 1)] !=
          board.position_key) {
    reset_history(board);
  }

  auto evaluator = [this](const Board& state) { return evaluate(state); };

  const int base_ply = history_.ply_count;
  const int repetition_start = history_.repetition_start;
  auto result =
      search_position(board, board.side_to_move, params, evaluator, &history_, repetition_start);
  history_.ply_count = base_ply;
  return result;
}

int Engine::evaluate(const Board& board) const {
  // Placeholder: future versions will blend features based on eval_config_
  (void)eval_config_;
  return evaluate_board(board);
}

void Engine::clear_history() {
  history_.key_history.fill(0);
  history_.ply_count = 0;
  history_.repetition_start = 0;
}

void Engine::reset_history(const Board& board) {
  history_.key_history.fill(0);
  history_.key_history[0] = board.position_key;
  history_.ply_count = 1;
  history_.repetition_start = 0;
}

void Engine::record_position(std::uint64_t key, bool irreversible) {
  if (history_.ply_count < static_cast<int>(history_.key_history.size())) {
    history_.key_history[static_cast<std::size_t>(history_.ply_count++)] = key;
  }
  if (irreversible) {
    history_.repetition_start = history_.ply_count;
  }
}

} // namespace chess
