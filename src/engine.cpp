#include "chess/engine.hpp"

#include "chess/scoring_rules.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>

namespace chess {

Engine::Engine() : eval_config_{}, history_{} {
  clear_history();
}

SearchResult Engine::search(Board& board, const SearchParameters& params) {
  auto session = create_session(board);
  return session.run(params);
}

int Engine::evaluate(const Board& board) const {
  // Placeholder: future versions will blend features based on eval_config_
  (void)eval_config_;
  return evaluate_board(board);
}

Engine::SearchSession Engine::create_session(Board& board) {
  return SearchSession(*this, board);
}

Engine::SearchSession::SearchSession(Engine& engine, Board& board)
    : engine_(&engine), board_(&board) {}

SearchResult Engine::SearchSession::run(const SearchParameters& params) {
  auto& engine = *engine_;
  auto& board = *board_;
  auto& history = engine.history_;
  auto& tt = engine.tt_;

  if (history.ply_count == 0) {
    engine.reset_history(board);
  } else {
    const auto idx = static_cast<std::size_t>(history.ply_count - 1);
    history.key_history[idx] = board.position_key;
  }

  auto evaluator = [eng_ptr = &engine](const Board& state) { return eng_ptr->evaluate(state); };

  const int base_ply = history.ply_count;
  const int repetition_start = history.repetition_start;

  const auto start = std::chrono::steady_clock::now();
  auto result = search_position(board, board.side_to_move, params, evaluator, &history, &tt,
                                repetition_start);
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  result.elapsed_ms = static_cast<std::uint64_t>(elapsed);
  history.ply_count = base_ply;
  return result;
}

void Engine::clear_history() {
  history_.key_history.fill(0);
  history_.repetition_counts.fill(0);
  history_.ply_count = 0;
  history_.repetition_start = 0;
}

void Engine::clear_transposition_table() {
  tt_.clear();
}

void Engine::reset_history(const Board& board) {
  history_.key_history.fill(0);
  history_.repetition_counts.fill(0);
  history_.key_history[0] = board.position_key;
  history_.ply_count = 1;
  history_.repetition_start = 0;
}

void Engine::record_position(std::uint64_t key, bool irreversible) {
  if (history_.ply_count < static_cast<int>(history_.key_history.size())) {
    const auto idx = static_cast<std::size_t>(history_.ply_count);
    history_.key_history[idx] = key;
    ++history_.ply_count;
  }
  if (irreversible) {
    history_.repetition_start = std::max(history_.ply_count - 1, 0);
  }
}

} // namespace chess
