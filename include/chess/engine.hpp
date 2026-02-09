#pragma once

#include "chess/board.hpp"
#include "chess/history.hpp"
#include "chess/nnue_sf.hpp"
#include "chess/search.hpp"
#include "chess/transposition_table.hpp"

#include <array>
#include <cstdint>

namespace chess {

class Engine {
public:
  class SearchSession;

  Engine();

  [[nodiscard]] SearchResult search(Board& board,
                                    const SearchParameters& params);
  [[nodiscard]] int evaluate(const Board& board, NnueAdapter* adapter) const;

  void set_thread_count(int count);
  [[nodiscard]] int thread_count() const {
    return thread_count_;
  }

  [[nodiscard]] SearchSession create_session(Board& board);

  void reset_history(const Board& board);
  void clear_history();
  void clear_transposition_table();
  void resize_transposition_table_mb(std::size_t megabytes);

  [[nodiscard]] const MoveHistory& history() const {
    return history_;
  }
  void record_position(std::uint64_t key, bool irreversible);

  void init_nnue();

private:
  MoveHistory history_;
  TranspositionTable tt_;
  int thread_count_ = 1;
};

class Engine::SearchSession {
public:
  SearchSession(Engine& engine, Board& board);
  SearchSession(const SearchSession&) = delete;
  SearchSession& operator=(const SearchSession&) = delete;
  SearchSession(SearchSession&&) = default;
  SearchSession& operator=(SearchSession&&) = default;

  [[nodiscard]] SearchResult run(const SearchParameters& params);

  [[nodiscard]] Board& board() {
    return *board_;
  }

  [[nodiscard]] MoveHistory& history() {
    return engine_->history_;
  }

  [[nodiscard]] TranspositionTable& transposition_table() {
    return engine_->tt_;
  }

private:
  Engine* engine_ = nullptr;
  Board* board_ = nullptr;
};

} // namespace chess
