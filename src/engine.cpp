// SPDX-License-Identifier: GPL-3.0-or-later
#include "chess/engine.hpp"

#include "chess/moves.hpp"
#include "chess/nnue_sf.hpp"
#include "chess/search_stats.hpp"
#include "chess/time_manager.hpp"
#include "evaluate.h"
#include "probe.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace chess {

namespace {

std::atomic<std::uint64_t> g_smp_fallback_count{0};

void log_smp_fallback() {
  const auto count =
      g_smp_fallback_count.fetch_add(1, std::memory_order_relaxed) + 1;
  std::cerr << "[skaks] SMP fallback activated; count=" << count << '\n';
#ifndef NDEBUG
  std::abort();
#endif
}

} // namespace

Engine::Engine() : history_{} {
  clear_history();
  tt_.resize_mb(DEFAULT_HASH_MB);
}

SearchResult Engine::search(Board& board, const SearchParameters& params) {
  auto session = create_session(board);
  return session.run(params);
}

void Engine::init_nnue() {
  const char* big_env = std::getenv("SKAKS_NNUE_BIG");
  const char* small_env = std::getenv("SKAKS_NNUE_SMALL");
  const char* big = (big_env && *big_env) ? big_env : EvalFileDefaultNameBig;
  const char* small =
      (small_env && *small_env) ? small_env : EvalFileDefaultNameSmall;
  Stockfish::Probe::init(big, small);
}

int Engine::evaluate(const Board& board, NnueAdapter* adapter) const {
  if (adapter) {
    return evaluate_nnue_stockfish_incremental(*adapter);
  }
  return evaluate_nnue_stockfish(board);
}

void Engine::set_thread_count(int count) {
  thread_count_ = std::max(count, 1);
}

Engine::SearchSession Engine::create_session(Board& board) {
  return SearchSession(*this, board);
}

Engine::SearchSession::SearchSession(Engine& engine, Board& board)
    : engine_(&engine), board_(&board) {}

/**
 * @brief Runs a complete search session for the current position.
 *
 * Orchestrates history initialization, evaluator binding, time management,
 * and dispatches either single-threaded or parallel search. Handles SMP
 * fallback when parallel search fails or aborts, aggregates node counts,
 * tracks elapsed time, and restores history state before returning the
 * SearchResult.
 *
 * Used by:
 * - Engine::search() entry point (CLI/UCI/self-play flows)
 * - Python bindings (via Engine::search)
 * - perf runner (create_session + run)
 *
 * @param params Search parameters and limits to use for this session.
 * @return The search result including best move, nodes searched, and elapsed
 * time.
 */
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

  EvaluatorFn evaluator = [eng_ptr = &engine](const Board& state,
                                              NnueAdapter* adapter) {
    if (search_stats_enabled()) {
      const auto start = std::chrono::steady_clock::now();
      const int score = eng_ptr->evaluate(state, adapter);
      const auto end = std::chrono::steady_clock::now();
      const std::uint64_t elapsed = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count());
      auto& stats = search_stats();
      stats.eval_calls.fetch_add(1, std::memory_order_relaxed);
      stats.eval_time_ns.fetch_add(elapsed, std::memory_order_relaxed);
      return score;
    }
    return eng_ptr->evaluate(state, adapter);
  };

  const int base_ply = history.ply_count;
  const int repetition_start = history.repetition_start;

  TimeManager time_manager;
  SearchParameters work_params = params;
  if (work_params.limits.use_time) {
    time_manager.configure(board.side_to_move, work_params.limits);
    time_manager.start();
    if (time_manager.enabled()) {
      work_params.time_manager = &time_manager;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  SearchResult result{};
  const int worker_count = std::max(engine.thread_count(), 1);

  if (worker_count <= 1) {
    result = search_position(board, board.side_to_move, work_params, evaluator,
                             &history, &tt, repetition_start);
  } else {
    std::shared_ptr<std::atomic<bool>> shared_abort_owner;
    if (!work_params.abort_flag) {
      shared_abort_owner = std::make_shared<std::atomic<bool>>(false);
      work_params.abort_flag = shared_abort_owner.get();
    }

    const int helper_threads = worker_count - 1;
    result = search_position_parallel(board, board.side_to_move, work_params,
                                      evaluator, &history, &tt, repetition_start,
                                      helper_threads);

    const bool needs_fallback =
        result.best_move.moving_pc == OccupancyType::empty || result.aborted;
    if (needs_fallback) {
      log_smp_fallback();
      SearchResult fallback =
          search_position(board, board.side_to_move, work_params, evaluator,
                          &history, &tt, repetition_start);
      fallback.nodes += result.nodes;
      result = std::move(fallback);
      result.aborted = false;
    }
  }

  const auto end = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  result.elapsed_ms = static_cast<std::uint64_t>(elapsed);
  history.ply_count = base_ply;
  history.repetition_start = repetition_start;
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

void Engine::resize_transposition_table_mb(std::size_t megabytes) {
  tt_.resize_mb(megabytes);
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
