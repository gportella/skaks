#include "chess/search.hpp"

#include "chess/complexity.hpp"
#include "chess/exchange.hpp"
#include "chess/move_ordering.hpp"
#include "chess/moves.hpp"
#include "chess/nnue_sf.hpp"
#include "chess/piece_values.hpp"
#include "chess/quiescence.hpp"
#include "chess/score.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/search_detail.hpp"
#include "chess/search_params.hpp"
#include "chess/search_stats.hpp"
#include "chess/time_manager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#define CHESS_USE_POSIX_THREADS 1
#include <pthread.h>
#else
#define CHESS_USE_POSIX_THREADS 0
#endif

/*
//  Parallel search implementation (Lazy SMP), ala Stockfish's but simplified.
//  search_position_parallel() sets up a shared LazySmpContext that contains
//  a queue, Transposition Table, evaluator and abort flags.
//  It spawns helper_threads workers with pthreads (on POSIX systems) via
//  pthread_create(), each running lazy_worker_loop().
//  The main thread runs the regular search;
//  When it hits a split point, it creates LazySmpTask
//  objects (board copy, move, local history/killer/order tables, eval context)
//  and enqueues them.
//  Workers pop tasks, run evaluate_move() on their local
//  board copy, and submit results back to the split point.
//  The main thread  periodically collects results and merges
//  them into the main PV/alpha‑beta state.
//
//  Parallelism is at move‑level: when the main thread reaches a split point, it
//  enqueues tasks for some later moves in the current node’s move list.
//  each worker runs that specific search subtree. The main thread keeps
//  searching its own move while helpers work.
*/

namespace chess {

namespace search_detail {

bool should_retry_pvs(SideToMove stm, int child_score, int narrow_alpha,
                      int narrow_beta) {
  if (stm == SideToMove::White) {
    return child_score >= narrow_beta;
  }
  return child_score <= narrow_alpha;
}

} // namespace search_detail

namespace {

constexpr int kMaxAspirationAttempts = 24;

struct LazySmpThread;

#if CHESS_USE_POSIX_THREADS
constexpr std::size_t kLazyWorkerStackSize = 16 * 1024 * 1024;
#endif

struct SearchScratch {
  MoveHistory* history = nullptr;
  KillerTable* killers = nullptr;
  TranspositionTable* tt = nullptr;
  MoveOrderingTables ordering;
  NnueAdapter* nnue_adapter = nullptr;
  TimeManager* time_manager = nullptr;
  std::uint64_t node_limit = 0;
  bool use_nodes = false;
  std::atomic<bool>* abort_requested = nullptr;
  std::atomic<bool>* local_abort = nullptr;
  LazySmpThread* thread = nullptr;
};

constexpr std::uint64_t kTimeCheckMask = 0x3FFULL;

inline bool should_abort_due_to_time(SearchScratch& scratch,
                                     std::uint64_t nodes) {
  if (scratch.use_nodes && scratch.node_limit > 0 &&
      nodes >= scratch.node_limit) {
    if (scratch.abort_requested) {
      scratch.abort_requested->store(true, std::memory_order_relaxed);
    }
    return true;
  }
  if (scratch.abort_requested &&
      scratch.abort_requested->load(std::memory_order_relaxed)) {
    return true;
  }
  if (scratch.local_abort &&
      scratch.local_abort->load(std::memory_order_relaxed)) {
    return true;
  }
  if (!scratch.time_manager || !scratch.time_manager->enabled()) {
    return false;
  }
  if (scratch.time_manager->hard_limit_reached()) {
    if (scratch.abort_requested) {
      scratch.abort_requested->store(true, std::memory_order_relaxed);
    }
    return true;
  }
  if ((nodes & kTimeCheckMask) != 0) {
    return false;
  }
  if (!scratch.time_manager->soft_limit_reached()) {
    return false;
  }
  if (scratch.abort_requested) {
    scratch.abort_requested->store(true, std::memory_order_relaxed);
  }
  return true;
}

SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta,
                               SideToMove stm, const EvaluatorFn& evaluator,
                               SearchScratch& scratch, int ply,
                               int repetition_start, int ply_from_root,
                               bool is_pv, std::uint64_t& nodes,
                               const Move* parent_move,
                               const uint32_t* excluded_root_moves,
                               std::size_t excluded_root_count);

/**
 * @brief Runs a shallow internal iterative deepening (IID) search.
 *
 * Used to seed move ordering when no TT move is available. The search is
 * intentionally shallow and uses a full window to discover a plausible
 * principal move without skewing bounds.
 */
bool run_internal_iterative_deepening(Board& board, SideToMove stm, int depth,
                                      const EvaluatorFn& evaluator,
                                      SearchScratch& scratch, int ply,
                                      int repetition_start, int ply_from_root,
                                      std::uint64_t& nodes, Move& out_move) {
  if (depth < 3) {
    return false;
  }

  const int iid_depth = std::max(1, depth - 2);
  SearchResult iid = alphabeta_minimax(
      board, iid_depth, -INF, INF, stm, evaluator, scratch, ply,
      repetition_start, ply_from_root, false, nodes, nullptr, nullptr, 0);
  if (iid.aborted) {
    return false;
  }
  if (iid.best_move.moving_pc == OccupancyType::empty) {
    return false;
  }
  out_move = iid.best_move;
  return true;
}

inline SearchResult make_aborted_result() {
  SearchResult aborted{};
  aborted.aborted = true;
  aborted.pv_length = 0;
  return aborted;
}

struct MoveEvaluationContext {
  int depth = 0;
  int alpha = -INF;
  int beta = INF;
  bool is_pv = false;
  bool parent_in_check = false;
  int move_number = 1;
  SideToMove stm = SideToMove::White;
  int ply = 0;
  int ply_from_root = 0;
  int repetition_start = 0;
  bool allow_pvs = true;
  const Move* parent_move = nullptr;
};

struct MoveEvaluationResult {
  SearchResult child;
  int score = 0;
  bool aborted = false;
};

// This is it, the core move evaluation function, calling alphabeta_minimax()
MoveEvaluationResult evaluate_move(Board& board, const Move& move,
                                   const MoveEvaluationContext& ctx,
                                   SearchScratch& scratch, std::uint64_t& nodes,
                                   const EvaluatorFn& evaluator) {
  MoveEvaluationResult output{};

  const bool is_capture = move.captured_pc != OccupancyType::empty;
  const bool is_promo = move.promo_pc != OccupancyType::empty;
  const bool quiet_like = !is_capture && !is_promo;
  const bool king_move =
      move.moving_pc == OccupancyType::wK || move.moving_pc == OccupancyType::bK;
  const bool castle_move =
      flag_is_castle(move.flags) || flag_is_long_castle(move.flags);

  int child_depth = ctx.depth - 1;
  const int move_number = ctx.move_number;

  // Search tuning: extend on checking moves and avoid LMR on known-good
  // killer/counter replies to preserve tactical reliability.
  const uint32_t move_code =
      encode_move(move.from, move.to, move.moving_pc, move.captured_pc,
                  move.promo_pc, move.flags);
  bool is_killer = false;
  if (scratch.killers && ctx.ply >= 0 && ctx.ply < static_cast<int>(MAX_PLY)) {
    const auto idx = static_cast<std::size_t>(ctx.ply);
    const uint32_t primary = scratch.killers->primary[idx];
    const uint32_t secondary = scratch.killers->secondary[idx];
    is_killer = (move_code == primary) || (move_code == secondary);
  }
  uint32_t counter_code = 0;
  if (ctx.parent_move) {
    counter_code = scratch.ordering.counter_move(*ctx.parent_move);
  }
  const bool is_counter = (counter_code != 0 && move_code == counter_code);

  Undo undo = make_move(board, move);
  scratch.nnue_adapter->push_move(move);

  const bool irreversible = move_is_irreversible(move);
  const int next_repetition_start =
      irreversible ? (ctx.ply + 1) : ctx.repetition_start;

  const bool in_check_after_move = is_check(board, flip_side(ctx.stm));
  if (in_check_after_move && ctx.ply < static_cast<int>(MAX_PLY) - 1) {
    child_depth += CHECK_EXTENSION;
  }

  const int history_score = scratch.ordering.history_score(move);

  int reduction = 0;
  // LATER MOVE REDUCTION
  if (quiet_like && !king_move && !castle_move && !in_check_after_move &&
      child_depth >= 2 && move_number > 1 && !is_killer && !is_counter) {

    const double depth_factor = std::log1p(std::min(child_depth, 63));
    const double order_factor = std::log1p(std::min(move_number, 63));

    const auto& sparams = search_params();
    double scaled =
        sparams.lmr_intercept +
        (depth_factor * order_factor) / std::max(sparams.lmr_divisor, 0.01);
    if (ctx.is_pv) {
      scaled -= sparams.lmr_pv_offset;
    }
    scaled -= static_cast<double>(history_score) /
              std::max(sparams.lmr_history_divisor, 1.0);

    const int tentative = static_cast<int>(std::round(scaled));
    reduction = std::clamp(tentative, 0, child_depth - 1);
  }

  int search_depth = std::max(0, child_depth);

  auto run_search = [&](int depth_to_use, int a_val, int b_val,
                        bool pv_flag) -> SearchResult {
    SearchResult res = alphabeta_minimax(
        board, depth_to_use, a_val, b_val, flip_side(ctx.stm), evaluator,
        scratch, ctx.ply + 1, next_repetition_start, ctx.ply_from_root + 1,
        pv_flag, nodes, &move, nullptr, 0);
    res.score = normalize_mate_score(res.score, ctx.ply);
    return res;
  };

  bool need_full_search = true;
  SearchResult child = {};
  if (reduction > 0) {
    const int reduced_depth = std::max(0, search_depth - reduction);
    child = run_search(reduced_depth, ctx.alpha, ctx.beta, false);
    if (child.aborted) {
      undo_move(board, undo);
      scratch.nnue_adapter->pop_move();
      if (ctx.ply + 1 < MAX_PLY && scratch.history) {
        scratch.history
            ->repetition_counts[static_cast<std::size_t>(ctx.ply + 1)] = 0;
      }
      output.child = child;
      output.score = child.score;
      output.aborted = true;
      return output;
    }
    if (ctx.stm == SideToMove::White) {
      need_full_search = child.score > ctx.alpha;
    } else {
      need_full_search = child.score < ctx.beta;
    }
  }

  bool use_pvs = false;
  int narrow_alpha = ctx.alpha;
  int narrow_beta = ctx.beta;

  if (need_full_search) {
    const bool is_first_move = (move_number == 1);
    use_pvs = ctx.allow_pvs && ctx.is_pv && !is_first_move &&
              (search_depth >= 0) && (ctx.alpha > -MATE_BOUND) &&
              ctx.beta < MATE_BOUND && (ctx.beta - ctx.alpha) > 1;
    if (ctx.stm == SideToMove::White) {
      narrow_beta = std::min(ctx.beta, ctx.alpha + 1);
      use_pvs = use_pvs && narrow_alpha <= narrow_beta;
    } else {
      narrow_alpha = std::max(ctx.alpha, ctx.beta - 1);
      use_pvs = use_pvs && narrow_beta >= narrow_alpha;
    }

    if (use_pvs) {
      child = run_search(search_depth, narrow_alpha, narrow_beta, true);
      if (child.aborted) {
        undo_move(board, undo);
        scratch.nnue_adapter->pop_move();
        if (ctx.ply + 1 < MAX_PLY && scratch.history) {
          scratch.history
              ->repetition_counts[static_cast<std::size_t>(ctx.ply + 1)] = 0;
        }
        output.child = child;
        output.score = child.score;
        output.aborted = true;
        return output;
      }
      const bool need_retry = search_detail::should_retry_pvs(
          ctx.stm, child.score, narrow_alpha, narrow_beta);
      if (need_retry) {
        if (search_stats_enabled()) {
          search_stats().pvs_researches.fetch_add(1, std::memory_order_relaxed);
        }
        child = run_search(search_depth, ctx.alpha, ctx.beta, ctx.is_pv);
        if (child.aborted) {
          undo_move(board, undo);
          scratch.nnue_adapter->pop_move();
          if (ctx.ply + 1 < MAX_PLY && scratch.history) {
            scratch.history
                ->repetition_counts[static_cast<std::size_t>(ctx.ply + 1)] = 0;
          }
          output.child = child;
          output.score = child.score;
          output.aborted = true;
          return output;
        }
      }
    } else {
      child = run_search(search_depth, ctx.alpha, ctx.beta,
                         ctx.is_pv && is_first_move);
      if (child.aborted) {
        undo_move(board, undo);
        scratch.nnue_adapter->pop_move();
        if (ctx.ply + 1 < MAX_PLY && scratch.history) {
          scratch.history
              ->repetition_counts[static_cast<std::size_t>(ctx.ply + 1)] = 0;
        }
        output.child = child;
        output.score = child.score;
        output.aborted = true;
        return output;
      }
    }
  }

  const int score = child.score;

  undo_move(board, undo);
  scratch.nnue_adapter->pop_move();

  if (ctx.ply + 1 < MAX_PLY && scratch.history) {
    scratch.history->repetition_counts[static_cast<std::size_t>(ctx.ply + 1)] =
        0;
  }

  output.child = child;
  output.score = score;
  output.aborted = child.aborted;
  return output;
}

struct LazySmpResult {
  Move move;
  int move_number = 0;
  SearchResult child;
  int score = 0;
  std::uint64_t nodes = 0;
  bool aborted = false;
};

struct LazySmpSplitPoint {
  void add_task() {
    pending.fetch_add(1, std::memory_order_relaxed);
  }

  void submit(LazySmpResult&& result) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ready.emplace_back(std::move(result));
    }
    pending.fetch_sub(1, std::memory_order_relaxed);
    cv.notify_one();
  }

  bool try_pop(LazySmpResult& out) {
    std::lock_guard<std::mutex> lock(mutex);
    if (ready.empty()) {
      return false;
    }
    out = std::move(ready.back());
    ready.pop_back();
    return true;
  }

  bool wait_pop(LazySmpResult& out) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&]() {
      return !ready.empty() || pending.load(std::memory_order_relaxed) == 0;
    });
    if (ready.empty()) {
      return false;
    }
    out = std::move(ready.back());
    ready.pop_back();
    return true;
  }

  bool has_pending() const {
    return pending.load(std::memory_order_relaxed) > 0;
  }

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<LazySmpResult> ready;
  std::atomic<bool> cancel_flag{false};
  std::atomic<int> pending{0};
};

struct LazySmpTask {
  std::shared_ptr<LazySmpSplitPoint> split_point;
  Board board;
  MoveHistory history;
  KillerTable killers;
  MoveOrderingTables ordering;
  bool has_history = false;
  bool has_killers = false;
  Move move{};
  int move_number = 0;
  MoveEvaluationContext eval_ctx;
  bool has_parent_move = false;
  Move parent_move_storage{};
  std::atomic<bool>* abort_flag = nullptr;

  static void* operator new(std::size_t size) {
    return ::operator new(size,
                          static_cast<std::align_val_t>(alignof(LazySmpTask)));
  }

  static void operator delete(void* ptr) noexcept {
    ::operator delete(ptr, static_cast<std::align_val_t>(alignof(LazySmpTask)));
  }

  static void operator delete(void* ptr, std::size_t size) noexcept {
    (void)size;
    ::operator delete(ptr, static_cast<std::align_val_t>(alignof(LazySmpTask)));
  }
};

struct LazySmpContext {
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::deque<std::unique_ptr<LazySmpTask>> queue;
  bool shutting_down = false;

  const EvaluatorFn* evaluator = nullptr;
  TranspositionTable* tt = nullptr;
  TimeManager* time_manager = nullptr;
  std::atomic<bool> internal_abort{false};
  std::atomic<bool>* abort_flag = nullptr;
  int helper_count = 0;
  int min_split_depth = 5;

  void enqueue(std::unique_ptr<LazySmpTask> task) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      queue.emplace_back(std::move(task));
    }
    queue_cv.notify_one();
  }

  bool pop(std::unique_ptr<LazySmpTask>& out) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [&]() { return shutting_down || !queue.empty(); });
    if (shutting_down && queue.empty()) {
      return false;
    }
    out = std::move(queue.front());
    queue.pop_front();
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      shutting_down = true;
    }
    queue_cv.notify_all();
  }
};

// owned by each worker thread
struct LazySmpThread {
  LazySmpContext* context = nullptr;
  std::unique_ptr<NnueAdapter> nnue_adapter = nullptr;
  int id = 0;
};

thread_local LazySmpThread* tls_lazy_thread = nullptr;

void execute_lazy_task(LazySmpTask& task, LazySmpThread& thread) {
  LazySmpContext* ctx = thread.context;
  if (!ctx) {
    return;
  }

  thread.nnue_adapter->reset(task.board);

  SearchScratch scratch{};
  scratch.history = task.has_history ? &task.history : nullptr;
  scratch.killers = task.has_killers ? &task.killers : nullptr;
  scratch.tt = ctx->tt;
  scratch.ordering = task.ordering;
  scratch.time_manager = ctx->time_manager;
  scratch.abort_requested = task.abort_flag ? task.abort_flag : ctx->abort_flag;
  scratch.local_abort = &task.split_point->cancel_flag;
  scratch.thread = &thread;
  // more convenint to keep threadless implementation
  // stash it in the scratch, don't rely on tls
  scratch.nnue_adapter = thread.nnue_adapter.get();

  std::uint64_t nodes = 0;
  MoveEvaluationContext eval_ctx = task.eval_ctx;
  if (task.has_parent_move) {
    eval_ctx.parent_move = &task.parent_move_storage;
  } else {
    eval_ctx.parent_move = nullptr;
  }

  MoveEvaluationResult eval = evaluate_move(task.board, task.move, eval_ctx,
                                            scratch, nodes, *ctx->evaluator);

  LazySmpResult result{};
  result.move = task.move;
  result.move_number = task.move_number;
  result.child = std::move(eval.child);
  result.score = eval.score;
  result.nodes = nodes;
  result.aborted = eval.aborted;

  task.split_point->submit(std::move(result));
}

// wait for tasks and execute them
void lazy_worker_loop(LazySmpThread& thread) {
  tls_lazy_thread = &thread;
  std::unique_ptr<LazySmpTask> task;
  while (thread.context && thread.context->pop(task)) {
    execute_lazy_task(*task, thread);
    task.reset();
  }
  tls_lazy_thread = nullptr;
}

#if CHESS_USE_POSIX_THREADS
struct LazyWorkerPayload {
  LazySmpThread* thread = nullptr;
};

// entry point for POSIX thread
void* lazy_worker_entry(void* payload_ptr) {
  std::unique_ptr<LazyWorkerPayload> payload(
      static_cast<LazyWorkerPayload*>(payload_ptr));
  if (!payload || !payload->thread) {
    return nullptr;
  }
  lazy_worker_loop(*payload->thread);
  return nullptr;
}
#endif

inline void update_history(SearchScratch& scratch, const Move& move, int depth,
                           bool success) {
  scratch.ordering.record_history(move, depth, success);
  scratch.ordering.maybe_decay_history();
}

inline void record_counter(SearchScratch& scratch, const Move* parent,
                           const Move& reply) {
  if (!parent) {
    return;
  }
  scratch.ordering.record_counter(*parent, reply);
}

inline void record_continuation(SearchScratch& scratch, const Move* parent,
                                const Move& move, int depth, bool success) {
  if (!parent) {
    return;
  }
  scratch.ordering.record_continuation(*parent, move, depth, success);
}

inline void store_killer_move(SearchScratch& scratch, int ply,
                              const Move& move) {
  if (scratch.killers == nullptr || ply < 0 || ply >= MAX_PLY) {
    return;
  }
  const bool quiet = (move.captured_pc == OccupancyType::empty) &&
                     (move.promo_pc == OccupancyType::empty);
  if (!quiet) {
    return;
  }

  const uint32_t code = encode_move(move.from, move.to, move.moving_pc,
                                    move.captured_pc, move.promo_pc, move.flags);

  auto& primary = scratch.killers->primary[static_cast<std::size_t>(ply)];
  auto& secondary = scratch.killers->secondary[static_cast<std::size_t>(ply)];

  if (primary == code) {
    return;
  }
  if (secondary == code) {
    std::swap(primary, secondary);
    return;
  }
  secondary = primary;
  primary = code;
}

SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta,
                               SideToMove stm, const EvaluatorFn& evaluator,
                               SearchScratch& scratch, int ply,
                               int repetition_start, int ply_from_root,
                               bool is_pv, std::uint64_t& nodes,
                               const Move* parent_move,
                               const uint32_t* excluded_root_moves,
                               std::size_t excluded_root_count) {
  if (scratch.use_nodes && scratch.node_limit > 0 &&
      nodes >= scratch.node_limit) {
    if (scratch.abort_requested) {
      scratch.abort_requested->store(true, std::memory_order_relaxed);
    }
    return make_aborted_result();
  }
  ++nodes;

  if (should_abort_due_to_time(scratch, nodes)) {
    return make_aborted_result();
  }

  MoveHistory* history = scratch.history;
  TranspositionTable* tt = scratch.tt;

  // Check for root move exclusions, which could be used to
  // implement multi-PV search by excluding already searched moves
  // currently not used at all, added for future use
  const bool apply_root_exclusions = excluded_root_moves != nullptr &&
                                     excluded_root_count > 0 &&
                                     ply_from_root == 0;

  auto is_excluded_code = [&](uint32_t encoded) -> bool {
    if (!apply_root_exclusions) {
      return false;
    }
    for (std::size_t idx = 0; idx < excluded_root_count; ++idx) {
      if (excluded_root_moves[idx] == encoded) {
        return true;
      }
    }
    return false;
  };

  auto is_excluded_move = [&](const Move& move) -> bool {
    if (!apply_root_exclusions || move.moving_pc == OccupancyType::empty) {
      return false;
    }
    const uint32_t encoded =
        encode_move(move.from, move.to, move.moving_pc, move.captured_pc,
                    move.promo_pc, move.flags);
    return is_excluded_code(encoded);
  };

  // check for repetitions
  if (history) {
    if (ply < MAX_PLY) {
      history->key_history[static_cast<std::size_t>(ply)] = board.position_key;
      history->ply_count = std::max(history->ply_count, ply + 1);
      uint8_t repeats = 0;
      const int repeat_begin = std::max(0, std::min(repetition_start, ply));
      for (int i = repeat_begin; i < ply; ++i) {
        if (history->key_history[static_cast<std::size_t>(i)] ==
            board.position_key) {
          ++repeats;
        }
      }
      history->repetition_counts[static_cast<std::size_t>(ply)] = repeats;
      if (repeats >= 2) {
        const int repetition_score = (stm == SideToMove::White)
                                         ? -REPETITION_PENALTY
                                         : REPETITION_PENALTY;
        SearchResult repetition{};
        repetition.score = repetition_score;
        repetition.best_move = Move{};
        repetition.outcome = SearchResult::Outcome::DrawByRepetition;
        repetition.pv_length = 0;
        return repetition;
      }
    }
  }

  // TRANSPOSITION TABLE PROBE, check if we have a cached move stored
  TranspositionEntry cached_entry;
  bool has_cached_move = false;
  Move cached_move{};
  bool tt_hit = false;
  if (tt) {
    if (search_stats_enabled()) {
      const auto start = std::chrono::steady_clock::now();
      tt_hit = tt->probe(board.position_key, cached_entry);
      const auto end = std::chrono::steady_clock::now();
      const std::uint64_t elapsed = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count());
      auto& stats = search_stats();
      stats.tt_probes.fetch_add(1, std::memory_order_relaxed);
      if (tt_hit) {
        stats.tt_hits.fetch_add(1, std::memory_order_relaxed);
      }
      stats.tt_probe_time_ns.fetch_add(elapsed, std::memory_order_relaxed);
    } else {
      tt_hit = tt->probe(board.position_key, cached_entry);
    }
  }
  if (tt_hit) {
    if (!is_excluded_move(cached_entry.best_move)) {
      const int cached_score = normalize_mate_score(
          TranspositionTable::decode_score(cached_entry.score, ply), ply);
      if (cached_entry.depth >= depth) {
        switch (cached_entry.flag) {
        case TranspositionFlag::Exact: {
          SearchResult tt_result{};
          tt_result.score = cached_score;
          tt_result.best_move = cached_entry.best_move;
          tt_result.outcome = SearchResult::Outcome::InProgress;
          if (tt_result.best_move.moving_pc != OccupancyType::empty) {
            tt_result.principal_variation.clear();
            tt_result.principal_variation.push_back(tt_result.best_move);
            tt_result.pv_length = 1;
          }
          if (search_stats_enabled()) {
            search_stats().tt_cutoffs.fetch_add(1, std::memory_order_relaxed);
          }
          return tt_result;
        }
        case TranspositionFlag::LowerBound:
          alpha = std::max(alpha, cached_score);
          break;
        case TranspositionFlag::UpperBound:
          beta = std::min(beta, cached_score);
          break;
        case TranspositionFlag::None:
          break;
        }
        if (alpha >= beta) {
          SearchResult tt_cutoff{};
          tt_cutoff.score = cached_score;
          tt_cutoff.best_move = cached_entry.best_move;
          tt_cutoff.outcome = SearchResult::Outcome::InProgress;
          if (tt_cutoff.best_move.moving_pc != OccupancyType::empty) {
            tt_cutoff.principal_variation.clear();
            tt_cutoff.principal_variation.push_back(tt_cutoff.best_move);
            tt_cutoff.pv_length = 1;
          }
          if (search_stats_enabled()) {
            search_stats().tt_cutoffs.fetch_add(1, std::memory_order_relaxed);
          }
          return tt_cutoff;
        }
      }
      if (cached_entry.best_move.moving_pc != OccupancyType::empty) {
        has_cached_move = true;
        cached_move = cached_entry.best_move;
      }
    }
  }

  // IID: if no TT move is available at PV nodes, run a shallow search to
  // seed move ordering with a likely principal move.
  if (!has_cached_move && is_pv) {
    Move iid_move{};
    if (run_internal_iterative_deepening(board, stm, depth, evaluator, scratch,
                                         ply, repetition_start, ply_from_root,
                                         nodes, iid_move) &&
        !is_excluded_move(iid_move)) {
      has_cached_move = true;
      cached_move = iid_move;
    }
  }

  // NULL MOVE PRUNING
  // Ahead of main search loop to see if we can prune immediately,
  // ie. can this side pass and still be >= beta
  const auto& sparams = search_params();
  const int null_min_depth =
      std::max(sparams.null_move_min_depth, sparams.null_move_reduction + 1);
  if (allow_null_move(board, depth, null_min_depth)) {
    UndoNull undo_null = make_null_move(board);
    scratch.nnue_adapter->push_null();
    const int divisor = std::max(1, sparams.null_move_reduction_divisor);
    int null_move_reduction = sparams.null_move_reduction + depth / divisor;
    if (depth <= null_min_depth) {
      null_move_reduction =
          std::min(null_move_reduction, sparams.null_move_reduction);
    }
    null_move_reduction = std::min(null_move_reduction, std::max(0, depth - 1));
    SearchResult null_result = alphabeta_minimax(
        board, std::max(0, depth - 1 - null_move_reduction), beta - 1, beta,
        flip_side(stm), evaluator, scratch, ply + 1, repetition_start,
        ply_from_root + 1, false, nodes, nullptr, nullptr, 0);
    if (null_result.aborted) {
      undo_null_move(board, undo_null);
      scratch.nnue_adapter->pop_null();
      return null_result;
    }
    const int score_after_null = normalize_mate_score(null_result.score, ply);
    undo_null_move(board, undo_null);
    scratch.nnue_adapter->pop_null();
    if (score_after_null >= beta) {
      if (search_stats_enabled()) {
        search_stats().null_move_cutoffs.fetch_add(1, std::memory_order_relaxed);
      }
      SearchResult cutoff{};
      cutoff.score = score_after_null;
      cutoff.outcome = SearchResult::Outcome::InProgress;
      cutoff.pv_length = 0;
      return cutoff;
    }
  }

  // capture the TT-adjusted window for later fail-low/high classification
  int alpha_base = alpha;
  int beta_base = beta;

  bool do_quiescence = true;
  // QUIESCENCE SEARCH
  if (depth == 0) {

    int qs_raw = 0;
    if (do_quiescence) {
      qs_raw = quiescence(board, alpha, beta, stm, evaluator,
                          scratch.nnue_adapter, nodes, tt, ply);
    } else {
      int eval =
          evaluator(static_cast<const Board&>(board), scratch.nnue_adapter);
      qs_raw = eval;
    }
    const int qs = normalize_mate_score(qs_raw, ply);
    if (tt) {
      TranspositionEntry entry;
      if (tt->probe(board.position_key, entry)) {
        if (entry.depth <= 0) {
          tt->store(board.position_key, 0, qs, entry.flag, entry.best_move, ply);
        }
      } else {
        tt->store(board.position_key, 0, qs, TranspositionFlag::Exact, Move{},
                  ply);
      }
    }
    SearchResult leaf{};
    leaf.score = qs;
    leaf.outcome = SearchResult::Outcome::InProgress;
    leaf.pv_length = 0;
    return leaf;
  }

  // Cache static eval so reverse-futility and later quiet-move futility share
  // it.
  bool static_eval_ready = false;
  int static_eval = 0;
  int static_eval_normalized = 0;

  // reverse futility pruning: early cutoff in non-PV, non-check nodes
  if (!is_pv && depth >= 3 && depth <= 8 && !is_check(board, stm)) {
    const int eval =
        evaluator(static_cast<const Board&>(board), scratch.nnue_adapter);
    const int margin_idx = std::min(depth, 3);
    const int margin =
        sparams.futility_margins[static_cast<std::size_t>(margin_idx)];
    const int eval_normalized = normalize_mate_score(eval, ply);
    static_eval = eval;
    static_eval_normalized = eval_normalized;
    static_eval_ready = true;
    if (stm == SideToMove::White) {
      if (eval_normalized - margin >= beta) {
        SearchResult prune{};
        prune.score = eval_normalized;
        prune.outcome = SearchResult::Outcome::InProgress;
        prune.pv_length = 0;
        return prune;
      }
    } else {
      if (eval_normalized + margin <= alpha) {
        SearchResult prune{};
        prune.score = eval_normalized;
        prune.outcome = SearchResult::Outcome::InProgress;
        prune.pv_length = 0;
        return prune;
      }
    }
  }

  // razor pruning: shallow check before full search at low depth in non-PV
  // nodes to save eval calls when the position is already far below alpha.
  if (!is_pv && depth <= 3 && !is_check(board, stm)) {
    if (!static_eval_ready) {
      const int eval =
          evaluator(static_cast<const Board&>(board), scratch.nnue_adapter);
      static_eval = eval;
      static_eval_normalized = normalize_mate_score(eval, ply);
      static_eval_ready = true;
    }
    const int margin_idx = std::min(depth, 3);
    const int razor_margin =
        sparams.futility_margins[static_cast<std::size_t>(margin_idx)] +
        sparams.razor_margin_bonus;
    bool do_razor = false;
    if (stm == SideToMove::White) {
      do_razor = static_eval_normalized + razor_margin <= alpha;
    } else {
      do_razor = static_eval_normalized - razor_margin >= beta;
    }
    if (do_razor) {
      const int qs =
          normalize_mate_score(quiescence(board, alpha, beta, stm, evaluator,
                                          scratch.nnue_adapter, nodes, tt, ply),
                               ply);
      if (stm == SideToMove::White) {
        if (qs <= alpha) {
          SearchResult prune{};
          prune.score = qs;
          prune.outcome = SearchResult::Outcome::InProgress;
          prune.pv_length = 0;
          return prune;
        }
      } else {
        if (qs >= beta) {
          SearchResult prune{};
          prune.score = qs;
          prune.outcome = SearchResult::Outcome::InProgress;
          prune.pv_length = 0;
          return prune;
        }
      }
    }
  }
  uint16_t move_count = 0;
  // GENERATE LEGAL MOVES, filtering excluded root moves if any and
  // then sorting them
  std::array<uint32_t, kMaxMovementCount> moves{};
  if (search_stats_enabled()) {
    const auto start = std::chrono::steady_clock::now();
    moves = generate_legal_moves(board, stm, move_count);
    const auto end = std::chrono::steady_clock::now();
    const std::uint64_t elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count());
    auto& stats = search_stats();
    stats.movegen_calls.fetch_add(1, std::memory_order_relaxed);
    stats.movegen_time_ns.fetch_add(elapsed, std::memory_order_relaxed);
  } else {
    moves = generate_legal_moves(board, stm, move_count);
  }

  if (apply_root_exclusions) {
    uint16_t write_idx = 0;
    for (uint16_t i = 0; i < move_count; ++i) {
      const uint32_t code = moves[i];
      if (is_excluded_code(code)) {
        continue;
      }
      moves[write_idx++] = code;
    }
    move_count = write_idx;
  }

  // out of moves: check for checkmate or stalemate
  if (move_count == 0) {
    if (is_check(board, stm)) {
      const int mate_score = normalize_mate_score(
          (stm == SideToMove::White) ? -MATE_VALUE : MATE_VALUE, ply);
      if (tt) {
        tt->store(board.position_key, depth, mate_score,
                  TranspositionFlag::Exact, Move{}, ply);
      }
      SearchResult mate{};
      mate.score = mate_score;
      mate.outcome = SearchResult::Outcome::Mate;
      mate.pv_length = 0;
      return mate;
    }
    constexpr int draw_score = 0;
    if (tt) {
      tt->store(board.position_key, depth, draw_score, TranspositionFlag::Exact,
                Move{}, ply);
    }
    SearchResult draw{};
    draw.score = draw_score;
    draw.outcome = SearchResult::Outcome::DrawByStalemate;
    draw.pv_length = 0;
    return draw;
  }

  /**
   * @brief ProbCut: shallow tactical search on promising captures.
   *
   * If a reduced search on a good capture already fails high/low by a margin,
   * we can prune the node early. This saves eval work at deeper time controls
   * while retaining tactical reliability.
   */
  if (!is_pv && depth >= 6 && !is_check(board, stm)) {
    const int probcut_depth = depth - sparams.probcut_reduction;
    if (probcut_depth >= 1) {
      const int bound = (stm == SideToMove::White) ? beta : alpha;
      const int alpha_pc =
          (stm == SideToMove::White) ? bound : bound - sparams.probcut_margin;
      const int beta_pc =
          (stm == SideToMove::White) ? bound + sparams.probcut_margin : bound;
      int captures_checked = 0;
      for (uint16_t i = 0; i < move_count; ++i) {
        const Move move = decode_move(moves[i]);
        if (move.captured_pc == OccupancyType::empty) {
          continue;
        }
        if (static_exchange_eval(board, move) < 0) {
          continue;
        }
        if (sparams.probcut_max_captures > 0 &&
            ++captures_checked > sparams.probcut_max_captures) {
          break;
        }
        const Undo undo = make_move(board, move);
        scratch.nnue_adapter->push_move(move);
        const bool irreversible = move_is_irreversible(move);
        const int next_repetition_start =
            irreversible ? (ply + 1) : repetition_start;
        SearchResult probcut = alphabeta_minimax(
            board, probcut_depth, alpha_pc, beta_pc, flip_side(stm), evaluator,
            scratch, ply + 1, next_repetition_start, ply_from_root + 1, false,
            nodes, &move, nullptr, 0);
        scratch.nnue_adapter->pop_move();
        undo_move(board, undo);
        if (probcut.aborted) {
          return probcut;
        }
        const int probcut_score = normalize_mate_score(probcut.score, ply);
        if ((stm == SideToMove::White && probcut_score >= beta_pc) ||
            (stm == SideToMove::Black && probcut_score <= alpha_pc)) {
          SearchResult cutoff{};
          cutoff.score = probcut_score;
          cutoff.outcome = SearchResult::Outcome::InProgress;
          cutoff.pv_length = 0;
          return cutoff;
        }
      }
    }
  }

  uint32_t tt_code = 0;
  if (has_cached_move) {
    if (is_excluded_move(cached_move)) {
      has_cached_move = false;
    } else {
      // we'll encode it for sorting, same format as generated moves
      tt_code = encode_move(cached_move.from, cached_move.to,
                            cached_move.moving_pc, cached_move.captured_pc,
                            cached_move.promo_pc, cached_move.flags);
    }
  }
  // get history matrix pointer for sorting
  const auto* history_matrix = scratch.ordering.history_matrix();
  const uint32_t counter_code =
      parent_move ? scratch.ordering.counter_move(*parent_move) : 0;

  sort_moves(board, moves, move_count, tt_code, scratch.killers, history_matrix,
             ply, counter_code, scratch.ordering.continuation_table(),
             parent_move);

  // MAIN SEARCH LOOP, after move generation and ordering
  SearchResult best{};
  best.score = (stm == SideToMove::White) ? -INF : INF;
  best.outcome = SearchResult::Outcome::InProgress;
  best.pv_length = 0;

  const bool parent_in_check = is_check(board, stm);
  int moves_tried = 0;

  // Prepare move evaluation context for SMP tasks and main loop
  MoveEvaluationContext move_ctx{};
  move_ctx.depth = depth;
  move_ctx.is_pv = is_pv;
  move_ctx.stm = stm;
  move_ctx.ply = ply;
  move_ctx.ply_from_root = ply_from_root;
  move_ctx.repetition_start = repetition_start;
  move_ctx.parent_in_check = parent_in_check;
  move_ctx.parent_move = parent_move;
  move_ctx.allow_pvs = true;

  LazySmpThread* smp_thread = scratch.thread ? scratch.thread : tls_lazy_thread;
  LazySmpContext* smp_ctx = smp_thread ? smp_thread->context : nullptr;
  const bool smp_supported =
      smp_ctx && smp_ctx->helper_count > 0 && smp_thread && smp_thread->id == 0;
  std::shared_ptr<LazySmpSplitPoint> split_point;
  LazySmpSplitPoint* split_raw = nullptr;
  // done setting up SMP split point and tasks

  bool cutoff_triggered = false;

  // Lambda to apply the result of a move evaluation, whether from
  // the main thread or from an SMP helper
  auto apply_result = [&](const Move& current_move,
                          const SearchResult& child_result,
                          int score_value) -> bool {
    if (cutoff_triggered) {
      return true;
    }

    const bool quiet_like = (current_move.captured_pc == OccupancyType::empty) &&
                            (current_move.promo_pc == OccupancyType::empty);

    const bool is_better =
        (best.best_move.moving_pc == OccupancyType::empty) ? true
        : (stm == SideToMove::White) ? (score_value > best.score)
                                     : (score_value < best.score);
    if (is_better) {
      best.score = score_value;
      best.best_move = current_move;
      best.outcome = child_result.outcome;

      best.principal_variation.clear();
      best.principal_variation.reserve(child_result.principal_variation.size() +
                                       1);
      best.principal_variation.push_back(current_move);
      best.principal_variation.insert(best.principal_variation.end(),
                                      child_result.principal_variation.begin(),
                                      child_result.principal_variation.end());
      best.pv_length = static_cast<int>(best.principal_variation.size());
      update_history(scratch, current_move, depth, true);
      record_counter(scratch, parent_move, current_move);
      record_continuation(scratch, parent_move, current_move, depth, true);
    } else {
      const bool fail_low = (stm == SideToMove::White)
                                ? (score_value <= alpha_base)
                                : (score_value >= beta_base);
      if (fail_low) {
        update_history(scratch, current_move, depth, false);
        record_continuation(scratch, parent_move, current_move, depth, false);
      }
    }

    bool cutoff_now = false;
    if (stm == SideToMove::White) {
      if (score_value > alpha) {
        alpha = score_value;
      }
      cutoff_now = alpha >= beta;
    } else {
      if (score_value < beta) {
        beta = score_value;
      }
      cutoff_now = beta <= alpha;
    }

    if (cutoff_now) {
      cutoff_triggered = true;
      if (quiet_like) {
        store_killer_move(scratch, ply, current_move);
      }
      if (split_raw) {
        split_raw->cancel_flag.store(true, std::memory_order_relaxed);
      }
    }

    return cutoff_now;
  };

  // Iterate over all generated moves
  for (uint16_t i = 0; i < move_count; ++i) {
    const uint32_t move_code = moves[i];
    Move move = decode_move(move_code);
    const bool is_capture = move.captured_pc != OccupancyType::empty;
    const bool is_promo = move.promo_pc != OccupancyType::empty;
    const bool quiet_like = !is_capture && !is_promo;

    const int move_number = moves_tried + 1;
    int child_depth_lmp = depth - 1;
    // LATE MOVE PRUNING, avoid searching very late quiet moves
    // except in PV nodes or when in check
    if (!is_pv && !parent_in_check && quiet_like && child_depth_lmp > 0 &&
        child_depth_lmp <= 3) {
      static constexpr int lmp_limits[4] = {0, 3, 6, 9};
      const int history_score = scratch.ordering.history_score(move);
      const bool is_counter = (counter_code != 0 && move_code == counter_code);
      bool is_killer = false;
      if (scratch.killers && ply >= 0 && ply < MAX_PLY) {
        const auto idx = static_cast<std::size_t>(ply);
        const uint32_t primary = scratch.killers->primary[idx];
        const uint32_t secondary = scratch.killers->secondary[idx];
        is_killer = (move_code == primary) || (move_code == secondary);
      }
      int limit = lmp_limits[child_depth_lmp];
      if (history_score > 10000) {
        limit += 2;
      }
      if (!is_killer && !is_counter && move_number > limit) {
        if (search_stats_enabled()) {
          search_stats().lmp_prunes.fetch_add(1, std::memory_order_relaxed);
        }
        continue;
      }
    }
    // Skip captures with negative SEE in non-PV nodes,
    // not in check and not TT move
    if (is_capture && !is_pv && !parent_in_check &&
        (!has_cached_move || move_code != tt_code)) {
      const int see = static_exchange_eval(board, move);
      int see_threshold = 0;
      if (depth <= 4) {
        const int captured_value = piece_material_magnitude(move.captured_pc);
        if (captured_value > 0 &&
            captured_value <= sparams.see_capture_max_value) {
          see_threshold = sparams.see_capture_threshold;
        }
      }
      if (see < see_threshold) {
        if (search_stats_enabled()) {
          search_stats().see_prunes.fetch_add(1, std::memory_order_relaxed);
        }
        continue;
      }
    }
    // futility pruning for quiet moves in non-PV nodes
    // conditions for futility pruning
    if (!is_pv && quiet_like && child_depth_lmp <= 3 && child_depth_lmp > 0 &&
        !parent_in_check) {
      if (!static_eval_ready) {
        static_eval =
            evaluator(static_cast<const Board&>(board), scratch.nnue_adapter);
        static_eval_normalized = normalize_mate_score(static_eval, ply);
        static_eval_ready = true;
      }
      const auto margin =
          search_params()
              .futility_margins[static_cast<std::size_t>(child_depth_lmp)];
      const int adjusted_eval = static_eval_normalized + margin;
      if ((stm == SideToMove::White && adjusted_eval <= alpha) ||
          (stm == SideToMove::Black && adjusted_eval >= beta)) {
        if (search_stats_enabled()) {
          search_stats().futility_prunes.fetch_add(1, std::memory_order_relaxed);
        }
        continue;
      }
    }

    moves_tried = move_number;

    const bool can_split =
        smp_supported && depth >= smp_ctx->min_split_depth && move_number > 1;

    // SPLIT POINT CREATION AND TASK ENQUEUEING for SMP
    if (can_split) {
      if (!split_point) {
        split_point = std::make_shared<LazySmpSplitPoint>();
        split_raw = split_point.get();
      }

      auto task = std::make_unique<LazySmpTask>();
      task->split_point = split_point;
      task->board = board;
      task->move = move;
      task->move_number = move_number;
      task->eval_ctx = move_ctx;
      task->eval_ctx.move_number = move_number;
      task->eval_ctx.alpha = alpha;
      task->eval_ctx.beta = beta;
      task->eval_ctx.allow_pvs = false;
      task->eval_ctx.parent_move = nullptr;
      task->eval_ctx.parent_in_check = parent_in_check;
      task->has_history = scratch.history != nullptr;
      if (task->has_history) {
        task->history = *scratch.history;
      }
      task->has_killers = scratch.killers != nullptr;
      if (task->has_killers) {
        task->killers = *scratch.killers;
      }
      task->ordering = scratch.ordering;
      task->has_parent_move = parent_move != nullptr;
      if (task->has_parent_move) {
        task->parent_move_storage = *parent_move;
      }
      task->abort_flag = smp_ctx->abort_flag;
      split_point->add_task();
      smp_ctx->enqueue(std::move(task));
      continue;
    }

    move_ctx.move_number = move_number;
    move_ctx.alpha = alpha;
    move_ctx.beta = beta;
    move_ctx.allow_pvs = true;

    // ACTUAL MOVE EVALUATION IN THE MAIN THREAD
    // recursive call to alphabeta_minimax happens inside evaluate_move()
    MoveEvaluationResult eval =
        evaluate_move(board, move, move_ctx, scratch, nodes, evaluator);

    if (eval.aborted) {
      if (split_raw) {
        split_raw->cancel_flag.store(true, std::memory_order_relaxed);
      }
      return eval.child;
    }

    const bool cutoff_now = apply_result(move, eval.child, eval.score);

    // Collect results from SMP if done // any, don't block
    if (split_point) {
      LazySmpResult ready{};
      // no-blocking collection of available results
      while (split_point->try_pop(ready)) {
        nodes += ready.nodes;
        if (ready.aborted) {
          if (!split_raw ||
              !split_raw->cancel_flag.load(std::memory_order_relaxed)) {
            return ready.child;
          }
          continue;
        }
        apply_result(ready.move, ready.child, ready.score);
      }
    }

    if (cutoff_now) {
      break;
    }
  }

  // FINAL COLLECTION OF SMP RESULTS, blocking until all are in
  if (split_point) {
    LazySmpResult ready{};
    while (split_point->wait_pop(ready)) {
      nodes += ready.nodes;
      if (ready.aborted) {
        if (!split_raw ||
            !split_raw->cancel_flag.load(std::memory_order_relaxed)) {
          return ready.child;
        }
        continue;
      }
      apply_result(ready.move, ready.child, ready.score);
    }
  }

  if (best.aborted) {
    return best;
  }

  // STORE TO TRANSPOSITION TABLE THE RESULT OF THIS NODE's SEARCH
  if (tt) {
    const int normalized_best = normalize_mate_score(best.score, ply);
    best.score = normalized_best;

    TranspositionFlag flag = TranspositionFlag::Exact;
    if (normalized_best <= alpha_base) {
      flag = TranspositionFlag::UpperBound;
    } else if (normalized_best >= beta_base) {
      flag = TranspositionFlag::LowerBound;
    }
    tt->store(board.position_key, depth, normalized_best, flag, best.best_move,
              ply);
  }
  // TODO: clean up transposition table after lots of searches?

  return best;
}

} // namespace

// Main search function, entry point
// Sets up the search scratch space and parameters
// and calls alphabeta_minimax to perform the actual search
SearchResult search_position(Board& board, SideToMove stm,
                             const SearchParameters& params,
                             const EvaluatorFn& evaluator, MoveHistory* history,
                             TranspositionTable* tt, int repetition_start) {
  const bool track_info = static_cast<bool>(params.info_callback);
  const auto search_start = track_info ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
  int base_ply = 0;
  int start_ply = 0;

  // setup history if provided, initialize position key at current ply
  if (history) {
    if (history->ply_count == 0) {
      history->key_history[0] = board.position_key;
      history->ply_count = 1;
    } else if (history->ply_count > 0) {
      const auto idx = static_cast<std::size_t>(history->ply_count - 1);
      history->key_history[idx] = board.position_key;
    }

    base_ply = history->ply_count;
    start_ply = std::max(history->ply_count - 1, 0);
    repetition_start = std::max(0, std::min(repetition_start, start_ply + 1));
  }

  KillerTable killers{};
  std::atomic<bool> abort_requested{false};
  SearchScratch scratch{};
  std::unique_ptr<NnueAdapter> nnue_adapter_owner;
  scratch.ordering.reset();
  scratch.history = history;
  scratch.killers = &killers;
  scratch.tt = tt;
  scratch.time_manager = params.time_manager;
  scratch.use_nodes = params.limits.use_nodes && params.limits.node_limit > 0;
  scratch.node_limit = params.limits.node_limit;
  if (params.abort_flag) {
    scratch.abort_requested = params.abort_flag;
  } else {
    scratch.abort_requested = &abort_requested;
  }
  scratch.local_abort = nullptr;
  scratch.thread = tls_lazy_thread;
  if (scratch.thread) {
    scratch.nnue_adapter = scratch.thread->nnue_adapter.get();
  } else {
    nnue_adapter_owner = std::make_unique<NnueAdapter>(board);
    scratch.nnue_adapter = nnue_adapter_owner.get();
  }

  if (search_stats_enabled()) {
    reset_search_stats();
  }

  double root_complexity_hint = 0.0;
  if (params.time_manager) {
    const ComplexityMetrics metrics = compute_complexity(board, stm);
    root_complexity_hint = normalize_complexity(metrics);
    params.time_manager->set_complexity_hint(root_complexity_hint);
  }

  int max_root_moves = 0;
  uint16_t move_count = 0;

  // maybe early out: threefold repetition, before we start searching
  if (history) {
    const int repeat_begin = std::max(0, std::min(repetition_start, start_ply));
    int repeats = 0;
    for (int idx = repeat_begin; idx < start_ply; ++idx) {
      if (history->key_history[static_cast<std::size_t>(idx)] ==
          board.position_key) {
        ++repeats;
      }
    }
    if (repeats >= 2) {
      SearchResult draw{};
      draw.score = 0;
      draw.best_move = Move{};
      draw.outcome = SearchResult::Outcome::DrawByRepetition;
      draw.nodes = 0;
      draw.elapsed_ms = 0;
      return draw;
    }
  }

  // maybe early out: no legal moves
  generate_legal_moves(board, stm, move_count);
  if (move_count == 0) {
    const bool side_in_check = is_check(board, stm);
    SearchResult terminal{};
    terminal.best_move = Move{};
    terminal.nodes = 0;
    terminal.elapsed_ms = 0;
    if (side_in_check) {
      const int mate_score = normalize_mate_score(
          (stm == SideToMove::White) ? -MATE_VALUE : MATE_VALUE, start_ply);
      terminal.score = mate_score;
      terminal.outcome = SearchResult::Outcome::Mate;
    } else {
      terminal.score = 0;
      terminal.outcome = SearchResult::Outcome::DrawByStalemate;
    }
    return terminal;
  }
  max_root_moves = move_count;
  auto excluded_moves = std::array<uint32_t, kMaxMovementCount>{};
  std::size_t excluded_count = 0;
  if (params.root_excluded_moves && params.root_excluded_count > 0) {
    const std::size_t copy_count =
        std::min<std::size_t>(params.root_excluded_count, excluded_moves.size());
    std::copy_n(params.root_excluded_moves, copy_count, excluded_moves.begin());
    excluded_count = copy_count;
  }
  const std::size_t static_excluded_count = excluded_count;
  const bool multi_pv = params.pv_count > 1;
  const auto& sparams = search_params();
  std::uint64_t nodes = 0;
  int aspiration_window = sparams.aspiration_window_initial;
  SearchResult last_completed_result{};
  SearchResult result{};

  int last_completed_depth = 0;
  bool stopped_by_abort = false;
  bool stopped_by_soft_limit = false;

  int last_completed_score = 0;

  // ITERATIVE DEEPENING LOOP
  // depth means remaining depth to search
  for (int current_depth = 1; current_depth <= params.depth; ++current_depth) {
    if (scratch.time_manager && scratch.time_manager->soft_limit_reached()) {
      stopped_by_soft_limit = true;
      break;
    }
    excluded_count = static_excluded_count;
    int pv_generated = 0;
    SearchResult depth_best{};
    int depth_best_score = 0;
    bool depth_best_set = false;
    while (true) {
      int alpha = -INF;
      int beta = INF;
      // set up aspiration window based on last completed search
      if (current_depth == 1 ||
          last_completed_result.best_move.moving_pc == OccupancyType::empty) {
        alpha = -INF;
        beta = INF;
      } else {
        if (is_mate_score(last_completed_score)) {
          alpha = -INF;
          beta = INF;
          aspiration_window = sparams.aspiration_window_initial;
        } else {
          const int window_low = last_completed_score - aspiration_window;
          const int window_high = last_completed_score + aspiration_window;
          alpha = std::max(window_low, -MATE_BOUND);
          alpha = std::min(alpha, MATE_BOUND);
          beta = std::min(window_high, MATE_BOUND);
          beta = std::max(beta, -MATE_BOUND);
          if (alpha >= beta) {
            alpha = window_low;
            beta = window_high;
            if (alpha >= beta) {
              alpha = -INF;
              beta = INF;
            }
          }
        }
      }

      int window = aspiration_window;
      int attempts = 0;
      bool forced_full_window = false;
      bool iteration_fail_low = false;
      bool iteration_fail_high = false;
      while (true) {
        ++attempts;
        const bool is_pv = true;
        // call the actual search function via alphabeta_minimax
        // this will recurse down the tree and return the result
        result = alphabeta_minimax(
            board, current_depth, alpha, beta, stm, evaluator, scratch,
            start_ply, repetition_start, 0, is_pv, nodes, nullptr,
            excluded_count ? excluded_moves.data() : nullptr, excluded_count);
        result.searched_depth = current_depth;
        result.selective_depth = current_depth;
        if (result.aborted) {
          abort_requested.store(true, std::memory_order_relaxed);
          break;
        }
        // check aspiration window results,
        // First check for fail-low, then fail-high
        // if outside the window we need wider window search
        if (result.score <= alpha) {
          iteration_fail_low = true;
          if (!forced_full_window && attempts >= kMaxAspirationAttempts) {
            alpha = -INF;
            beta = INF;
            window = sparams.aspiration_window_initial;
            forced_full_window = true;
            attempts = 0;
            continue;
          }
          if (is_mate_score(result.score)) {
            if (!forced_full_window) {
              alpha = -INF;
              beta = INF;
              window = sparams.aspiration_window_initial;
              forced_full_window = true;
              attempts = 0;
              continue;
            }
            break;
          }
          if (alpha <= -MATE_BOUND) {
            alpha = -INF;
            beta = INF;
          } else {
            window = std::min(window * 2, sparams.aspiration_window_max);
            alpha = std::max(result.score - window, -MATE_BOUND);
            beta = std::min(result.score + window, MATE_BOUND);
            continue;
          }
        }
        if (result.score >= beta) {
          iteration_fail_high = true;
          if (!forced_full_window && attempts >= kMaxAspirationAttempts) {
            alpha = -INF;
            beta = INF;
            window = sparams.aspiration_window_initial;
            forced_full_window = true;
            attempts = 0;
            continue;
          }
          if (is_mate_score(result.score)) {
            if (!forced_full_window) {
              alpha = -INF;
              beta = INF;
              window = sparams.aspiration_window_initial;
              forced_full_window = true;
              attempts = 0;
              continue;
            }
            break;
          }
          if (beta >= MATE_BOUND) {
            alpha = -INF;
            beta = INF;
          } else {
            window = std::min(window * 2, sparams.aspiration_window_max);
            alpha = std::max(result.score - window, -MATE_BOUND);
            beta = std::min(result.score + window, MATE_BOUND);
            continue;
          }
        }
        break;
      }

      // stop aspiration if no best move found or max root moves reached
      if (result.best_move.moving_pc == OccupancyType::empty ||
          !max_root_moves) {
        if (!depth_best_set) {
          depth_best = result;
          depth_best_score = result.score;
          depth_best_set = true;
        }
        if (iteration_fail_low || iteration_fail_high) {
          aspiration_window =
              std::min(window * 2, sparams.aspiration_window_max);
        } else if (current_depth > 2 &&
                   window > sparams.aspiration_window_initial) {
          aspiration_window =
              std::max(sparams.aspiration_window_initial, window / 2);
        } else {
          aspiration_window = window;
        }
        break;
      }

      ++pv_generated;
      const bool improving =
          depth_best.best_move.moving_pc == OccupancyType::empty ? true
          : (stm == SideToMove::White) ? (result.score > depth_best_score)
                                       : (result.score < depth_best_score);
      if (improving) {
        depth_best = result;
        depth_best_score = result.score;
        depth_best_set = true;
      }

      // check if we can extend the PV search to generate more lines
      // typically not going to be used
      const bool can_extend = multi_pv && pv_generated < params.pv_count;
      if (can_extend && excluded_count < excluded_moves.size()) {
        excluded_moves[excluded_count++] =
            encode_move(result.best_move.from, result.best_move.to,
                        result.best_move.moving_pc, result.best_move.captured_pc,
                        result.best_move.promo_pc, result.best_move.flags);
        aspiration_window = sparams.aspiration_window_initial;
        continue;
      }

      if (iteration_fail_low || iteration_fail_high) {
        aspiration_window = std::min(window * 2, sparams.aspiration_window_max);
      } else if (current_depth > 2 &&
                 window > sparams.aspiration_window_initial) {
        aspiration_window =
            std::max(sparams.aspiration_window_initial, window / 2);
      } else {
        aspiration_window = window;
      }
      break;
    }

    if (abort_requested.load(std::memory_order_relaxed)) {
      stopped_by_abort = true;
      break;
    }

    if (!result.aborted) {
      if (depth_best_set) {
        result = depth_best;
      }
      last_completed_result = result;
      last_completed_score = result.score;
    }

    if (track_info && params.info_callback) {
      result.nodes = nodes;
      result.elapsed_ms = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - search_start)
              .count());
      params.info_callback(result);
    }

    if (scratch.time_manager && scratch.time_manager->soft_limit_reached()) {
      stopped_by_soft_limit = true;
      break;
    }

    last_completed_depth = current_depth;
  }
  if (history) {
    history->ply_count = base_ply;
  }

  SearchResult final_result =
      last_completed_result.best_move.moving_pc != OccupancyType::empty
          ? last_completed_result
          : result;
  if (final_result.best_move.moving_pc == OccupancyType::empty) {
    uint16_t fallback_count = 0;
    auto fallback_moves = generate_legal_moves(board, stm, fallback_count);
    if (fallback_count > 0) {
      final_result.best_move = decode_move(fallback_moves[0]);
      final_result.principal_variation.resize(1);
      final_result.principal_variation[0] = final_result.best_move;
      final_result.pv_length = 1;
    }
  }
  final_result.score = normalize_mate_score(final_result.score, start_ply);
  final_result.nodes = nodes;
  const bool abort_flagged = abort_requested.load(std::memory_order_relaxed);
  const bool has_move = final_result.best_move.moving_pc != OccupancyType::empty;
  final_result.aborted = abort_flagged && !has_move;

  if (search_stats_enabled()) {
    const auto& stats = search_stats();
    const auto q_nodes = stats.quiescence_nodes.load(std::memory_order_relaxed);
    const auto q_stand_pat =
        stats.quiescence_stand_pat_cutoffs.load(std::memory_order_relaxed);
    const auto q_zero_gain =
        stats.quiescence_zero_gain_skips.load(std::memory_order_relaxed);
    const auto q_delta =
        stats.quiescence_delta_prunes.load(std::memory_order_relaxed);
    const auto null_cutoffs =
        stats.null_move_cutoffs.load(std::memory_order_relaxed);
    const auto lmp_prunes = stats.lmp_prunes.load(std::memory_order_relaxed);
    const auto pvs_researches =
        stats.pvs_researches.load(std::memory_order_relaxed);
    const auto eval_calls = stats.eval_calls.load(std::memory_order_relaxed);
    const auto eval_time_ns = stats.eval_time_ns.load(std::memory_order_relaxed);
    const auto movegen_calls =
        stats.movegen_calls.load(std::memory_order_relaxed);
    const auto movegen_time_ns =
        stats.movegen_time_ns.load(std::memory_order_relaxed);
    const auto tt_probes = stats.tt_probes.load(std::memory_order_relaxed);
    const auto tt_hits = stats.tt_hits.load(std::memory_order_relaxed);
    const auto tt_cutoffs = stats.tt_cutoffs.load(std::memory_order_relaxed);
    const auto tt_probe_time_ns =
        stats.tt_probe_time_ns.load(std::memory_order_relaxed);
    const double eval_ms = static_cast<double>(eval_time_ns) / 1e6;
    const double movegen_ms = static_cast<double>(movegen_time_ns) / 1e6;
    const double tt_probe_ms = static_cast<double>(tt_probe_time_ns) / 1e6;
    const double avg_eval_us = eval_calls > 0
                                   ? static_cast<double>(eval_time_ns) / 1e3 /
                                         static_cast<double>(eval_calls)
                                   : 0.0;
    const double avg_movegen_us =
        movegen_calls > 0 ? static_cast<double>(movegen_time_ns) / 1e3 /
                                static_cast<double>(movegen_calls)
                          : 0.0;
    const double tt_hit_rate = tt_probes > 0
                                   ? (100.0 * static_cast<double>(tt_hits) /
                                      static_cast<double>(tt_probes))
                                   : 0.0;
    std::cerr << "[search-stats] q_nodes=" << q_nodes
              << " q_stand_pat_cutoffs=" << q_stand_pat
              << " q_zero_gain_skips=" << q_zero_gain
              << " q_delta_prunes=" << q_delta
              << " null_cutoffs=" << null_cutoffs << " lmp_prunes=" << lmp_prunes
              << " pvs_researches=" << pvs_researches
              << " eval_calls=" << eval_calls
              << " eval_ms=" << static_cast<std::uint64_t>(eval_ms)
              << " avg_eval_us=" << avg_eval_us
              << " movegen_calls=" << movegen_calls
              << " movegen_ms=" << static_cast<std::uint64_t>(movegen_ms)
              << " avg_movegen_us=" << avg_movegen_us
              << " tt_probes=" << tt_probes << " tt_hits=" << tt_hits
              << " tt_hit_rate=" << tt_hit_rate << " tt_cutoffs=" << tt_cutoffs
              << " tt_probe_ms=" << static_cast<std::uint64_t>(tt_probe_ms)
              << '\n';
    const double total_bucket_ms = eval_ms + movegen_ms + tt_probe_ms;
    if (total_bucket_ms > 0.0) {
      const double eval_pct = 100.0 * eval_ms / total_bucket_ms;
      const double movegen_pct = 100.0 * movegen_ms / total_bucket_ms;
      const double tt_pct = 100.0 * tt_probe_ms / total_bucket_ms;
      std::cerr << "[search-prof] eval_pct=" << eval_pct
                << " movegen_pct=" << movegen_pct << " tt_probe_pct=" << tt_pct
                << " total_bucket_ms="
                << static_cast<std::uint64_t>(total_bucket_ms) << '\n';
    }
    std::cerr << "[search-stop] last_depth=" << last_completed_depth
              << " target_depth=" << params.depth
              << " aborted=" << (stopped_by_abort ? 1 : 0)
              << " soft_limit=" << (stopped_by_soft_limit ? 1 : 0)
              << " time_mgr="
              << ((scratch.time_manager && scratch.time_manager->enabled()) ? 1
                                                                            : 0)
              << '\n';
  }

  return final_result;
}

// Parallel search entry point
// Sets up the lazy SMP context and spawns helper threads
// before calling the main search function
SearchResult search_position_parallel(Board& board, SideToMove stm,
                                      const SearchParameters& params,
                                      const EvaluatorFn& evaluator,
                                      MoveHistory* history,
                                      TranspositionTable* tt,
                                      int repetition_start, int helper_threads) {

  // for now, use single-threaded search if no helpers or node limits are set
  // we might revisit the use_nodes later
  if (params.limits.use_nodes || helper_threads <= 0) {
    return search_position(board, stm, params, evaluator, history, tt,
                           repetition_start);
  }

  LazySmpContext context;
  context.helper_count = 0;
  context.tt = tt;
  context.evaluator = &evaluator;
  context.time_manager = params.time_manager;
  context.abort_flag =
      params.abort_flag ? params.abort_flag : &context.internal_abort;

  std::vector<LazySmpThread> threads(static_cast<std::size_t>(helper_threads) +
                                     1);

#if CHESS_USE_POSIX_THREADS
  std::vector<pthread_t> worker_handles;
  worker_handles.reserve(static_cast<std::size_t>(helper_threads));
#else
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(helper_threads));
#endif

  int launched_helpers = 0;
  for (int idx = 1; idx <= helper_threads; ++idx) {
    threads[static_cast<std::size_t>(idx)].context = &context;
    threads[static_cast<std::size_t>(idx)].id = idx;
    threads[static_cast<std::size_t>(idx)].nnue_adapter =
        std::make_unique<NnueAdapter>(board);

#if CHESS_USE_POSIX_THREADS
    pthread_attr_t attr;
    pthread_attr_init(&attr);
#ifdef PTHREAD_STACK_MIN
    const std::size_t min_stack = static_cast<std::size_t>(PTHREAD_STACK_MIN);
#else
    const std::size_t min_stack = 0;
#endif
    const std::size_t stack_size =
        std::max<std::size_t>(kLazyWorkerStackSize, min_stack);
    pthread_attr_setstacksize(&attr, stack_size);

    auto payload = std::make_unique<LazyWorkerPayload>();
    payload->thread = &threads[static_cast<std::size_t>(idx)];

    pthread_t handle{};
    const int err = pthread_create(&handle, &attr, &lazy_worker_entry,
                                   static_cast<void*>(payload.get()));
    pthread_attr_destroy(&attr);

    if (err != 0) {
      std::cerr << "[skaks] warning: failed to spawn helper thread " << idx
                << " (errno=" << err << ")\n";
      continue;
    }

    payload.release();
    worker_handles.push_back(handle);
    ++launched_helpers;
#else
    workers.emplace_back([&, idx]() {
      lazy_worker_loop(threads[static_cast<std::size_t>(idx)]);
    });
    ++launched_helpers;
#endif
  }

  context.helper_count = launched_helpers;

  SearchParameters parallel_params = params;
  parallel_params.abort_flag = context.abort_flag;

  LazySmpThread* previous_thread = tls_lazy_thread;
  threads[0].context = &context;
  threads[0].id = 0;
  threads[0].nnue_adapter = std::make_unique<NnueAdapter>(board);
  tls_lazy_thread = &threads[0];

  SearchResult result = search_position(board, stm, parallel_params, evaluator,
                                        history, tt, repetition_start);

  tls_lazy_thread = previous_thread;

  context.stop();

#if CHESS_USE_POSIX_THREADS
  for (pthread_t handle : worker_handles) {
    if (handle) {
      pthread_join(handle, nullptr);
    }
  }
#else
  for (auto& worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
#endif

  return result;
}

// used when no Engine is provided, uses default eval
namespace {
int default_evaluator(const Board& board, NnueAdapter* adapter) {
  (void)adapter;
  return evaluate_board(board);
}
} // namespace

SearchResult search_position(Board& board, SideToMove stm,
                             const SearchParameters& params) {
  const EvaluatorFn evaluator = default_evaluator;
  return search_position(board, stm, params, evaluator, nullptr, nullptr);
}

} // namespace chess
