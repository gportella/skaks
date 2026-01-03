#include "chess/search.hpp"

#include "chess/complexity.hpp"
#include "chess/move_ordering.hpp"
#include "chess/moves.hpp"
#include "chess/nnue.hpp"
#include "chess/nnue_incremental.hpp"
#include "chess/quiescence.hpp"
#include "chess/score.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/search_params.hpp"
#include "chess/time_manager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>
#include <condition_variable>

#if defined(__APPLE__) || defined(__linux__)
#define CHESS_USE_POSIX_THREADS 1
#include <pthread.h>
#else
#define CHESS_USE_POSIX_THREADS 0
#endif

namespace chess {
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
  TimeManager* time_manager = nullptr;
  std::atomic<bool>* abort_requested = nullptr;
  std::atomic<bool>* local_abort = nullptr;
  LazySmpThread* thread = nullptr;
};

constexpr std::uint64_t kTimeCheckMask = 0x3FFULL;

inline bool should_abort_due_to_time(SearchScratch& scratch,
                                     std::uint64_t nodes) {
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
  if ((nodes & kTimeCheckMask) != 0) {
    return false;
  }
  if (!scratch.time_manager->hard_limit_reached()) {
    return false;
  }
  if (scratch.abort_requested) {
    scratch.abort_requested->store(true, std::memory_order_relaxed);
  }
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

SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta,
                               SideToMove stm, const EvaluatorFn& evaluator,
                               SearchScratch& scratch, int ply,
                               int repetition_start, int ply_from_root,
                               bool is_pv, std::uint64_t& nodes,
                               const Move* parent_move,
                               const uint32_t* excluded_root_moves,
                               std::size_t excluded_root_count);

MoveEvaluationResult evaluate_move(Board& board, const Move& move,
                                   const MoveEvaluationContext& ctx,
                                   SearchScratch& scratch, std::uint64_t& nodes,
                                   const EvaluatorFn& evaluator) {
  MoveEvaluationResult output{};

  const bool is_capture = move.captured_pc != OccupancyType::empty;
  const bool is_promo = move.promo_pc != OccupancyType::empty;
  const bool quiet_like = !is_capture && !is_promo;

  int child_depth = ctx.depth - 1;
  const int move_number = ctx.move_number;

  Undo undo = make_move(board, move);

  SfNnueStack* nnue_stack = current_thread_nnue_stack();
  if (nnue_stack) {
    nnue_stack->push_move(board, move, undo);
  }

  const bool irreversible = move_is_irreversible(move);
  const int next_repetition_start =
      irreversible ? (ctx.ply + 1) : ctx.repetition_start;

  const bool in_check_after_move = is_check(board, flip_side(ctx.stm));
  if (child_depth == 0 && in_check_after_move &&
      ctx.ply < static_cast<int>(MAX_PLY) - 1) {
    child_depth = 1;
  }

  const int history_score = scratch.ordering.history_score(move);

  int reduction = 0;
  if (quiet_like && !in_check_after_move && child_depth >= 2 &&
      move_number > 1) {
    const int capped_depth = std::min(child_depth, 63);
    const int capped_order = std::min(move_number, 63);
    const double depth_factor = std::log1p(static_cast<double>(capped_depth));
    const double order_factor = std::log1p(static_cast<double>(capped_order));
    const double scaled = depth_factor * order_factor * 0.85;
    const int tentative = static_cast<int>(std::round(scaled));
    const int max_reduction = std::max(0, child_depth - 1);
    reduction = std::clamp(tentative, 0, max_reduction);
    if (ctx.is_pv) {
      reduction = std::max(0, reduction - 1);
    }
    if (history_score < 2000) {
      reduction = std::min(max_reduction, reduction + 1);
    }
    if (history_score < 0) {
      reduction = std::min(max_reduction, reduction + 1);
    }
    if (history_score > 12000) {
      reduction = std::max(0, reduction - 1);
    }
    if (history_score > 20000) {
      reduction = std::max(0, reduction - 1);
    }
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
      if (nnue_stack) {
        nnue_stack->pop();
      }
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
        if (nnue_stack) {
          nnue_stack->pop();
        }
        if (ctx.ply + 1 < MAX_PLY && scratch.history) {
          scratch.history
              ->repetition_counts[static_cast<std::size_t>(ctx.ply + 1)] = 0;
        }
        output.child = child;
        output.score = child.score;
        output.aborted = true;
        return output;
      }
      if (child.score >= narrow_beta) {
        child = run_search(search_depth, ctx.alpha, ctx.beta, ctx.is_pv);
        if (child.aborted) {
          undo_move(board, undo);
          if (nnue_stack) {
            nnue_stack->pop();
          }
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
        if (nnue_stack) {
          nnue_stack->pop();
        }
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
  if (nnue_stack) {
    nnue_stack->pop();
  }

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
  SfNnueStack nnue_stack;
  bool has_history = false;
  bool has_killers = false;
  bool has_nnue = false;
  Move move{};
  int move_number = 0;
  MoveEvaluationContext eval_ctx;
  bool has_parent_move = false;
  Move parent_move_storage{};
  std::atomic<bool>* abort_flag = nullptr;

  static void* operator new(std::size_t size) {
    return ::operator new(size, std::align_val_t{alignof(LazySmpTask)});
  }

  static void operator delete(void* ptr) noexcept {
    ::operator delete(ptr, std::align_val_t{alignof(LazySmpTask)});
  }

  static void operator delete(void* ptr, std::size_t size) noexcept {
    ::operator delete(ptr, size, std::align_val_t{alignof(LazySmpTask)});
  }
};

static_assert(alignof(LazySmpTask) >= alignof(NNUEdata),
              "LazySmpTask must satisfy NNUE alignment requirements");

struct LazySmpContext {
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::deque<std::unique_ptr<LazySmpTask>> queue;
  bool shutting_down = false;

  const EvaluatorFn* evaluator = nullptr;
  TranspositionTable* tt = nullptr;
  std::atomic<bool> internal_abort{false};
  std::atomic<bool>* abort_flag = nullptr;
  int helper_count = 0;
  int min_split_depth = 3;

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

struct LazySmpThread {
  LazySmpContext* context = nullptr;
  int id = 0;
};

thread_local LazySmpThread* tls_lazy_thread = nullptr;

void execute_lazy_task(LazySmpTask& task, LazySmpThread& thread) {
  LazySmpContext* ctx = thread.context;
  if (!ctx) {
    return;
  }

  SfNnueStack* nnue_stack = task.has_nnue ? &task.nnue_stack : nullptr;
  ScopedNnueThreadContext nnue_guard(nnue_stack);

  SearchScratch scratch{};
  scratch.history = task.has_history ? &task.history : nullptr;
  scratch.killers = task.has_killers ? &task.killers : nullptr;
  scratch.tt = ctx->tt;
  scratch.ordering = task.ordering;
  scratch.time_manager = nullptr;
  scratch.abort_requested = task.abort_flag ? task.abort_flag : ctx->abort_flag;
  scratch.local_abort = &task.split_point->cancel_flag;
  scratch.thread = &thread;

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
  ++nodes;

  if (should_abort_due_to_time(scratch, nodes)) {
    return make_aborted_result();
  }

  MoveHistory* history = scratch.history;
  TranspositionTable* tt = scratch.tt;

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

  int alpha_base = alpha;
  int beta_base = beta;

  TranspositionEntry cached_entry;
  bool has_cached_move = false;
  Move cached_move{};
  if (tt && tt->probe(board.position_key, cached_entry)) {
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
          return tt_cutoff;
        }
      }
      if (cached_entry.best_move.moving_pc != OccupancyType::empty) {
        has_cached_move = true;
        cached_move = cached_entry.best_move;
      }
    }
  }

  if (allow_null_move(board, depth)) {
    UndoNull undo_null = make_null_move(board);
    SfNnueStack* nnue_stack = current_thread_nnue_stack();
    if (nnue_stack) {
      nnue_stack->push_null();
    }
    const int null_move_reduction = search_params().null_move_reduction;
    SearchResult null_result = alphabeta_minimax(
        board, depth - 1 - null_move_reduction, beta - 1, beta, flip_side(stm),
        evaluator, scratch, ply + 1, repetition_start, ply_from_root + 1, false,
        nodes, nullptr, nullptr, 0);
    if (null_result.aborted) {
      undo_null_move(board, undo_null);
      if (nnue_stack) {
        nnue_stack->pop();
      }
      return null_result;
    }
    const int score_after_null = normalize_mate_score(null_result.score, ply);
    undo_null_move(board, undo_null);
    if (nnue_stack) {
      nnue_stack->pop();
    }
    if (score_after_null >= beta) {
      SearchResult cutoff{};
      cutoff.score = score_after_null;
      cutoff.outcome = SearchResult::Outcome::InProgress;
      cutoff.pv_length = 0;
      return cutoff;
    }
  }
  alpha_base = alpha;
  beta_base = beta;

  bool do_quiescence = true;
  if (depth == 0) {

    int qs_raw = 0;
    if (do_quiescence) {
      qs_raw = quiescence(board, alpha, beta, stm, evaluator, nodes, tt, ply);
    } else {
      int eval = evaluator(static_cast<const Board&>(board));
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

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);

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

  uint32_t tt_code = 0;
  if (has_cached_move) {
    if (is_excluded_move(cached_move)) {
      has_cached_move = false;
    } else {
      tt_code = encode_move(cached_move.from, cached_move.to,
                            cached_move.moving_pc, cached_move.captured_pc,
                            cached_move.promo_pc, cached_move.flags);
    }
  }
  const auto* history_matrix = scratch.ordering.history_matrix();
  const uint32_t counter_code =
      parent_move ? scratch.ordering.counter_move(*parent_move) : 0;
  sort_moves(board, moves, move_count, tt_code, scratch.killers, history_matrix,
             ply, counter_code);

  SearchResult best{};
  best.score = (stm == SideToMove::White) ? -INF : INF;
  best.outcome = SearchResult::Outcome::InProgress;
  best.pv_length = 0;

  const bool parent_in_check = is_check(board, stm);
  int moves_tried = 0;

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
  bool cutoff_triggered = false;

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
    } else {
      const bool fail_low = (stm == SideToMove::White)
                                ? (score_value <= alpha_base)
                                : (score_value >= beta_base);
      if (fail_low) {
        update_history(scratch, current_move, depth, false);
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
      update_history(scratch, current_move, depth, true);
      record_counter(scratch, parent_move, current_move);
      if (split_raw) {
        split_raw->cancel_flag.store(true, std::memory_order_relaxed);
      }
    }

    return cutoff_now;
  };

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);
    const bool is_capture = move.captured_pc != OccupancyType::empty;
    const bool is_promo = move.promo_pc != OccupancyType::empty;
    const bool quiet_like = !is_capture && !is_promo;

    const int move_number = moves_tried + 1;
    int child_depth_lmp = depth - 1;
    if (!is_pv && !parent_in_check && quiet_like && child_depth_lmp > 0 &&
        child_depth_lmp <= 3) {
      static constexpr int lmp_limits[4] = {0, 3, 6, 10};
      if (move_number > lmp_limits[child_depth_lmp]) {
        continue;
      }
    }

    moves_tried = move_number;

    const bool can_split =
        smp_supported && depth >= smp_ctx->min_split_depth && move_number > 1;

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
      if (SfNnueStack* nnue_stack = current_thread_nnue_stack()) {
        task->nnue_stack = *nnue_stack;
        task->has_nnue = true;
      } else {
        task->has_nnue = false;
      }
      split_point->add_task();
      smp_ctx->enqueue(std::move(task));
      continue;
    }

    move_ctx.move_number = move_number;
    move_ctx.alpha = alpha;
    move_ctx.beta = beta;
    move_ctx.allow_pvs = true;

    MoveEvaluationResult eval =
        evaluate_move(board, move, move_ctx, scratch, nodes, evaluator);
    if (eval.aborted) {
      if (split_raw) {
        split_raw->cancel_flag.store(true, std::memory_order_relaxed);
      }
      return eval.child;
    }

    const bool cutoff_now = apply_result(move, eval.child, eval.score);

    if (split_point) {
      LazySmpResult ready{};
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

SearchResult search_position(Board& board, SideToMove stm,
                             const SearchParameters& params,
                             const EvaluatorFn& evaluator, MoveHistory* history,
                             TranspositionTable* tt, int repetition_start) {
  int base_ply = 0;
  int start_ply = 0;

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
  scratch.ordering.reset();
  scratch.history = history;
  scratch.killers = &killers;
  scratch.tt = tt;
  scratch.time_manager = params.time_manager;
  if (params.abort_flag) {
    scratch.abort_requested = params.abort_flag;
  } else {
    scratch.abort_requested = &abort_requested;
  }
  scratch.local_abort = nullptr;
  scratch.thread = tls_lazy_thread;

  std::unique_ptr<SfNnueStack> nnue_stack;
  SfNnueStack* nnue_stack_ptr = nullptr;
  if (sf_nnue_active()) {
    nnue_stack = std::make_unique<SfNnueStack>();
    nnue_stack->reset();
    nnue_stack_ptr = nnue_stack.get();
  }
  ScopedNnueThreadContext nnue_guard(nnue_stack_ptr);

  double root_complexity_hint = 0.0;
  if (params.time_manager) {
    const ComplexityMetrics metrics = compute_complexity(board, stm);
    root_complexity_hint = normalize_complexity(metrics);
    params.time_manager->set_complexity_hint(root_complexity_hint);
  }

  int max_root_moves = 0;
  uint16_t move_count = 0;

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
  SearchResult best_result{};
  SearchResult result{};

  int best_score = 0;
  for (int current_depth = 1; current_depth <= params.depth; ++current_depth) {
    excluded_count = static_excluded_count;
    int pv_generated = 0;
    while (true) {
      int alpha = -INF;
      int beta = INF;
      if (current_depth == 1 ||
          best_result.best_move.moving_pc == OccupancyType::empty) {
        alpha = -INF;
        beta = INF;
      } else {
        if (is_mate_score(best_score)) {
          alpha = -INF;
          beta = INF;
          aspiration_window = sparams.aspiration_window_initial;
        } else {
          const int window_low = best_score - aspiration_window;
          const int window_high = best_score + aspiration_window;
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

      if (result.best_move.moving_pc == OccupancyType::empty ||
          !max_root_moves) {
        if (best_result.best_move.moving_pc == OccupancyType::empty) {
          best_result = result;
          best_score = result.score;
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
          best_result.best_move.moving_pc == OccupancyType::empty ? true
          : (stm == SideToMove::White) ? (result.score > best_score)
                                       : (result.score < best_score);
      if (improving) {
        best_result = result;
        best_score = result.score;
      }

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
      break;
    }

    if (scratch.time_manager && scratch.time_manager->soft_limit_reached()) {
      break;
    }
  }
  if (history) {
    history->ply_count = base_ply;
  }

  SearchResult final_result =
      best_result.best_move.moving_pc != OccupancyType::empty ? best_result
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

  return final_result;
}

SearchResult search_position_parallel(Board& board, SideToMove stm,
                                      const SearchParameters& params,
                                      const EvaluatorFn& evaluator,
                                      MoveHistory* history,
                                      TranspositionTable* tt,
                                      int repetition_start, int helper_threads) {
  if (helper_threads <= 0) {
    return search_position(board, stm, params, evaluator, history, tt,
                           repetition_start);
  }

  LazySmpContext context;
  context.helper_count = 0;
  context.tt = tt;
  context.evaluator = &evaluator;
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

namespace {
int default_evaluator(const Board& board) {
  return evaluate_board(board);
}
} // namespace

SearchResult search_position(Board& board, SideToMove stm,
                             const SearchParameters& params) {
  const EvaluatorFn evaluator = default_evaluator;
  return search_position(board, stm, params, evaluator, nullptr, nullptr);
}

} // namespace chess
