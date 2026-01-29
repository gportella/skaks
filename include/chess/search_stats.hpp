#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace chess {

struct SearchStats {
  std::atomic<std::uint64_t> quiescence_nodes{0};
  std::atomic<std::uint64_t> quiescence_stand_pat_cutoffs{0};
  std::atomic<std::uint64_t> quiescence_zero_gain_skips{0};
  std::atomic<std::uint64_t> quiescence_delta_prunes{0};
  std::atomic<std::uint64_t> null_move_cutoffs{0};
  std::atomic<std::uint64_t> lmp_prunes{0};
  std::atomic<std::uint64_t> pvs_researches{0};
  std::atomic<std::uint64_t> see_prunes{0};
  std::atomic<std::uint64_t> futility_prunes{0};
  std::atomic<std::uint64_t> eval_calls{0};
  std::atomic<std::uint64_t> eval_time_ns{0};
  std::atomic<std::uint64_t> movegen_calls{0};
  std::atomic<std::uint64_t> movegen_time_ns{0};
  std::atomic<std::uint64_t> tt_probes{0};
  std::atomic<std::uint64_t> tt_hits{0};
  std::atomic<std::uint64_t> tt_cutoffs{0};
  std::atomic<std::uint64_t> tt_probe_time_ns{0};
};

inline bool search_stats_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("SKAKS_SEARCH_STATS");
    return value && *value && *value != '0';
  }();
  return enabled;
}

inline SearchStats& search_stats() {
  static SearchStats stats{};
  return stats;
}

inline void reset_search_stats() {
  auto& stats = search_stats();
  stats.quiescence_nodes.store(0, std::memory_order_relaxed);
  stats.quiescence_stand_pat_cutoffs.store(0, std::memory_order_relaxed);
  stats.quiescence_zero_gain_skips.store(0, std::memory_order_relaxed);
  stats.quiescence_delta_prunes.store(0, std::memory_order_relaxed);
  stats.null_move_cutoffs.store(0, std::memory_order_relaxed);
  stats.lmp_prunes.store(0, std::memory_order_relaxed);
  stats.pvs_researches.store(0, std::memory_order_relaxed);
  stats.see_prunes.store(0, std::memory_order_relaxed);
  stats.futility_prunes.store(0, std::memory_order_relaxed);
  stats.eval_calls.store(0, std::memory_order_relaxed);
  stats.eval_time_ns.store(0, std::memory_order_relaxed);
  stats.movegen_calls.store(0, std::memory_order_relaxed);
  stats.movegen_time_ns.store(0, std::memory_order_relaxed);
  stats.tt_probes.store(0, std::memory_order_relaxed);
  stats.tt_hits.store(0, std::memory_order_relaxed);
  stats.tt_cutoffs.store(0, std::memory_order_relaxed);
  stats.tt_probe_time_ns.store(0, std::memory_order_relaxed);
}

} // namespace chess
