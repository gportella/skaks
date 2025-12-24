#include "chess/time_manager.hpp"

#include "chess/search.hpp"

#include <algorithm>
#include <limits>

namespace chess {
namespace {
constexpr std::uint64_t kMinTimeMs = 1;
constexpr std::uint64_t kSafetyMarginMs = 5;
constexpr std::uint32_t kDefaultMovesToGo = 40;

std::uint64_t clamp_positive(std::uint64_t value) {
  return std::max<std::uint64_t>(value, kMinTimeMs);
}
} // namespace

TimeManager::TimeManager() = default;

void TimeManager::configure(SideToMove stm, const SearchLimits& limits) {
  enabled_ = limits.use_time;
  running_ = false;
  soft_limit_ms_ = 0;
  hard_limit_ms_ = 0;

  if (!enabled_) {
    return;
  }

  if (limits.per_move) {
    const std::uint64_t cap = clamp_positive(limits.move_time_ms);
    const std::uint64_t margin =
        std::max<std::uint64_t>(cap / 10, kSafetyMarginMs);
    hard_limit_ms_ = cap;
    soft_limit_ms_ = (cap > margin) ? (cap - margin) : cap;
    return;
  }

  const std::uint64_t total =
      (stm == SideToMove::White) ? limits.white_time_ms : limits.black_time_ms;
  const std::uint64_t increment = (stm == SideToMove::White)
                                      ? limits.white_increment_ms
                                      : limits.black_increment_ms;

  if (total == 0 && increment == 0) {
    enabled_ = false;
    return;
  }

  const std::uint32_t moves_to_go =
      limits.moves_to_go > 0 ? limits.moves_to_go : kDefaultMovesToGo;

  std::uint64_t base_budget = 0;
  if (total > 0) {
    base_budget = total / std::max<std::uint32_t>(1, moves_to_go);
  }
  base_budget += increment / 2;

  if (base_budget == 0) {
    base_budget = std::max<std::uint64_t>(increment, kMinTimeMs);
  }

  const std::uint64_t spend_cap = [total]() -> std::uint64_t {
    if (total == 0) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    if (total <= 150) {
      return total;
    }
    const std::uint64_t conservative = total / 2;
    const std::uint64_t aggressive = total - 50;
    return std::max(conservative, aggressive);
  }();

  soft_limit_ms_ = std::min<std::uint64_t>(base_budget, spend_cap);
  soft_limit_ms_ = clamp_positive(soft_limit_ms_);

  const std::uint64_t extension =
      std::max<std::uint64_t>(base_budget / 2, increment);
  hard_limit_ms_ = soft_limit_ms_ + extension;
  if (total > 0) {
    const std::uint64_t absolute_cap = (total > 50) ? (total - 50) : total;
    hard_limit_ms_ = std::min<std::uint64_t>(hard_limit_ms_, absolute_cap);
  }
  hard_limit_ms_ = std::max(hard_limit_ms_, soft_limit_ms_);
  hard_limit_ms_ = clamp_positive(hard_limit_ms_);
}

void TimeManager::start() {
  if (!enabled_) {
    running_ = false;
    return;
  }
  start_point_ = std::chrono::steady_clock::now();
  running_ = true;
}

std::uint64_t TimeManager::elapsed_ms() const {
  if (!running_) {
    return 0;
  }
  const auto now = std::chrono::steady_clock::now();
  const auto delta =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start_point_);
  return static_cast<std::uint64_t>(delta.count());
}

bool TimeManager::soft_limit_reached() const {
  if (!enabled_ || !running_) {
    return false;
  }
  return elapsed_ms() >= soft_limit_ms_;
}

bool TimeManager::hard_limit_reached() const {
  if (!enabled_ || !running_) {
    return false;
  }
  return elapsed_ms() >= hard_limit_ms_;
}

} // namespace chess
