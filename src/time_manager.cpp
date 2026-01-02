#include "chess/time_manager.hpp"

#include "chess/search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace chess {
namespace {
constexpr std::uint64_t kMinTimeMs = 1;
constexpr std::uint64_t kSafetyMarginMs = 5;
constexpr std::uint64_t kReserveBufferMs = 40;
constexpr double kIncrementBlend = 0.75;
constexpr double kAggressiveFactor = 1.35;

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
  complexity_scale_applied_ = false;

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

  std::uint32_t moves_to_go = limits.moves_to_go;
  if (moves_to_go == 0) {
    if (total >= 300'000) {
      moves_to_go = 45;
    } else if (total >= 120'000) {
      moves_to_go = 35;
    } else if (total >= 60'000) {
      moves_to_go = 28;
    } else if (total >= 20'000) {
      moves_to_go = 20;
    } else {
      moves_to_go = 12;
    }
  }

  double base_budget = 0.0;
  if (total > 0 && moves_to_go > 0) {
    base_budget = static_cast<double>(total) /
                  static_cast<double>(std::max<std::uint32_t>(1, moves_to_go));
  }
  base_budget += static_cast<double>(increment) * kIncrementBlend;

  if (base_budget <= 0.0) {
    base_budget =
        static_cast<double>(std::max<std::uint64_t>(increment, kMinTimeMs));
  }

  const double aggressive_budget = base_budget * kAggressiveFactor;
  const std::uint64_t spend_cap = [total]() -> std::uint64_t {
    if (total == 0) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    if (total <= 150) {
      return total;
    }
    const std::uint64_t conservative = total / 2;
    const std::uint64_t aggressive =
        (total > kReserveBufferMs) ? (total - kReserveBufferMs) : total;
    return std::max(conservative, aggressive);
  }();

  const std::uint64_t base_ms = clamp_positive(static_cast<std::uint64_t>(
      std::round(std::min(aggressive_budget, static_cast<double>(spend_cap)))));

  soft_limit_ms_ = base_ms;

  std::uint64_t extension = std::max<std::uint64_t>(
      static_cast<std::uint64_t>(base_ms / 2), increment);
  if (total > 0) {
    const std::uint64_t reserve =
        (total > kReserveBufferMs) ? kReserveBufferMs : 0;
    const std::uint64_t absolute_cap =
        (total > reserve) ? (total - reserve) : total;
    hard_limit_ms_ = std::min(base_ms + extension, absolute_cap);
    if (base_ms + extension > absolute_cap && increment > 0) {
      hard_limit_ms_ =
          std::min<std::uint64_t>(base_ms + (increment * 2), absolute_cap);
    }
  } else {
    hard_limit_ms_ = base_ms + extension;
  }

  if (increment == 0 && total > 0 && total < 5'000) {
    hard_limit_ms_ = std::min<std::uint64_t>(hard_limit_ms_, total);
    soft_limit_ms_ = std::min<std::uint64_t>(soft_limit_ms_, hard_limit_ms_);
  }

  hard_limit_ms_ = std::max(hard_limit_ms_, soft_limit_ms_);
  soft_limit_ms_ = clamp_positive(soft_limit_ms_);
  hard_limit_ms_ = clamp_positive(hard_limit_ms_);
}

void TimeManager::set_complexity_hint(double hint) {
  if (!enabled_ || complexity_scale_applied_) {
    return;
  }
  const double clamped = std::clamp(hint, 0.0, 1.5);
  const double factor = std::clamp(1.0 + clamped * 0.2, 0.85, 1.3);

  const auto scale_value = [&](std::uint64_t value) -> std::uint64_t {
    const double scaled = static_cast<double>(value) * factor;
    return clamp_positive(static_cast<std::uint64_t>(std::round(scaled)));
  };

  const std::uint64_t new_soft = scale_value(soft_limit_ms_);
  const std::uint64_t new_hard = scale_value(hard_limit_ms_);

  soft_limit_ms_ = std::min(new_soft, new_hard);
  hard_limit_ms_ = std::max(new_soft, new_hard);
  complexity_scale_applied_ = true;
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
