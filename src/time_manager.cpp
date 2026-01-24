#include "chess/time_manager.hpp"

#include "chess/search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace chess {
namespace {
constexpr std::uint64_t kMinTimeMs = 1;
constexpr std::uint64_t kSafetyMarginMs = 5;
constexpr std::uint64_t kStopOverheadMs = 50;
constexpr std::uint64_t kReserveBufferMs = 40;
constexpr std::uint32_t kIncrementProjectionMoves = 20;

struct TimeRegimeProfile {
  std::uint64_t min_effective_ms;
  std::uint32_t default_moves_to_go;
  double increment_blend;
  double aggressive_factor;
};

constexpr std::array<TimeRegimeProfile, 4> kTimeRegimeProfiles{{
    {1'800'000, 45, 0.60, 1.20}, // Classical: cautious increment usage
    {600'000, 30, 0.70, 1.30},   // Rapid: balanced behaviour
    {150'000, 20, 0.55, 1.15},   // Blitz: closer to increment spending
    {0, 12, 0.45, 1.05}          // Bullet: rely almost entirely on increment
}};

void trace_time_line(const std::string& line) {
  const char* trace_file = std::getenv("SKAKS_TIME_TRACE_FILE");
  if (trace_file && *trace_file) {
    std::ofstream out(trace_file, std::ios::app);
    out << line << '\n';
    return;
  }
  std::cerr << line << '\n';
}

const TimeRegimeProfile& select_time_regime(std::uint64_t total_ms,
                                            std::uint64_t increment_ms) {
  const std::uint64_t effective_ms =
      total_ms + increment_ms * kIncrementProjectionMoves;
  for (const auto& profile : kTimeRegimeProfiles) {
    if (effective_ms >= profile.min_effective_ms) {
      return profile;
    }
  }
  return kTimeRegimeProfiles.back();
}

std::uint64_t clamp_positive(std::uint64_t value) {
  return std::max<std::uint64_t>(value, kMinTimeMs);
}
} // namespace

namespace detail {
std::uint32_t estimate_moves_to_go(std::uint64_t total_ms,
                                   std::uint64_t increment_ms) {
  return select_time_regime(total_ms, increment_ms).default_moves_to_go;
}
} // namespace detail

TimeManager::TimeManager() = default;

void TimeManager::configure(SideToMove stm, const SearchLimits& limits) {
  enabled_ = limits.use_time;
  running_ = false;
  soft_limit_ms_ = 0;
  hard_limit_ms_ = 0;
  complexity_scale_applied_ = false;

  const bool trace = std::getenv("SKAKS_TIME_TRACE") != nullptr;

  if (!enabled_) {
    if (trace) {
      trace_time_line("[time-trace] disabled");
    }
    return;
  }

  if (limits.per_move) {
    const std::uint64_t cap = clamp_positive(limits.move_time_ms);
    const std::uint64_t margin =
        std::max<std::uint64_t>(cap / 10, kSafetyMarginMs);
    hard_limit_ms_ = cap;
    soft_limit_ms_ = (cap > margin) ? (cap - margin) : cap;
    if (hard_limit_ms_ > kStopOverheadMs) {
      hard_limit_ms_ -= kStopOverheadMs;
    }
    if (soft_limit_ms_ > kStopOverheadMs) {
      soft_limit_ms_ -= kStopOverheadMs;
    }
    if (trace) {
      std::ostringstream oss;
      oss << "[time-trace] per_move cap=" << cap << " soft=" << soft_limit_ms_
          << " hard=" << hard_limit_ms_;
      trace_time_line(oss.str());
    }
    return;
  }

  const std::uint64_t total =
      (stm == SideToMove::White) ? limits.white_time_ms : limits.black_time_ms;
  const std::uint64_t increment = (stm == SideToMove::White)
                                      ? limits.white_increment_ms
                                      : limits.black_increment_ms;

  if (total == 0 && increment == 0) {
    enabled_ = false;
    if (trace) {
      trace_time_line("[time-trace] disabled (no time/increment)");
    }
    return;
  }

  const auto& profile = select_time_regime(total, increment);
  const double increment_blend = profile.increment_blend;
  const double aggressive_factor = profile.aggressive_factor;

  std::uint32_t moves_to_go = limits.moves_to_go;
  if (moves_to_go == 0) {
    moves_to_go = profile.default_moves_to_go;
  }

  double base_budget = 0.0;
  if (total > 0 && moves_to_go > 0) {
    base_budget = static_cast<double>(total) /
                  static_cast<double>(std::max<std::uint32_t>(1, moves_to_go));
  }
  base_budget += static_cast<double>(increment) * increment_blend;

  if (total <= 180'000) {
    const double min_floor = static_cast<double>(increment) * 1.3;
    const double max_cap = static_cast<double>(total) /
                           static_cast<double>(std::max<std::uint32_t>(1, 10u));
    base_budget = std::clamp(base_budget, min_floor, max_cap);
  }

  if (base_budget <= 0.0) {
    base_budget =
        static_cast<double>(std::max<std::uint64_t>(increment, kMinTimeMs));
  }

  const double aggressive_budget = base_budget * aggressive_factor;
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
      static_cast<std::uint64_t>(base_ms / 3), increment);
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

  if (total > 0 && total <= 180'000) {
    const std::uint64_t blitz_soft_cap =
        std::max<std::uint64_t>(increment * 2, base_ms);
    const std::uint64_t blitz_hard_cap = blitz_soft_cap + increment;
    soft_limit_ms_ = std::min(soft_limit_ms_, blitz_soft_cap);
    hard_limit_ms_ = std::min(hard_limit_ms_, blitz_hard_cap);
  }

  if (increment == 0 && total > 0 && total < 5'000) {
    hard_limit_ms_ = std::min<std::uint64_t>(hard_limit_ms_, total);
    soft_limit_ms_ = std::min<std::uint64_t>(soft_limit_ms_, hard_limit_ms_);
  }

  hard_limit_ms_ = std::max(hard_limit_ms_, soft_limit_ms_);
  soft_limit_ms_ = clamp_positive(soft_limit_ms_);
  hard_limit_ms_ = clamp_positive(hard_limit_ms_);
  if (hard_limit_ms_ > kStopOverheadMs) {
    hard_limit_ms_ -= kStopOverheadMs;
  }
  if (soft_limit_ms_ > kStopOverheadMs) {
    soft_limit_ms_ -= kStopOverheadMs;
  }
  if (trace) {
    std::ostringstream oss;
    oss << "[time-trace] total=" << total << " inc=" << increment
        << " moves_to_go=" << moves_to_go << " soft=" << soft_limit_ms_
        << " hard=" << hard_limit_ms_;
    trace_time_line(oss.str());
  }
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
