#pragma once

#include "chess/types.hpp"

#include <chrono>
#include <cstdint>

namespace chess {

struct SearchLimits;

class TimeManager {
public:
  TimeManager();

  void configure(SideToMove stm, const SearchLimits& limits);
  void start();
  void set_complexity_hint(double hint);

  [[nodiscard]] bool enabled() const {
    return enabled_;
  }

  [[nodiscard]] std::uint64_t soft_limit_ms() const {
    return soft_limit_ms_;
  }

  [[nodiscard]] std::uint64_t hard_limit_ms() const {
    return hard_limit_ms_;
  }

  [[nodiscard]] std::uint64_t elapsed_ms() const;

  [[nodiscard]] bool soft_limit_reached() const;

  [[nodiscard]] bool hard_limit_reached() const;

private:
  std::chrono::steady_clock::time_point start_point_{};
  bool running_ = false;
  bool enabled_ = false;
  std::uint64_t soft_limit_ms_ = 0;
  std::uint64_t hard_limit_ms_ = 0;
  bool complexity_scale_applied_ = false;
};

} // namespace chess
