#include "chess/search.hpp"
#include "chess/time_manager.hpp"
#include "chess/types.hpp"

#include <gtest/gtest.h>

namespace chess::detail {
std::uint32_t estimate_moves_to_go(std::uint64_t total_ms,
                                   std::uint64_t increment_ms);
}

namespace {

TEST(TimeManagerTests, MovesToGoRegimesAdjustWithTimeControl) {
  using chess::detail::estimate_moves_to_go;

  EXPECT_EQ(estimate_moves_to_go(3'600'000, 0), 45u);   // Classical
  EXPECT_EQ(estimate_moves_to_go(600'000, 0), 30u);     // Rapid
  EXPECT_EQ(estimate_moves_to_go(180'000, 5'000), 20u); // Blitz (3+5)
  EXPECT_EQ(estimate_moves_to_go(60'000, 0), 12u);      // Bullet
}

TEST(TimeManagerTests, PerMoveBudget) {
  chess::TimeManager manager;
  chess::SearchLimits limits{};
  limits.use_time = true;
  limits.per_move = true;
  limits.move_time_ms = 5000;

  manager.configure(chess::SideToMove::White, limits);
  manager.start();

  EXPECT_TRUE(manager.enabled());
  EXPECT_EQ(manager.hard_limit_ms(), 5000u);
  EXPECT_EQ(manager.soft_limit_ms(), 4500u);
}

TEST(TimeManagerTests, ClockBudgetIncludesIncrement) {
  chess::TimeManager manager;
  chess::SearchLimits limits{};
  limits.use_time = true;
  limits.per_move = false;
  limits.white_time_ms = 60000;
  limits.black_time_ms = 60000;
  limits.white_increment_ms = 1000;
  limits.black_increment_ms = 1000;
  limits.moves_to_go = 30;

  manager.configure(chess::SideToMove::White, limits);
  manager.start();

  EXPECT_TRUE(manager.enabled());
  EXPECT_GE(manager.soft_limit_ms(), 2000u);
  EXPECT_LT(manager.hard_limit_ms(), 60000u);
  EXPECT_GE(manager.hard_limit_ms(), manager.soft_limit_ms());
}

TEST(TimeManagerTests, DisabledWhenNoTimeProvided) {
  chess::TimeManager manager;
  chess::SearchLimits limits{};
  limits.use_time = true;
  limits.per_move = false;
  limits.white_time_ms = 0;
  limits.black_time_ms = 0;
  limits.white_increment_ms = 0;
  limits.black_increment_ms = 0;

  manager.configure(chess::SideToMove::Black, limits);
  manager.start();

  EXPECT_FALSE(manager.enabled());
  EXPECT_EQ(manager.soft_limit_ms(), 0u);
  EXPECT_EQ(manager.hard_limit_ms(), 0u);
}

} // namespace
