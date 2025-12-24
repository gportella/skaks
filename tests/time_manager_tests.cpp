#include "chess/search.hpp"
#include "chess/time_manager.hpp"
#include "chess/types.hpp"

#include <gtest/gtest.h>

namespace {

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
