#include "chess/search_detail.hpp"

#include <gtest/gtest.h>

namespace {

using chess::SideToMove;

TEST(SearchPvsRetry, WhiteTriggersOnFailHigh) {
  const bool needs_retry =
      chess::search_detail::should_retry_pvs(SideToMove::White, 25, 10, 20);
  EXPECT_TRUE(needs_retry);
}

TEST(SearchPvsRetry, WhiteDoesNotTriggerOnFailLow) {
  const bool needs_retry =
      chess::search_detail::should_retry_pvs(SideToMove::White, 5, 10, 20);
  EXPECT_FALSE(needs_retry);
}

TEST(SearchPvsRetry, BlackTriggersOnFailLow) {
  const bool needs_retry =
      chess::search_detail::should_retry_pvs(SideToMove::Black, 5, 10, 20);
  EXPECT_TRUE(needs_retry);
}

TEST(SearchPvsRetry, BlackDoesNotTriggerOnFailHigh) {
  const bool needs_retry =
      chess::search_detail::should_retry_pvs(SideToMove::Black, 25, 10, 20);
  EXPECT_FALSE(needs_retry);
}

} // namespace
