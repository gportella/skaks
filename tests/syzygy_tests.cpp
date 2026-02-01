#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/syzygy.hpp"

#include <gtest/gtest.h>

namespace chess::tests {

/**
 * @brief Probing should be unavailable if Syzygy is not initialized.
 */
TEST(SyzygyTests, ProbeWdlUnavailableWhenDisabled) {
  syzygy::free();
  syzygy::init("");

  Board board = initial_board(kStartFEN);
  const auto result = syzygy::probe_wdl(board);
  EXPECT_FALSE(result.available);
}

/**
 * @brief Empty path initialization should disable Syzygy probing.
 */
TEST(SyzygyTests, InitEmptyPathDisables) {
  syzygy::free();
  const bool ok = syzygy::init("");
  EXPECT_FALSE(ok);
  EXPECT_FALSE(syzygy::available());
  EXPECT_TRUE(syzygy::path().empty());
  syzygy::free();
}

} // namespace chess::tests
