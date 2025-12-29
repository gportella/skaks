#include "chess/board.hpp"
#include "chess/nnue.hpp"

#include <gtest/gtest.h>

TEST(NnueSf, MissingFileDoesNotActivateBackend) {
  std::string error;
  // Loading a non-existent .nnue should fail cleanly and keep SF backend idle.
  EXPECT_FALSE(chess::load_sf_nnue("nonexistent.nnue", error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(chess::sf_nnue_active());
  EXPECT_THROW(
      {
        chess::evaluate_sf_nnue(chess::initial_board(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
      },
      std::runtime_error);
}
