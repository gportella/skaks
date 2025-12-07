#include "chess/engine.hpp"

#include <gtest/gtest.h>

TEST(EngineTests, DefaultEvaluationIsZero) {
  const chess::Engine engine;
  EXPECT_EQ(engine.sample_evaluation(), 0);
}
