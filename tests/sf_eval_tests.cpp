#include "sf_eval.hpp"

#include <gtest/gtest.h>

using namespace sf_eval;

TEST(SF_Eval, ParseFenStarting) {
  Board bd;
  bool ok =
      parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", bd);
  ASSERT_TRUE(ok);
  // starting position material should be 0 (equal)
  auto r = evaluate(bd);
  EXPECT_EQ(r.score, 0);
}

TEST(SF_Eval, WhiteAdvantage) {
  Board bd;
  // white up a queen
  bool ok =
      parse_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPQPPP/RNB1KBNR w KQkq - 0 1", bd);
  ASSERT_TRUE(ok);
  auto r = evaluate(bd);
  EXPECT_GT(r.score, 700); // queen advantage should stay comfortably high
}

TEST(SF_Eval, BlackAdvantage) {
  Board bd;
  bool ok =
      parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1", bd);
  ASSERT_TRUE(ok);
  // black has an extra knight in this fictional test, ensure negative
  auto r = evaluate(bd);
  EXPECT_LT(r.score, 0);
}
