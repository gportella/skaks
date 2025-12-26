#include "chess/board.hpp"
#include "chess/nnue.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace {

using chess::make_nnue_features;
using chess::nnue_piece_index;
using chess::NnueFeatures;
using chess::SideToMove;

TEST(NnueFeatures, BasicExtractionSetsExpectedBits) {
  const std::string fen = "8/8/8/8/8/8/8/4K2k w - - 0 1";
  chess::Board b = chess::initial_board(fen);
  NnueFeatures f = make_nnue_features(b);

  // Two kings => 2 pieces * 2 king buckets = 4 one-hot bits + stm bit.
  int ones = 0;
  for (auto v : f.values) {
    ones += static_cast<int>(v);
  }
  EXPECT_EQ(ones, 5);
  EXPECT_EQ(f.values[chess::kNnueInputs - 1], 1); // white to move
}

TEST(NnueFeatures, SideToMoveBitFlips) {
  const std::string fen_w = "8/8/8/8/8/8/8/4K2k w - - 0 1";
  const std::string fen_b = "8/8/8/8/8/8/8/4K2k b - - 0 1";
  auto fw = make_nnue_features(chess::initial_board(fen_w));
  auto fb = make_nnue_features(chess::initial_board(fen_b));
  EXPECT_EQ(fw.values[chess::kNnueInputs - 1], 1);
  EXPECT_EQ(fb.values[chess::kNnueInputs - 1], 0);
}

TEST(NnueNetwork, ForwardSimpleWeights) {
  chess::NnueNetwork net{};
  const std::size_t hidden = 2;
  net.b1 = {0.0f, 0.0f};
  net.w1.assign(hidden * chess::kNnueInputs, 0.0f);
  // Activate first hidden unit on the stm bit
  net.w1[0 * chess::kNnueInputs + (chess::kNnueInputs - 1)] = 2.0f;
  // Activate second hidden unit on white king at E1 (present in the test FEN)
  const std::size_t king_e1_idx =
      5 * 64 + 4; // wK piece index 5, square E1 idx 4
  net.w1[1 * chess::kNnueInputs + king_e1_idx] = 1.0f;
  net.w2 = {3.0f, 5.0f};
  net.b2 = 1.0f;

  chess::Board b = chess::initial_board("8/8/8/8/8/8/8/4K2k w - - 0 1");
  auto feat = make_nnue_features(b);
  const float out = net.forward(feat);
  // hidden0: relu(2*1) = 2 => 3*2 = 6; hidden1: relu(1*1) =1 => 5*1=5; +b2=1 =>
  // total 12
  EXPECT_FLOAT_EQ(out, 12.0f);
}

TEST(NnueLoader, LoadsFromYamlFile) {
  const auto tmp_path = std::filesystem::temp_directory_path() /
                        std::filesystem::path("nnue_test.yaml");
  {
    std::ofstream os(tmp_path);
    os << "nnue:\n";
    os << "  hidden: 1\n";
    const std::size_t king_e1_idx =
        5 * 64 + 4; // wK piece index 5, square E1 idx 4
    os << "  w1: [";
    for (std::size_t i = 0; i < chess::kNnueInputs; ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << (i == king_e1_idx ? 1.0f : 0.0f);
    }
    os << "]\n";
    os << "  b1: [0.0]\n";
    os << "  w2: [2.0]\n";
    os << "  b2: 0.5\n";
  }

  chess::NnueNetwork net{};
  std::string error;
  ASSERT_TRUE(chess::load_nnue_from_file(tmp_path.string(), net, error))
      << error;

  chess::Board b = chess::initial_board("8/8/8/8/8/8/8/4K2k w - - 0 1");
  auto feat = chess::make_nnue_features(b);
  const float out = net.forward(feat);
  EXPECT_FLOAT_EQ(out, 2.5f);

  std::filesystem::remove(tmp_path);
}

} // namespace
