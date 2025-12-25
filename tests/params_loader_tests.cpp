#include "chess/engine_params.hpp"
#include "chess/evaluation_params.hpp"
#include "chess/params_loader.hpp"
#include "chess/search_params.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace {

std::filesystem::path write_temp_yaml(const std::string& content) {
  auto path = std::filesystem::temp_directory_path() / "skaks_params_test.yaml";
  std::ofstream out(path);
  out << content;
  return path;
}

} // namespace

TEST(ParamsLoader, EmptyFileKeepsDefaults) {
  chess::EngineParams params = chess::default_engine_params();
  const auto original_eval = params.evaluation;
  const auto original_search = params.search;

  const auto path = write_temp_yaml("");
  std::string error;
  ASSERT_TRUE(chess::load_engine_params_from_file(path.string(), params, error))
      << error;

  EXPECT_EQ(params.evaluation.check_penalty, original_eval.check_penalty);
  EXPECT_EQ(params.search.aspiration_window_initial,
            original_search.aspiration_window_initial);
}

TEST(ParamsLoader, AppliesOverrides) {
  chess::EngineParams params = chess::default_engine_params();
  const auto path = write_temp_yaml(R"(
    evaluation:
      check_penalty: 222
      king_attack_weights: [1,2,3,4,5,6,7,8,9,10,11,12]
    search:
      aspiration_window_initial: 256
      quiescence_max_noisy_moves: 8
  )");

  std::string error;
  ASSERT_TRUE(chess::load_engine_params_from_file(path.string(), params, error))
      << error;

  EXPECT_EQ(params.evaluation.check_penalty, 222);
  EXPECT_EQ(params.evaluation.king_attack_weights[0], 1);
  EXPECT_EQ(params.evaluation.king_attack_weights[11], 12);
  EXPECT_EQ(params.search.aspiration_window_initial, 256);
  EXPECT_EQ(params.search.quiescence_max_noisy_moves, 8);
}

TEST(ParamsLoader, InvalidTypeFails) {
  chess::EngineParams params = chess::default_engine_params();
  const auto path = write_temp_yaml(R"(
    evaluation:
      check_penalty: wrong
  )");

  std::string error;
  EXPECT_FALSE(
      chess::load_engine_params_from_file(path.string(), params, error));
  EXPECT_FALSE(error.empty());

  // params must remain unchanged on failure
  EXPECT_EQ(params.evaluation.check_penalty,
            chess::default_evaluation_params().check_penalty);
}
