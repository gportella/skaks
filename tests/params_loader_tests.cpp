#include "chess/engine_params.hpp"
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
  const auto original_search = params.search;

  const auto path = write_temp_yaml("");
  std::string error;
  ASSERT_TRUE(chess::load_engine_params_from_file(path.string(), params, error))
      << error;

  EXPECT_EQ(params.search.aspiration_window_initial,
            original_search.aspiration_window_initial);
}

TEST(ParamsLoader, AppliesOverrides) {
  chess::EngineParams params = chess::default_engine_params();
  const auto path = write_temp_yaml(R"(
    search:
      aspiration_window_initial: 256
      quiescence_max_noisy_moves: 8
  )");

  std::string error;
  ASSERT_TRUE(chess::load_engine_params_from_file(path.string(), params, error))
      << error;

  EXPECT_EQ(params.search.aspiration_window_initial, 256);
  EXPECT_EQ(params.search.quiescence_max_noisy_moves, 8);
}

TEST(ParamsLoader, InvalidTypeFails) {
  chess::EngineParams params = chess::default_engine_params();
  const auto path = write_temp_yaml(R"(
    search:
      aspiration_window_initial: wrong
  )");

  std::string error;
  EXPECT_FALSE(
      chess::load_engine_params_from_file(path.string(), params, error));
  EXPECT_FALSE(error.empty());

  // params must remain unchanged on failure
  EXPECT_EQ(params.search.aspiration_window_initial,
            chess::default_search_params().aspiration_window_initial);
}
