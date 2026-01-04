#include "chess/cli.hpp"

#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

chess::CliParseResult parse(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  const int argc = static_cast<int>(argv.size());
  return chess::parse_cli(argc, argv.data());
}

TEST(CliParseTests, AcceptsMoveTimeOption) {
  std::vector<std::string> args = {"skaks", "--move-time", "1500"};
  const auto result = parse(std::move(args));
  ASSERT_FALSE(result.parse_error) << result.message;
  EXPECT_TRUE(result.options.time_control.enabled);
  EXPECT_TRUE(result.options.time_control.per_move);
  EXPECT_EQ(result.options.time_control.move_time_ms, 1500u);
}

TEST(CliParseTests, RejectsMoveTimeWithDepth) {
  std::vector<std::string> args = {"skaks", "--move-time", "2000", "--depth",
                                   "6"};
  const auto result = parse(std::move(args));
  EXPECT_TRUE(result.parse_error);
  EXPECT_FALSE(result.options.time_control.enabled);
}

TEST(CliParseTests, RejectsCombinedClockAndMoveTime) {
  std::vector<std::string> args = {"skaks", "--move-time", "2000", "--wtime",
                                   "60000"};
  const auto result = parse(std::move(args));
  EXPECT_TRUE(result.parse_error);
  EXPECT_FALSE(result.options.time_control.enabled);
}

TEST(CliParseTests, AcceptsExplicitClockOptions) {
  std::vector<std::string> args = {"skaks", "--wtime",     "60000", "--btime",
                                   "60000", "--winc",      "1000",  "--binc",
                                   "1000",  "--movestogo", "20"};
  const auto result = parse(std::move(args));
  ASSERT_FALSE(result.parse_error) << result.message;
  EXPECT_TRUE(result.options.time_control.enabled);
  EXPECT_FALSE(result.options.time_control.per_move);
  EXPECT_EQ(result.options.time_control.white_time_ms, 60000u);
  EXPECT_EQ(result.options.time_control.black_time_ms, 60000u);
  EXPECT_EQ(result.options.time_control.white_increment_ms, 1000u);
  EXPECT_EQ(result.options.time_control.black_increment_ms, 1000u);
  EXPECT_EQ(result.options.time_control.moves_to_go, 20u);
}

TEST(CliParseTests, RejectsZeroClockConfiguration) {
  std::vector<std::string> args = {"skaks", "--winc", "1000"};
  const auto result = parse(std::move(args));
  EXPECT_TRUE(result.parse_error);
  EXPECT_FALSE(result.options.time_control.enabled);
}

} // namespace
