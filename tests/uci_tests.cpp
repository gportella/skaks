#include "chess/engine.hpp"
#include "chess/uci.hpp"

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

class StreamRedirect {
public:
  StreamRedirect(std::istream& in, std::istringstream& replacement)
      : in_(in), old_buf_(in.rdbuf(replacement.rdbuf())) {}

  ~StreamRedirect() {
    in_.rdbuf(old_buf_);
  }

private:
  std::istream& in_;
  std::streambuf* old_buf_;
};

class OStreamRedirect {
public:
  OStreamRedirect(std::ostream& out, std::ostringstream& replacement)
      : out_(out), old_buf_(out.rdbuf(replacement.rdbuf())),
        old_flags_(out.flags()) {}

  ~OStreamRedirect() {
    out_.rdbuf(old_buf_);
    out_.flags(old_flags_);
  }

private:
  std::ostream& out_;
  std::streambuf* old_buf_;
  std::ios::fmtflags old_flags_;
};

class StdoutBufferGuard {
public:
  StdoutBufferGuard() = default;
  ~StdoutBufferGuard() {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
  }
};

} // namespace

namespace chess {
namespace {

TEST(UciLoopTests, AdvertisesThreadsOption) {
  Engine engine;
  std::istringstream input("uci\nquit\n");
  std::ostringstream output;

  StreamRedirect cin_redirect(std::cin, input);
  OStreamRedirect cout_redirect(std::cout, output);
  StdoutBufferGuard buf_guard;

  run_uci_loop(engine, 4);

  const std::string out = output.str();
  EXPECT_NE(out.find("option name Threads"), std::string::npos);
}

TEST(UciLoopTests, ThreadsOptionClampsToSupportedRange) {
  Engine engine;
  std::istringstream input("uci\nsetoption name Threads value 4\nquit\n");
  std::ostringstream output;

  StreamRedirect cin_redirect(std::cin, input);
  OStreamRedirect cout_redirect(std::cout, output);
  StdoutBufferGuard buf_guard;

  run_uci_loop(engine, 4);

  const std::string out = output.str();
  const unsigned detected = std::thread::hardware_concurrency();
  const int runtime_max =
      std::clamp(static_cast<int>(detected == 0 ? 1 : detected), 1, 256);
  const int expected_threads = std::clamp(4, 1, runtime_max);
  EXPECT_EQ(engine.thread_count(), expected_threads);

  const auto limit_msg = out.find("Threads option is limited to");
  if (expected_threads < 4) {
    EXPECT_NE(limit_msg, std::string::npos);
    if (limit_msg != std::string::npos) {
      EXPECT_NE(out.find(std::to_string(runtime_max), limit_msg),
                std::string::npos);
    }
  } else {
    EXPECT_EQ(limit_msg, std::string::npos);
  }
}

} // namespace
} // namespace chess
