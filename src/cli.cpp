#include "chess/cli.hpp"

#include "chess/defaults.hpp"
#include "cxxopts.hpp"

#include <string>
#include <vector>

namespace chess {

CliParseResult parse_cli(int argc, char** argv) {
  CliParseResult result{};
  if (argc > 0 && argv != nullptr && argv[0] != nullptr) {
    result.options.executable_path = argv[0];
  }
  cxxopts::Options options("skaks", "Skaks chess engine options");

  // clang-format off
  options.add_options()
      ("d,depth", "Search depth in plies",
       cxxopts::value<int>()->default_value("4"))
      ("m,max-moves", "Maximum number of full moves to play",
       cxxopts::value<int>()->default_value(std::to_string(kMaxMovementCount)))
      ("f,fen", "Start position in FEN notation", cxxopts::value<std::string>())
      ("P,profile", "Enable profiling output")
      ("u,uci", "Force UCI protocol mode")
      ("polyglot", "Enable Polyglot book usage")
      ("polyglot-book", "Path to Polyglot opening book",
       cxxopts::value<std::string>())
      ("o,onlyfen", "Print FEN only in self-play mode")
      ("s,self", "Run self-play CLI loop")
      ("bm,bestmove", "Print best move for the given FEN and exit")
      ("v,version", "Show version information (repeat for extended details)")
      ("p,perf", "Run built-in performance benchmark")
      ("perf-iters", "Number of benchmark iterations",
       cxxopts::value<int>()->default_value("3"))
      ("h,help", "Show this help message");
  // clang-format on

  try {
    std::vector<std::string> adjusted_args;
    adjusted_args.reserve(static_cast<std::size_t>(argc > 0 ? argc : 1));
    for (int i = 0; i < argc; ++i) {
      if (argv == nullptr || argv[i] == nullptr) {
        continue;
      }
      std::string current = argv[i];
      if (current == "-bm") {
        current = "--bestmove";
      }
      adjusted_args.push_back(std::move(current));
    }
    if (adjusted_args.empty()) {
      adjusted_args.emplace_back("skaks");
    }
    std::vector<char*> arg_ptrs;
    arg_ptrs.reserve(adjusted_args.size());
    for (auto& arg : adjusted_args) {
      arg_ptrs.push_back(arg.data());
    }
    const int adj_argc = static_cast<int>(adjusted_args.size());

    const auto parsed = options.parse(adj_argc, arg_ptrs.data());

    if (parsed.count("help")) {
      result.show_help = true;
      result.message = options.help();
      return result;
    }

    result.options.search_depth = parsed["depth"].as<int>();
    result.options.max_full_moves = parsed["max-moves"].as<int>();

    if (parsed.count("fen")) {
      result.options.fen = parsed["fen"].as<std::string>();
      result.options.use_custom_fen = true;
    }

    result.options.enable_profile = parsed.count("profile") > 0;
    result.options.only_fen = parsed.count("onlyfen") > 0;
    result.options.best_move = parsed.count("bestmove") > 0;
    result.options.polyglot = parsed.count("polyglot") > 0;
    if (parsed.count("polyglot-book") > 0) {
      result.options.polyglot_book_path =
          parsed["polyglot-book"].as<std::string>();
      result.options.polyglot_book_override = true;
      result.options.polyglot = true;
    }
    result.options.perf_mode = parsed.count("perf") > 0;
    result.options.perf_iterations = parsed["perf-iters"].as<int>();
    const auto version_requests = parsed.count("version");
    result.options.version_requests = static_cast<int>(version_requests);
    result.options.show_version = version_requests > 0;
    result.options.show_extended_version = version_requests > 1;

    const bool want_uci = parsed.count("uci") > 0;
    const bool want_self = parsed.count("self") > 0;
    if (result.options.best_move && (want_uci || want_self)) {
      result.parse_error = true;
      result.message = "--bestmove cannot be combined with --uci or --self";
      return result;
    }
    if (want_uci && want_self) {
      result.parse_error = true;
      result.message = "--uci and --self cannot be used together";
      return result;
    }
    if (want_uci && parsed.count("perf") > 0) {
      result.parse_error = true;
      result.message = "--uci and --perf cannot be used together";
      return result;
    }
    result.options.self_play = want_self;
    result.options.use_uci = want_uci || !want_self;

    if (result.options.perf_mode) {
      if (result.options.best_move) {
        result.parse_error = true;
        result.message = "--bestmove cannot be combined with --perf";
        return result;
      }
      result.options.self_play = false;
      result.options.use_uci = false;
    }

    if (result.options.best_move) {
      result.options.self_play = false;
      result.options.use_uci = false;
      result.options.perf_mode = false;
    }

    if (!want_uci && !want_self && !result.options.show_version &&
        !result.options.perf_mode && !result.options.best_move) {
      // No explicit mode requested: default to UCI when no extra args were
      // provided.
      const bool user_supplied_args = adj_argc > 1;
      if (user_supplied_args) {
        result.options.self_play = true;
        result.options.use_uci = false;
      }
    }

    if (result.options.show_version) {
      result.options.perf_mode = false;
      result.options.self_play = false;
      result.options.use_uci = false;
    }

    if (result.options.use_uci) {
      result.options.polyglot = true;
    }

    if (result.options.search_depth < 0) {
      result.parse_error = true;
      result.message = "depth must be non-negative";
      return result;
    }

    if (result.options.max_full_moves <= 0) {
      result.parse_error = true;
      result.message = "max-moves must be positive";
      return result;
    }

    if (result.options.perf_iterations <= 0) {
      result.parse_error = true;
      result.message = "perf-iters must be positive";
      return result;
    }
  } catch (const cxxopts::exceptions::exception& ex) {
    result.parse_error = true;
    result.message = ex.what();
    return result;
  }

  if (!result.options.use_custom_fen) {
    result.options.fen = kStartFEN;
  }

  return result;
}

} // namespace chess
