#include "chess/cli.hpp"

#include "chess/defaults.hpp"
#include "cxxopts.hpp"

#include <cstdint>
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
      ("polyglot", "Enable Polyglot book usage (default: enabled)")
      ("no-polyglot", "Disable Polyglot book usage")
      ("polyglot-book", "Path to Polyglot opening book",
       cxxopts::value<std::string>())
      ("params", "Path to YAML engine parameter file",
        cxxopts::value<std::string>())
            ("nnue", "Path to NNUE weights YAML file",
        cxxopts::value<std::string>())
      ("static-eval", "Print static evaluation for provided FEN",
       cxxopts::value<bool>()->default_value("false"))
      ("o,onlyfen", "Print FEN only in self-play mode")
      ("s,self", "Run self-play CLI loop")
      ("arena", "Run built-in baseline-vs-params arena (no UCI)")
      ("arena-games", "Number of games for arena mode",
       cxxopts::value<int>()->default_value("20"))
      ("bm,bestmove", "Print best move for the given FEN and exit")
      ("v,version", "Show version information (repeat for extended details)")
      ("p,perf", "Run built-in performance benchmark")
      ("perf-iters", "Number of benchmark iterations",
         cxxopts::value<int>()->default_value("3"))
        ("move-time", "Time allowed per move in milliseconds",
         cxxopts::value<std::uint64_t>())
        ("wtime", "White remaining time in milliseconds",
         cxxopts::value<std::uint64_t>())
        ("btime", "Black remaining time in milliseconds",
         cxxopts::value<std::uint64_t>())
        ("winc", "White increment in milliseconds",
         cxxopts::value<std::uint64_t>())
        ("binc", "Black increment in milliseconds",
         cxxopts::value<std::uint64_t>())
        ("movestogo", "Estimated moves to the next time control",
         cxxopts::value<std::uint32_t>())
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
    result.options.static_eval = parsed.count("static-eval") > 0;

    const bool request_polyglot = parsed.count("polyglot") > 0;
    const bool request_no_polyglot = parsed.count("no-polyglot") > 0;

    if (request_polyglot && request_no_polyglot) {
      result.parse_error = true;
      result.message = "--polyglot cannot be combined with --no-polyglot";
      return result;
    }

    result.options.polyglot = !request_no_polyglot;

    if (parsed.count("polyglot-book") > 0) {
      if (request_no_polyglot) {
        result.parse_error = true;
        result.message =
            "--polyglot-book cannot be used when Polyglot is disabled";
        return result;
      }
      result.options.polyglot_book_path =
          parsed["polyglot-book"].as<std::string>();
      result.options.polyglot_book_override = true;
      result.options.polyglot = true;
    }

    if (request_polyglot) {
      result.options.polyglot = true;
    }

    if (parsed.count("params") > 0) {
      result.options.params_path = parsed["params"].as<std::string>();
      result.options.params_override = true;
    }
    if (parsed.count("nnue") > 0) {
      result.options.nnue_path = parsed["nnue"].as<std::string>();
      result.options.nnue_override = true;
    }
    result.options.perf_mode = parsed.count("perf") > 0;
    result.options.perf_iterations = parsed["perf-iters"].as<int>();
    const auto version_requests = parsed.count("version");
    result.options.version_requests = static_cast<int>(version_requests);
    result.options.show_version = version_requests > 0;
    result.options.show_extended_version = version_requests > 1;

    const bool has_move_time = parsed.count("move-time") > 0;
    const bool has_clock_fields =
        (parsed.count("wtime") > 0) || (parsed.count("btime") > 0) ||
        (parsed.count("winc") > 0) || (parsed.count("binc") > 0) ||
        (parsed.count("movestogo") > 0);

    if (has_move_time && has_clock_fields) {
      result.parse_error = true;
      result.message = "--move-time cannot be combined with clock options";
      return result;
    }

    if (has_move_time) {
      const auto move_time_val = parsed["move-time"].as<std::uint64_t>();
      if (move_time_val == 0) {
        result.parse_error = true;
        result.message = "--move-time must be positive";
        return result;
      }
      if (parsed.count("depth") > 0) {
        result.parse_error = true;
        result.message = "--depth cannot be combined with --move-time";
        return result;
      }
      result.options.time_control.enabled = true;
      result.options.time_control.per_move = true;
      result.options.time_control.move_time_ms = move_time_val;
    } else if (has_clock_fields) {
      if (parsed.count("depth") > 0) {
        result.parse_error = true;
        result.message =
            "--depth cannot be combined with explicit clock options";
        return result;
      }
      const auto wtime =
          parsed.count("wtime") ? parsed["wtime"].as<std::uint64_t>() : 0;
      const auto btime =
          parsed.count("btime") ? parsed["btime"].as<std::uint64_t>() : 0;
      const auto winc =
          parsed.count("winc") ? parsed["winc"].as<std::uint64_t>() : 0;
      const auto binc =
          parsed.count("binc") ? parsed["binc"].as<std::uint64_t>() : 0;
      const auto moves_to_go = parsed.count("movestogo")
                                   ? parsed["movestogo"].as<std::uint32_t>()
                                   : 0U;

      if (wtime == 0 && btime == 0) {
        result.parse_error = true;
        result.message =
            "Clock options require --wtime or --btime to be greater than 0";
        return result;
      }

      result.options.time_control.enabled = true;
      result.options.time_control.per_move = false;
      result.options.time_control.white_time_ms = wtime;
      result.options.time_control.black_time_ms = btime;
      result.options.time_control.white_increment_ms = winc;
      result.options.time_control.black_increment_ms = binc;
      result.options.time_control.moves_to_go = moves_to_go;
    }

    const bool want_uci = parsed.count("uci") > 0;
    const bool want_self = parsed.count("self") > 0;
    const bool want_arena = parsed.count("arena") > 0;
    if (result.options.best_move && (want_uci || want_self || want_arena)) {
      result.parse_error = true;
      result.message =
          "--bestmove cannot be combined with --uci, --self, or --arena";
      return result;
    }
    if (result.options.static_eval && (want_uci || want_self || want_arena)) {
      result.parse_error = true;
      result.message =
          "--static-eval cannot be combined with --uci, --self, or --arena";
      return result;
    }
    if ((want_uci && want_self) || (want_uci && want_arena) ||
        (want_self && want_arena)) {
      result.parse_error = true;
      result.message = "--uci, --self, and --arena are mutually exclusive";
      return result;
    }
    if (want_uci && parsed.count("perf") > 0) {
      result.parse_error = true;
      result.message = "--uci and --perf cannot be used together";
      return result;
    }
    result.options.self_play = want_self;
    result.options.arena_mode = want_arena;
    result.options.use_uci = want_uci || (!want_self && !want_arena);

    if (result.options.perf_mode) {
      if (result.options.best_move) {
        result.parse_error = true;
        result.message = "--bestmove cannot be combined with --perf";
        return result;
      }
      if (result.options.static_eval) {
        result.parse_error = true;
        result.message = "--static-eval cannot be combined with --perf";
        return result;
      }
      if (result.options.arena_mode) {
        result.parse_error = true;
        result.message = "--arena cannot be combined with --perf";
        return result;
      }
      result.options.self_play = false;
      result.options.use_uci = false;
    }

    if (result.options.best_move) {
      result.options.self_play = false;
      result.options.use_uci = false;
      result.options.perf_mode = false;
      result.options.arena_mode = false;
    }

    if (result.options.static_eval) {
      result.options.self_play = false;
      result.options.use_uci = false;
      result.options.perf_mode = false;
      result.options.arena_mode = false;
    }

    if (!want_uci && !want_self && !want_arena && !result.options.show_version &&
        !result.options.perf_mode && !result.options.best_move &&
        !result.options.static_eval) {
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

    if (result.options.use_uci && !request_no_polyglot) {
      result.options.polyglot = true;
    }

    if (result.options.arena_mode) {
      result.options.arena_games = parsed["arena-games"].as<int>();
      if (result.options.arena_games <= 0) {
        result.parse_error = true;
        result.message = "--arena-games must be positive";
        return result;
      }
      // Force Polyglot off for deterministic arena runs.
      result.options.polyglot = false;
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

    if (result.options.time_control.enabled && result.options.perf_mode) {
      result.parse_error = true;
      result.message = "Time controls cannot be combined with --perf";
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
