#include "chess/cli.hpp"

#include "chess/defaults.hpp"
#include "cxxopts.hpp"

namespace chess {

CliParseResult parse_cli(int argc, char** argv) {
  CliParseResult result{};
  cxxopts::Options options("skaks", "Skaks chess engine options");

  options.add_options()("d,depth", "Search depth in plies",
                        cxxopts::value<int>()->default_value("4"))(
      "m,max-moves", "Maximum number of full moves to play",
      cxxopts::value<int>()->default_value(std::to_string(kMaxMovementCount)))(
      "f,fen", "Start position in FEN notation", cxxopts::value<std::string>())(
      "p,profile", "Enable profiling output")("u,uci", "Force UCI protocol mode")(
      "s,self", "Run self-play CLI loop")("h,help", "Show this help message");

  try {
    const auto parsed = options.parse(argc, argv);

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

    const bool want_uci = parsed.count("uci") > 0;
    const bool want_self = parsed.count("self") > 0;
    if (want_uci && want_self) {
      result.parse_error = true;
      result.message = "--uci and --self cannot be used together";
      return result;
    }

    result.options.self_play = want_self;
    result.options.use_uci = want_uci || !want_self;

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
