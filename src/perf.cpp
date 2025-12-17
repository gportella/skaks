
#include "chess/perf.hpp"

#include "chess/board.hpp"
#include "chess/cli.hpp"
#include "chess/engine.hpp"

namespace chess {

// Runs a simple benchmark loop to gauge search throughput.
int run_perf_mode(Engine& engine, const chess::CliOptions& options) {
  chess::Board board{};
  try {
    board = chess::initial_board(options.fen);
  } catch (const std::exception& ex) {
    std::cerr << "Failed to load FEN: " << ex.what()
              << "\nHint: pass custom positions with --fen \"<fen>\" or -f \"<fen>\"." << std::endl;
    return EXIT_FAILURE;
  }

  chess::SearchParameters params{};
  params.depth = options.search_depth;

  std::cout << "[perf] depth=" << params.depth << " iterations=" << options.perf_iterations
            << " fen=\"" << options.fen << "\"\n";

  std::uint64_t total_nodes = 0;
  std::uint64_t total_ms = 0;

  for (int i = 0; i < options.perf_iterations; ++i) {
    engine.reset_history(board);
    engine.clear_transposition_table();

    auto session = engine.create_session(board);
    auto result = session.run(params);

    total_nodes += result.nodes;
    total_ms += result.elapsed_ms;

    const auto iter_ms = result.elapsed_ms;
    const auto iter_nps = (iter_ms == 0) ? 0 : (result.nodes * 1000ULL) / iter_ms;
    std::cout << "[perf] iter=" << (i + 1) << "/" << options.perf_iterations
              << " nodes=" << result.nodes << " elapsed_ms=" << iter_ms << " nps=" << iter_nps
              << "\n";
  }

  const auto clamped_ms = (total_ms == 0) ? 1 : total_ms;
  const auto total_nps = (total_ms == 0) ? 0 : (total_nodes * 1000ULL) / clamped_ms;
  const auto avg_nodes = total_nodes / static_cast<std::uint64_t>(options.perf_iterations);
  const auto avg_ms = clamped_ms / static_cast<std::uint64_t>(options.perf_iterations);

  std::cout << "[perf] total_nodes=" << total_nodes << " total_ms=" << total_ms
            << " total_nps=" << total_nps << " avg_nodes=" << avg_nodes << " avg_ms=" << avg_ms
            << "\n";

  return EXIT_SUCCESS;
}

} // namespace chess
