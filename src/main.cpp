#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/cli.hpp"
#include "chess/defaults.hpp"
#include "chess/demo_debug.hpp"
#include "chess/engine.hpp"
#include "chess/moves.hpp"
#include "chess/search.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "chess/uci.hpp"
#include "chess/version.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

// Runs a simple benchmark loop to gauge search throughput.
int run_perf_mode(chess::Engine& engine, const chess::CliOptions& options) {
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

} // namespace

int main(int argc, char** argv) {
  const auto cli = chess::parse_cli(argc, argv);
  if (cli.parse_error) {
    std::cerr << "Error: " << cli.message << "\n";
    return EXIT_FAILURE;
  }
  if (cli.show_help) {
    std::cout << cli.message << "\n";
    return EXIT_SUCCESS;
  }

  if (cli.options.show_version) {
    std::cout << chess::kEngineName << " version " << chess::kEngineVersion << "\n";
    if (cli.options.show_extended_version) {
      std::cout << "Optimizations:\n";
      for (const auto feature : chess::kOptimizationFeatures) {
        std::cout << " - " << feature << "\n";
      }
    } else {
      std::cout << "Use -vv for details.\n";
    }
    return EXIT_SUCCESS;
  }

  chess::Engine engine;

  if (cli.options.perf_mode) {
    return run_perf_mode(engine, cli.options);
  }

  if (cli.options.use_uci) {
    chess::run_uci_loop(engine, cli.options.search_depth);
    return EXIT_SUCCESS;
  }

  chess::Board board{};
  try {
    board = chess::initial_board(cli.options.fen);
  } catch (const std::exception& ex) {
    std::cerr << "Failed to load FEN: " << ex.what()
              << "\nHint: pass custom positions with --fen \"<fen>\" or -f \"<fen>\"." << std::endl;
    return EXIT_FAILURE;
  }
  engine.reset_history(board);

  const bool profile = (std::getenv("SKAKS_PROFILE") != nullptr) || cli.options.enable_profile;
  std::chrono::steady_clock::time_point total_start{};
  if (profile) {
    total_start = std::chrono::steady_clock::now();
  }

  if (cli.options.only_fen) {
    std::cout << chess::board_to_fen(board) << "\n";
    std::cout.flush();
  } else {
    std::cout << "Board of color to move: " << board.side_to_move << "\n";
    chess::terminal_board_print(board);
  }
  int move_number = 1;
  const int max_full_moves = cli.options.max_full_moves;

  while (true) {
    std::chrono::steady_clock::time_point move_start{};
    if (profile) {
      move_start = std::chrono::steady_clock::now();
    }
    const int current_full_move = (move_number / 2) + 1;
    if (!cli.options.only_fen) {
      std::cout << "Move: " << current_full_move << " Ply: " << move_number << ", "
                << board.side_to_move << " to move.\n";
    }
    chess::SearchParameters params{};
    params.depth = cli.options.search_depth;
    params.alpha = -10000;
    params.beta = 10000;

    auto result = engine.search(board, params);

    if (profile && !cli.options.only_fen) {
      const auto move_end = std::chrono::steady_clock::now();
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(move_end - move_start);
      std::cout << "[timing] search_ms=" << elapsed.count() << "\n";
    }

    if (!cli.options.only_fen) {
      std::cout << "Best move score: " << result.score << "\n";
    }

    const bool has_move = result.best_move.moving_pc != chess::OccupancyType::empty;
    const auto outcome = result.outcome;
    if (!has_move) {
      if (!cli.options.only_fen) {
        const bool side_in_check = chess::is_check(board, board.side_to_move);
        if (side_in_check) {
          const auto winner = chess::flip_side(board.side_to_move);
          std::cout << "Checkmate! " << chess::to_string(winner) << " wins.\n";
        } else {
          std::cout << "Stalemate.\n";
        }
      }
      break;
    }
    if (!cli.options.only_fen && outcome != chess::SearchResult::Outcome::InProgress) {
      switch (outcome) {
      case chess::SearchResult::Outcome::Mate:
        std::cout << "(search) Mate sequence detected.\n";
        break;
      case chess::SearchResult::Outcome::DrawByRepetition:
        std::cout << "(search) Draw by repetition forecast.\n";
        break;
      case chess::SearchResult::Outcome::DrawByStalemate:
        std::cout << "(search) Stalemate forecast.\n";
        break;
      case chess::SearchResult::Outcome::InProgress:
        break;
      }
    }

    if (!cli.options.only_fen) {
      std::cout << "Best move from " << chess::square_to_string(result.best_move.from) << " to "
                << chess::square_to_string(result.best_move.to) << "\n";
    }
    const bool irreversible = chess::move_is_irreversible(result.best_move);
    chess::make_move(board, result.best_move);
    engine.record_position(board.position_key, irreversible);
    if (cli.options.only_fen) {
      std::cout << chess::board_to_fen(board) << "\n";
      std::cout.flush();
    } else {
      chess::terminal_board_print(board);
      std::cout << "FEN: " << chess::board_to_fen(board) << "\n\n";
    }
    // board.side_to_move = chess::flip_side(board.side_to_move);
    move_number++;

    if (((move_number / 2) + 1 > max_full_moves) || board.is_terminal()) {
      break;
    }
  }
  if (!cli.options.only_fen && board.is_terminal()) {
    std::cout << "Game over detected.\n";
    if (board.king_captured == chess::PieceColor::White) {
      std::cout << "Black wins!\n";
    } else if (board.king_captured == chess::PieceColor::Black) {
      std::cout << "White wins!\n";
    } else {
      std::cout << "Draw!\n";
    }
  }

  if (profile && !cli.options.only_fen) {
    const auto total_end = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "[timing] total_ms=" << elapsed.count() << "\n";
  }

  return 0;
}
