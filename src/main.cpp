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

#include <chrono>
#include <cstdlib>
#include <iostream>

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

  const bool profile = (std::getenv("SKAKS_PROFILE") != nullptr) || cli.options.enable_profile;
  std::chrono::steady_clock::time_point total_start{};
  if (profile) {
    total_start = std::chrono::steady_clock::now();
  }

  chess::Engine engine;
  chess::Board board = chess::initial_board(cli.options.fen);
  engine.reset_history(board);

  std::cout << "Position key: 0x" << std::hex << board.position_key << std::dec << "\n";
  std::cout << "Board of color to move: " << board.side_to_move << "\n";
  chess::terminal_board_print(board);
  int move_number = 1;
  const int max_full_moves = cli.options.max_full_moves;

  while (true) {
    std::chrono::steady_clock::time_point move_start{};
    if (profile) {
      move_start = std::chrono::steady_clock::now();
    }
    const int current_full_move = (move_number / 2) + 1;
    std::cout << "Move: " << current_full_move << " Ply: " << move_number << ", "
              << board.side_to_move << " to move.\n";
    chess::SearchParameters params{};
    params.depth = cli.options.search_depth;
    params.alpha = -10000;
    params.beta = 10000;

    auto result = engine.search(board, params);

    if (profile) {
      const auto move_end = std::chrono::steady_clock::now();
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(move_end - move_start);
      std::cout << "[timing] search_ms=" << elapsed.count() << "\n";
    }

    std::cout << "Best move score: " << result.score << "\n";

    const bool has_move = result.best_move.moving_pc != chess::OccupancyType::empty;
    if (!has_move) {
      const bool side_in_check = chess::is_check(board, board.side_to_move);
      if (side_in_check) {
        const auto winner = chess::flip_side(board.side_to_move);
        std::cout << "Checkmate! " << chess::to_string(winner) << " wins.\n";
      } else {
        std::cout << "Stalemate.\n";
      }
      break;
    }

    std::cout << "Best move from " << chess::square_to_string(result.best_move.from) << " to "
              << chess::square_to_string(result.best_move.to) << "\n";
    const bool irreversible = chess::move_is_irreversible(result.best_move);
    chess::make_move(board, result.best_move);
    engine.record_position(board.position_key, irreversible);
    chess::terminal_board_print(board);
    std::cout << "FEN: " << chess::board_to_fen(board) << "\n\n";
    // board.side_to_move = chess::flip_side(board.side_to_move);
    move_number++;

    if (((move_number / 2) + 1 > max_full_moves) || board.is_terminal()) {
      break;
    }
  }
  if (board.is_terminal()) {
    std::cout << "Game over detected.\n";
    if (board.king_captured == chess::PieceColor::White) {
      std::cout << "Black wins!\n";
    } else if (board.king_captured == chess::PieceColor::Black) {
      std::cout << "White wins!\n";
    } else {
      std::cout << "Draw!\n";
    }
  }

  if (profile) {
    const auto total_end = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "[timing] total_ms=" << elapsed.count() << "\n";
  }

  return 0;
}
