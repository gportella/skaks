#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/demo_debug.hpp"
#include "chess/engine.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>

int main() {
  const bool profile = std::getenv("SKAKS_PROFILE") != nullptr;
  std::chrono::steady_clock::time_point total_start{};
  if (profile) {
    total_start = std::chrono::steady_clock::now();
  }

  const chess::Engine engine;
  chess::Board board = chess::initial_board(chess::kStartFEN);

  std::cout << "Position key: 0x" << std::hex << board.position_key << std::dec << "\n";
  std::cout << "Board of color to move: " << board.side_to_move << "\n";
  chess::terminal_board_print(board);
  // chess::test_masks();
  // std::cout << "Checking attacks for black side:\n";
  // chess::dump_attacks(board, chess::flip_side(board.side_to_move));
  // std::cout << "Checking attacks for white side:\n";
  // chess::dump_attacks(board, board.side_to_move);
  int move_number = 1;
  while (true) {
    std::chrono::steady_clock::time_point move_start{};
    if (profile) {
      move_start = std::chrono::steady_clock::now();
    }
    std::cout << "Move: " << (move_number / 2 + 1) << " Ply: " << move_number << ", "
              << board.side_to_move << " to move.\n";
    auto result = chess::alphabeta_minimax(board, 5, -10000, 10000, board.side_to_move);
    if (profile) {
      const auto move_end = std::chrono::steady_clock::now();
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(move_end - move_start);
      std::cout << "[timing] search_ms=" << elapsed.count() << "\n";
    }
    std::cout << "Best move score: " << result.score << "\n";
    std::cout << "Best move from " << chess::square_to_string(result.best_move.from) << " to "
              << chess::square_to_string(result.best_move.to) << "\n";
    chess::make_move(board, result.best_move);
    chess::terminal_board_print(board);
    std::cout << "FEN: " << chess::board_to_fen(board) << "\n\n";
    // board.side_to_move = chess::flip_side(board.side_to_move);
    move_number++;
    if (move_number > 30 || board.is_terminal()) {
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
