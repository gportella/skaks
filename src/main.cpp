#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/demo_debug.hpp"
#include "chess/engine.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <iostream>

int main() {
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
  auto result = chess::alphabeta_minimax(board, 3, -10000, 10000, board.side_to_move);
  std::cout << "Best move score: " << result.score << "\n";
  std::cout << "Best move from " << chess::square_to_string(result.best_move.from) << " to "
            << chess::square_to_string(result.best_move.to) << "\n";
  chess::make_move(board, result.best_move);
  chess::terminal_board_print(board);

  return 0;
}
