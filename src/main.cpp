#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/engine.hpp"
#include "chess/types_io.hpp"

#include <iostream>

int main() {
  const chess::Engine engine;
  chess::Board board = chess::initial_board(chess::kStartFEN);

  std::cout << "Board of color to move: " << board.side_to_move << "\n";

  chess::termianl_board_print(board);

  std::cout << "Sample evaluation: " << engine.sample_evaluation() << "\n";
  return 0;
}
