#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/debug.hpp"
#include "chess/defaults.hpp"
#include "chess/engine.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <iostream>

int main() {
  const chess::Engine engine;
  chess::Board board = chess::initial_board(chess::kStartFEN);

  std::cout << "Board of color to move: " << board.side_to_move << "\n";

  chess::terminal_board_print(board);

  chess::terminal_mask_print(
      chess::rook_attack_bm(board, static_cast<u_int8_t>(chess::Square::A1), board.side_to_move) |
          chess::bishop_attack_bm(board, static_cast<u_int8_t>(chess::Square::C1),
                                  board.side_to_move),
      board);

  chess::terminal_mask_print(
      chess::queen_attack_bm(board, static_cast<u_int8_t>(chess::Square::D1), board.side_to_move),
      board);

  chess::PawnMasks pawn_masks = chess::gen_pawn_masks(board, board.side_to_move);
  chess::PawnMasks black_pawn_masks =
      chess::gen_pawn_masks(board, chess::flip_side(board.side_to_move));
  std::cout << "White pawn single pushes:\n" << std::endl;
  chess::terminal_mask_print(pawn_masks.single_push, board);
  std::cout << "White pawn double pushes:\n" << std::endl;
  chess::terminal_mask_print(pawn_masks.double_push, board);
  std::cout << "White pawn captures:\n" << std::endl;
  chess::terminal_mask_print(pawn_masks.captures, board);

  std::cout << "Black pawn single pushes:\n" << std::endl;
  chess::terminal_mask_print(black_pawn_masks.single_push, board);
  std::cout << "Black pawn double pushes:\n" << std::endl;
  chess::terminal_mask_print(black_pawn_masks.double_push, board);

  std::cout << "Knight attacks from G1:\n" << std::endl;
  chess::terminal_mask_print(
      chess::knight_attack_bm(board, static_cast<u_int8_t>(chess::Square::G1), board.side_to_move),
      board);

  chess::terminal_mask_print(
      chess::queen_attack_bm(board, static_cast<u_int8_t>(chess::Square::D1), board.side_to_move),
      board);

  std::cout << "\nBlack king attacks from e8:\n" << std::endl;
  chess::terminal_mask_print(chess::king_attack_bm(board, static_cast<u_int8_t>(chess::Square::E8),
                                                   chess::flip_side(board.side_to_move)),
                             board);
  std::cout << "\nWhite king attacks from e1:\n" << std::endl;
  chess::terminal_mask_print(
      chess::king_attack_bm(board, static_cast<u_int8_t>(chess::Square::E1), board.side_to_move),
      board);

  std::cout << "Sample evaluation: " << engine.sample_evaluation() << "\n";

  chess::debug::dump_moves("White pawn moves", [&](auto& moves, auto& count) {
    chess::emit_white_pawn_moves(board, pawn_masks, moves, count);
  });
  chess::debug::dump_moves("Black pawn moves", [&](auto& moves, auto& count) {
    chess::emit_black_pawn_moves(board, black_pawn_masks, moves, count);
  });
  chess::debug::dump_moves("Knight moves", [&](auto& moves, auto& count) {
    chess::emit_knight_moves(board, board.side_to_move, moves, count);
  });
  chess::debug::dump_moves("Bishop moves", [&](auto& moves, auto& count) {
    chess::emit_bishop_moves(board, board.side_to_move, moves, count);
  });
  chess::debug::dump_moves("Rook moves", [&](auto& moves, auto& count) {
    chess::emit_rook_moves(board, board.side_to_move, moves, count);
  });
  chess::debug::dump_moves("Queen moves", [&](auto& moves, auto& count) {
    chess::emit_queen_moves(board, board.side_to_move, moves, count);
  });
  chess::debug::dump_moves("King moves", [&](auto& moves, auto& count) {
    chess::emit_king_moves(board, board.side_to_move, moves, count);
  });

  return 0;
}
