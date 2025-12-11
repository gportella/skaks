#include "chess/demo_debug.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/engine.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <ranges>
#include <string_view>
#include <utility>

namespace chess::debug {

template <typename Emitter>
void dump_moves(std::string_view label, Emitter&& emitter, std::ostream& os = std::cout,
                std::uint16_t max_to_show = 10) {
  std::array<std::uint32_t, kMaxMovementCount> moves{};
  std::uint16_t count = 0;
  std::forward<Emitter>(emitter)(moves, count);

  os << label << " (" << count << ")\n";

  const std::uint16_t limit = std::min<std::uint16_t>(count, max_to_show);
  for (std::uint16_t i = 0; i < limit; ++i) {
    const auto move = moves[i];
    const auto from = move_from(move);
    const auto to = move_to(move);
    const auto captured = move_captured(move);
    const auto promo = move_promo(move);

    os << i << ": " << square_to_string(from) << " -> " << square_to_string(to);

    if (captured != static_cast<int>(OccupancyType::empty)) {
      os << " x" << to_string(static_cast<OccupancyType>(captured));
    }

    if (promo != static_cast<int>(OccupancyType::empty)) {
      os << " =" << to_string(static_cast<OccupancyType>(promo));
    }

    os << '\n';
  }

  if (count > max_to_show) {
    os << "..." << '\n';
  }

  os << '\n';
}

void check_attacks(Board& b, SideToMove stm) {
  for (auto sq : std::views::iota(0, 64)) {
    const bool attacked = is_square_attacked(b, static_cast<u_int8_t>(sq), stm);
    if (attacked) {
      std::cout << "Square " << square_to_string(static_cast<int>(sq)) << " is attacked by "
                << to_string(stm) << "\n";
    }
  }
}

} // namespace chess::debug

namespace chess {
int test_masks() {
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
  chess::terminal_board_print(board);

  return 0;
}

void dump_attacks(Board& b, SideToMove stm) {
  chess::debug::check_attacks(b, stm);
}

} // namespace chess