#pragma once
#include "chess/defaults.hpp"
#include "chess/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>

namespace chess {

inline int file_of(int sq) {
  return sq & 7;
}
inline int rank_of(int sq) {
  return sq >> 3;
}
inline std::uint64_t bb_of(int sq) {
  return 1ULL << sq;
}

// You must implement these tiny predicates based on your OccupancyType:
inline bool is_white_pawn(OccupancyType o) {
  return o == OccupancyType::wP;
}
inline bool is_black_pawn(OccupancyType o) {
  return o == OccupancyType::bP;
}

struct SearchContext {
  std::array<Bitboard, kMaxMovementCount> key_stack{};
  std::size_t key_stack_size{0};

  void push_key(Bitboard key) {
    key_stack[key_stack_size++] = key;
  }
  void pop_key() {
    --key_stack_size;
  }
  std::span<const Bitboard> keys() const {
    return {key_stack.data(), key_stack_size};
  }
};

struct PieceList {
  std::array<Square, 10> squares{}; // max pawns per side is 8, rooks 2
  std::uint8_t count{0};
};

struct Board {
  std::array<Bitboard, kPieceCount + 1> pieces_bb = {0};
  std::array<OccupancyType, 64> pieces = {OccupancyType::empty};
  Bitboard occupancy[3];
  SideToMove side_to_move = SideToMove::White;
  CastlingRights castling_rights = CastlingRights::NoCastling;
  Bitboard ep_square = 0;
  int en_passant = -1;
  int ply_count = 0;
  int fifty_move_counter = 0;
  uint64_t position_key = 0;
  PieceColor king_captured = PieceColor::None;
  std::array<int, 2> king_positions = {-1, -1};
  PieceList rook_list[2];
  PieceList king_list[2];
  PieceList pawn_list[2];
  bool is_terminal() const {
    // todo: check for checkmate/stalemate properly
    return king_captured != PieceColor::None;
  }
};

inline FenFields split_fen(std::string_view fen) {
  FenFields fields{};
  std::array<std::string_view, 6> pieces{};
  std::size_t idx = 0;

  for (auto token : fen | std::views::split(' ')) {
    if (idx >= pieces.size())
      break;
    pieces[idx++] = std::string_view(token);
  }
  if (idx != pieces.size()) {
    throw std::invalid_argument("FEN must have 6 fields separated by spaces");
  }

  fields.placement = pieces[0];
  fields.side_to_move = pieces[1];
  fields.castling = pieces[2];
  fields.en_passant = pieces[3];
  fields.halfmove_clock = pieces[4];
  fields.fullmove_number = pieces[5];
  return fields;
}

Board initial_board(std::string_view fen);
void terminal_board_print(const Board& board);
void terminal_mask_print(Bitboard mask, const Board& board);
bool ep_capture_available(const Board& b);
Bitboard calculate_occupancy(const Board& board, PieceColor color);

} // namespace chess