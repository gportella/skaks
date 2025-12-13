#pragma once

#include "chess/board.hpp"
#include "chess/defaults.hpp"
#include "chess/history.hpp"
#include "chess/types.hpp"

#include <array>
#include <cstdint>

namespace chess {

constexpr uint8_t kFlagEnPassant = 1u << 0;
constexpr uint8_t kFlagDoublePush = 1u << 1;
constexpr uint8_t kFlagCastle = 1u << 2;
constexpr uint8_t kFlagCastleLong = 1u << 3;
constexpr uint8_t kFlagQuiet = 1u << 4;

inline constexpr uint32_t encode_move(int from, int to, OccupancyType moving_piece,
                                      OccupancyType captured_piece, OccupancyType promo_piece,
                                      uint8_t flags) {
  return (uint32_t(from) & 0x3F) | ((uint32_t(to) & 0x3F) << 6) |
         ((uint32_t(moving_piece) & 0x0F) << 12) | ((uint32_t(captured_piece) & 0x0F) << 16) |
         ((uint32_t(promo_piece) & 0x0F) << 20) | (uint32_t(flags) << 24);
}

inline constexpr uint16_t move_from(uint32_t m) {
  return m & 0x3F;
}
inline constexpr uint16_t move_to(uint32_t m) {
  return (m >> 6) & 0x3F;
}
inline constexpr uint16_t move_piece(uint32_t m) {
  return (m >> 12) & 0x0F;
}
inline constexpr uint16_t move_captured(uint32_t m) {
  return (m >> 16) & 0x0F;
}
inline constexpr uint16_t move_promo(uint32_t m) {
  return (m >> 20) & 0x0F;
}
inline constexpr uint8_t move_flags(uint32_t m) {
  return static_cast<uint8_t>(m >> 24);
}

inline constexpr bool flag_is_ep(uint8_t flags) {
  return (flags & kFlagEnPassant) != 0;
}
inline constexpr bool flag_is_double_push(uint8_t flags) {
  return (flags & kFlagDoublePush) != 0;
}
inline constexpr bool flag_is_castle(uint8_t flags) {
  return (flags & kFlagCastle) != 0;
}
inline constexpr bool flag_is_long_castle(uint8_t flags) {
  return (flags & kFlagCastleLong) != 0;
}
inline constexpr bool flag_is_quiet(uint8_t flags) {
  return (flags & kFlagQuiet) != 0;
}

inline constexpr bool move_is_ep(uint32_t m) {
  return flag_is_ep(move_flags(m));
}
inline constexpr bool move_is_double_push(uint32_t m) {
  return flag_is_double_push(move_flags(m));
}
inline constexpr bool move_is_castle(uint32_t m) {
  return flag_is_castle(move_flags(m));
}
inline constexpr bool move_is_long_castle(uint32_t m) {
  return flag_is_long_castle(move_flags(m));
}
inline constexpr bool move_is_quiet(uint32_t m) {
  return flag_is_quiet(move_flags(m));
}

struct Move {
  uint16_t from, to;
  OccupancyType moving_pc;
  OccupancyType captured_pc; // OccupancyType::empty if none
  OccupancyType promo_pc;    // OccupancyType::empty if none
  uint8_t flags;
};

struct PieceListChange {
  std::size_t index;
  Square old_square;
  bool used;
};

struct Undo {
  uint16_t from;
  uint16_t to;
  uint16_t captured_sq;
  std::array<Bitboard, MAX_BB_UPDATE_PIECES> pieces_bb = {0};
  Bitboard occupancy[3];
  std::uint64_t position_key_before;
  int fifty_move_counter_before;
  int en_passant_before;
  CastlingRights castling_rights_before;
  OccupancyType captured_pc;
  OccupancyType moving_pc;
  OccupancyType promo_pc;
  uint8_t flags;
  bool was_en_passant;
  bool was_castling;
  PieceListChange list_rook_change;
  PieceListChange list_king_change;
  PieceListChange list_pawn_change;
};

struct SearchResult {
  int score;
  Move best_move;
};

struct ThreadState {
  Board board; // mutated in-place
  MoveHistory move_history;
  int repetition_start = 0; // index after last irreversible move
};

inline constexpr Move decode_move(uint32_t encoded_move) {
  Move m;
  m.from = move_from(encoded_move);
  m.to = move_to(encoded_move);
  m.moving_pc = static_cast<OccupancyType>(move_piece(encoded_move));
  m.captured_pc = static_cast<OccupancyType>(move_captured(encoded_move));
  m.promo_pc = static_cast<OccupancyType>(move_promo(encoded_move));
  m.flags = move_flags(encoded_move);
  return m;
}
Undo make_move(Board& b, const Move& m);
int update_castling_rights(Board& b, const Move&);
void undo_move(Board& b, const Undo& u);
SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta, SideToMove stm);
} // namespace chess