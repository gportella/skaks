#pragma once

#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/magic_bitboards.hpp"
#include "chess/ray_tables.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "moves.hpp"

#include <array>
#include <ostream>
#include <string>
#include <string_view>

namespace chess {

inline std::array<Bitboard, 64> build_knight_attacks() {
  std::array<Bitboard, 64> A{};
  for (size_t sq = 0; sq < 64; ++sq) {
    Bitboard b = Bitboard(1) << sq;
    Bitboard m = 0;

    // Two left, one up/down
    if (b & ~(FILE_A | FILE_B)) {
      m |= (b << 6);  // 2L 1U
      m |= (b >> 10); // 2L 1D
    }
    // Two right, one up/dow
    if (b & ~(FILE_H | FILE_G)) {
      m |= (b << 10); // 2R 1U
      m |= (b >> 6);  // 2R 1D
    }
    // One left, two up/down
    if (b & ~FILE_A) {
      m |= (b << 15); // 1L 2U
      m |= (b >> 17); // 1L 2D
    }
    // One right, two up/down
    if (b & ~FILE_H) {
      m |= (b << 17); // 1R 2U
      m |= (b >> 15); // 1R 2D
    }

    A[sq] = m;
  }
  return A;
}

inline const std::array<Bitboard, 64> KNIGHT_ATTACKS = build_knight_attacks();

inline std::array<Bitboard, 64> build_king_attack_patterns() {
  std::array<Bitboard, 64> attacks{};
  for (std::size_t sq = 0; sq < 64; ++sq) {
    Bitboard bit = Bitboard(1) << sq;
    Bitboard mask = 0;
    mask |= (bit << 8) & U64_MASK;
    mask |= (bit >> 8) & U64_MASK;
    if (bit & NOT_FILE_H) {
      mask |= (bit << 9) & U64_MASK;
      mask |= (bit >> 7) & U64_MASK;
      mask |= (bit << 1) & U64_MASK;
    }
    if (bit & NOT_FILE_A) {
      mask |= (bit << 7) & U64_MASK;
      mask |= (bit >> 9) & U64_MASK;
      mask |= (bit >> 1) & U64_MASK;
    }
    attacks[sq] = Bitboard(mask & U64_MASK);
  }
  return attacks;
}

inline const std::array<Bitboard, 64> KING_ATTACKS =
    build_king_attack_patterns();

inline Bitboard rook_attacks_from(u_int8_t sq, Bitboard b_occupancy) {
  return rook_attacks_magic(sq, b_occupancy);
}

inline Bitboard bishop_attacks_from(u_int8_t sq, Bitboard b_occupancy) {
  return bishop_attacks_magic(sq, b_occupancy);
}

// pawns are a pain

inline std::array<Bitboard, 64> build_white_pawn_attacks() {
  std::array<Bitboard, 64> a{};
  for (int sq = 0; sq < 64; ++sq) {
    Bitboard b = Bitboard(1) << sq;
    Bitboard m = 0;
    if (b & NOT_FILE_A)
      m |= (b << 7);
    if (b & NOT_FILE_H)
      m |= (b << 9);
    a[static_cast<std::size_t>(sq)] = m & U64_MASK;
  }
  return a;
}

inline std::array<Bitboard, 64> build_black_pawn_attacks() {
  std::array<Bitboard, 64> a{};
  for (int sq = 0; sq < 64; ++sq) {
    Bitboard b = Bitboard(1) << sq;
    Bitboard m = 0;
    if (b & NOT_FILE_H)
      m |= (b >> 7);
    if (b & NOT_FILE_A)
      m |= (b >> 9);
    a[static_cast<std::size_t>(sq)] = m & U64_MASK;
  }
  return a;
}

struct PawnMoves {
  Bitboard nonpromo_push;
  Bitboard double_push;
  Bitboard nonpromo_caps;
  Bitboard promo_push;
  Bitboard promo_caps;
  Bitboard ep_caps;
};

inline PawnMoves gen_white_pawn_moves(Bitboard wp, Bitboard occ,
                                      Bitboard black_occ, Bitboard ep) {
  Bitboard empty = ~occ;

  Bitboard single = (wp << 8) & empty;
  Bitboard doubles = ((wp & RANK_2) << 8) & empty;
  doubles = (doubles << 8) & empty;

  Bitboard capL = ((wp & NOT_FILE_A) << 7) & black_occ;
  Bitboard capR = ((wp & NOT_FILE_H) << 9) & black_occ;

  Bitboard promo_push = single & RANK_8;
  Bitboard nonpromo_push = single & ~RANK_8;

  Bitboard promo_caps = (capL | capR) & RANK_8;
  Bitboard nonpromo_caps = (capL | capR) & ~RANK_8;

  Bitboard ep_caps =
      ((((wp & NOT_FILE_A) << 7) | ((wp & NOT_FILE_H) << 9)) & ep);

  return {nonpromo_push, doubles,    nonpromo_caps,
          promo_push,    promo_caps, ep_caps};
}
struct PawnMasks {
  Bitboard single_push;    // non-promo single pushes
  Bitboard double_push;    // double pawn pushes
  Bitboard captures;       // non-promo captures
  Bitboard promo_push;     // single pushes to promotion rank
  Bitboard promo_captures; // captures to promotion rank
  Bitboard ep_captures;    // en passant captures
};

struct SmallestAttacker {
  Piece piece;
  u_int8_t square;
};

// Declarations

bool is_square_attacked(const Board& board, u_int8_t sq,
                        SideToMove attacker_side);

PawnMasks gen_pawn_masks(const Board& board, SideToMove stm);
Bitboard knight_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard bishop_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard rook_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard queen_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard king_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);

void emit_white_pawn_moves(const Board& board, const PawnMasks& masks,
                           std::array<uint32_t, kMaxMovementCount>& out,
                           std::uint16_t& move_count);
void emit_black_pawn_moves(const Board& board, const PawnMasks& masks,
                           std::array<uint32_t, kMaxMovementCount>& out,
                           std::uint16_t& move_count);

void emit_knight_moves(const Board& board, SideToMove stm,
                       std::array<uint32_t, kMaxMovementCount>& out,
                       std::uint16_t& move_count);
void emit_bishop_moves(const Board& board, SideToMove stm,
                       std::array<uint32_t, kMaxMovementCount>& out,
                       std::uint16_t& move_count);
void emit_rook_moves(const Board& board, SideToMove stm,
                     std::array<uint32_t, kMaxMovementCount>& out,
                     std::uint16_t& move_count);
void emit_queen_moves(const Board& board, SideToMove stm,
                      std::array<uint32_t, kMaxMovementCount>& out,
                      std::uint16_t& move_count);
void emit_king_moves(const Board& board, SideToMove stm,
                     std::array<uint32_t, kMaxMovementCount>& out,
                     std::uint16_t& move_count);

void emit_all_moves(const Board& board, SideToMove stm,
                    std::array<uint32_t, kMaxMovementCount>& out,
                    std::uint16_t& move_count);
CastlingRights king_castle_rights(const Board& board, SideToMove stm);
bool is_square_attacked(const Board& board, u_int8_t sq,
                        SideToMove attacker_side);

std::optional<SmallestAttacker> find_smallest_attacker(const Board& board,
                                                       u_int8_t sq,
                                                       SideToMove attacker_side);

} // namespace chess