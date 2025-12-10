#pragma once

#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <array>
#include <ostream>
#include <string>
#include <string_view>

namespace chess {

inline uint32_t encode_move(int from, int to, OccupancyType moving_piece,
                            OccupancyType captured_piece, OccupancyType promo_piece,
                            uint32_t flags) {
  return (uint32_t(from) & 0x3F) | ((uint32_t(to) & 0x3F) << 6) |
         ((uint32_t(moving_piece) & 0x0F) << 12) | ((uint32_t(captured_piece) & 0x0F) << 16) |
         ((uint32_t(promo_piece) & 0x0F) << 20) |
         (flags & 0xFF000000u); // upper 8 bits reserved for flags
}

inline int move_from(uint32_t m) {
  return m & 0x3F;
}
inline int move_to(uint32_t m) {
  return (m >> 6) & 0x3F;
}
inline int move_piece(uint32_t m) {
  return (m >> 12) & 0x0F;
}
inline int move_captured(uint32_t m) {
  return (m >> 16) & 0x0F;
}
inline int move_promo(uint32_t m) {
  return (m >> 20) & 0x0F;
}
inline bool move_is_ep(uint32_t m) {
  return (m >> 24) & 1u;
}
inline bool move_is_castle(uint32_t m) {
  return (m >> 25) & 1u;
}
inline bool move_is_double(uint32_t m) {
  return (m >> 26) & 1u;
}
inline bool move_is_quiet(uint32_t m) {
  return (m >> 27) & 1u;
}

struct BishopRays {
  Bitboard northeast{};
  Bitboard northwest{};
  Bitboard southeast{};
  Bitboard southwest{};
};
struct RookRays {
  Bitboard north{};
  Bitboard south{};
  Bitboard east{};
  Bitboard west{};
};

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

inline const std::array<Bitboard, 64> KING_ATTACKS = build_king_attack_patterns();

constexpr std::array<RookRays, 64> build_rook_rays() {
  std::array<RookRays, 64> rays{};
  for (std::size_t sq = 0; sq < 64; ++sq) {
    auto& ray = rays[sq];
    const std::size_t rank = sq / 8;
    const std::size_t file = sq % 8;

    for (std::size_t r = rank + 1; r < 8; ++r) {
      const std::size_t target = (r * 8) + file;
      ray.north |= Bitboard(1) << target;
    }
    for (int r = static_cast<int>(rank) - 1; r >= 0; --r) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + file;
      ray.south |= Bitboard(1) << target;
    }
    for (std::size_t f = file + 1; f < 8; ++f) {
      const std::size_t target = (rank * 8) + f;
      ray.east |= Bitboard(1) << target;
    }
    for (int f = static_cast<int>(file) - 1; f >= 0; --f) {
      const std::size_t target = (rank * 8) + static_cast<std::size_t>(f);
      ray.west |= Bitboard(1) << target;
    }
  }
  return rays;
}

inline constexpr std::array<RookRays, 64> ROOK_RAYS = build_rook_rays();

/**
 * Calculate rooks attacks from a given square
 * We might want to cache this one if possible
 * @param[in] sq The square index in the 64 mailbox
 * @param[in] b_occ The occupancy of the board
 * return A bitboard with the possible locations to attack

**/
inline Bitboard rook_attacks_from(u_int8_t sq, Bitboard b_occupancy) {

  const auto& rays = ROOK_RAYS[sq];
  //  north
  auto north = rays.north;
  auto blocker = north & b_occupancy;
  if (blocker) {
    int b = lsb_index(blocker);
    const Bitboard mask_inclusive = (b == 63) ? U64_MASK : ((Bitboard(1) << (b + 1)) - 1);
    north &= mask_inclusive;
  }
  // East
  Bitboard east = rays.east;
  Bitboard east_blockers = east & b_occupancy;
  if (east_blockers) {
    int b = lsb_index(east_blockers); // nearest east blocker
    const Bitboard mask_inclusive = (b == 63) ? U64_MASK : ((Bitboard(1) << (b + 1)) - 1);
    east &= mask_inclusive;
  }

  // South
  Bitboard south = rays.south;
  Bitboard south_blockers = south & b_occupancy;
  if (south_blockers) {
    int b = msb_index(south_blockers);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    south &= mask_inclusive;
  }
  // West
  Bitboard west = rays.west;
  Bitboard west_blockers = west & b_occupancy;
  if (west_blockers) {
    int b = msb_index(west_blockers); // nearest west blocker
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    west &= mask_inclusive;
  }

  return north | south | east | west;
}

/**
 * Generate rook attack based removing friedly occupancy
 * @params[in] board struct
 * @params[in] square of interest
 * @params[in] what side are we moving
 * @return a bitboard with possible locations to attack
 *
 */

constexpr std::array<BishopRays, 64> build_bishop_rays() {
  std::array<BishopRays, 64> rays{};
  for (std::size_t sq = 0; sq < 64; ++sq) {
    auto& ray = rays[sq];
    const std::size_t rank = sq / 8;
    const std::size_t file = sq % 8;

    // Northeast (indices increase)
    for (std::size_t r = rank + 1, f = file + 1; r < 8 && f < 8; ++r, ++f) {
      const std::size_t target = (r * 8) + f;
      ray.northeast |= Bitboard(1) << target;
    }
    // Northwest (indices increase)
    for (std::size_t r = rank + 1, f = file; r < 8 && f > 0; ++r, --f) {
      const std::size_t target = (r * 8) + (f - 1);
      ray.northwest |= Bitboard(1) << target;
    }
    // Southeast (indices decrease)
    for (int r = static_cast<int>(rank) - 1, f = static_cast<int>(file) + 1; r >= 0 && f < 8;
         --r, ++f) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + static_cast<std::size_t>(f);
      ray.southeast |= Bitboard(1) << target;
    }
    // Southwest (indices decrease)
    for (int r = static_cast<int>(rank) - 1, f = static_cast<int>(file) - 1; r >= 0 && f >= 0;
         --r, --f) {
      const std::size_t target = (static_cast<std::size_t>(r) * 8) + static_cast<std::size_t>(f);
      ray.southwest |= Bitboard(1) << target;
    }
  }
  return rays;
}

inline constexpr std::array<BishopRays, 64> BISHOP_RAYS = build_bishop_rays();

inline Bitboard bishop_attacks_from(u_int8_t sq, Bitboard b_occupancy) {

  const auto& rays = BISHOP_RAYS[sq];
  //  northeast
  auto northeast = rays.northeast;
  auto ne_blocker = northeast & b_occupancy;
  if (ne_blocker) {
    int b = lsb_index(ne_blocker);
    const Bitboard mask_inclusive = (b == 63) ? U64_MASK : ((Bitboard(1) << (b + 1)) - 1);
    northeast &= mask_inclusive;
  }
  auto southeast = rays.southeast;
  auto se_blocker = southeast & b_occupancy;
  if (se_blocker) {
    int b = msb_index(se_blocker);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    southeast &= mask_inclusive;
  }
  auto northwest = rays.northwest;
  auto nw_blocker = northwest & b_occupancy;
  if (nw_blocker) {
    int b = lsb_index(nw_blocker);
    const Bitboard mask_inclusive = (b == 63) ? U64_MASK : ((Bitboard(1) << (b + 1)) - 1);
    northwest &= mask_inclusive;
  }
  auto southwest = rays.southwest;
  auto sw_blocker = southwest & b_occupancy;
  if (sw_blocker) {
    int b = msb_index(sw_blocker);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    southwest &= mask_inclusive;
  }

  return northeast | southeast | northwest | southwest;
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

inline PawnMoves gen_white_pawn_moves(Bitboard wp, Bitboard occ, Bitboard black_occ, Bitboard ep) {
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

  Bitboard ep_caps = ((((wp & NOT_FILE_A) << 7) | ((wp & NOT_FILE_H) << 9)) & ep);

  return {nonpromo_push, doubles, nonpromo_caps, promo_push, promo_caps, ep_caps};
}
struct PawnMasks {
  Bitboard single_push;    // non-promo single pushes
  Bitboard double_push;    // double pawn pushes
  Bitboard captures;       // non-promo captures
  Bitboard promo_push;     // single pushes to promotion rank
  Bitboard promo_captures; // captures to promotion rank
  Bitboard ep_captures;    // en passant captures
};

PawnMasks gen_pawn_masks(const Board& board, SideToMove stm);
Bitboard knight_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard bishop_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard rook_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard queen_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);
Bitboard king_attack_bm(Board board, u_int8_t sq, SideToMove sidetm);

void emit_white_pawn_moves(const Board& board, const PawnMasks& masks,
                           std::array<uint32_t, kMaxMovementCount>& out, std::uint16_t& move_count);
void emit_black_pawn_moves(const Board& board, const PawnMasks& masks,
                           std::array<uint32_t, kMaxMovementCount>& out, std::uint16_t& move_count);

void emit_knight_moves(const Board& board, SideToMove stm,
                       std::array<uint32_t, kMaxMovementCount>& out, std::uint16_t& move_count);
void emit_bishop_moves(const Board& board, SideToMove stm,
                       std::array<uint32_t, kMaxMovementCount>& out, std::uint16_t& move_count);
void emit_rook_moves(const Board& board, SideToMove stm,
                     std::array<uint32_t, kMaxMovementCount>& out, std::uint16_t& move_count);
void emit_queen_moves(const Board& board, SideToMove stm,
                      std::array<uint32_t, kMaxMovementCount>& out, std::uint16_t& move_count);
void emit_king_moves(const Board& board, SideToMove stm,
                     std::array<uint32_t, kMaxMovementCount>& out, std::uint16_t& move_count);

} // namespace chess