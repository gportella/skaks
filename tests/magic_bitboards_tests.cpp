#include "chess/board_arithmetic.hpp"
#include "chess/magic_bitboards.hpp"
#include "chess/ray_tables.hpp"
#include "chess/types.hpp"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>

namespace {

Bitboard rook_attacks_slow(int sq, Bitboard occ) {
  const auto& rays = chess::ROOK_RAYS[static_cast<std::size_t>(sq)];

  Bitboard north = rays.north;
  Bitboard blocker = north & occ;
  if (blocker) {
    int b = chess::lsb_index(blocker);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    north &= mask_inclusive;
  }

  Bitboard east = rays.east;
  Bitboard east_blockers = east & occ;
  if (east_blockers) {
    int b = chess::lsb_index(east_blockers);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    east &= mask_inclusive;
  }

  Bitboard south = rays.south;
  Bitboard south_blockers = south & occ;
  if (south_blockers) {
    int b = chess::msb_index(south_blockers);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    south &= mask_inclusive;
  }

  Bitboard west = rays.west;
  Bitboard west_blockers = west & occ;
  if (west_blockers) {
    int b = chess::msb_index(west_blockers);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    west &= mask_inclusive;
  }

  return north | south | east | west;
}

Bitboard bishop_attacks_slow(int sq, Bitboard occ) {
  const auto& rays = chess::BISHOP_RAYS[static_cast<std::size_t>(sq)];

  Bitboard northeast = rays.northeast;
  Bitboard ne_blocker = northeast & occ;
  if (ne_blocker) {
    int b = chess::lsb_index(ne_blocker);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    northeast &= mask_inclusive;
  }

  Bitboard southeast = rays.southeast;
  Bitboard se_blocker = southeast & occ;
  if (se_blocker) {
    int b = chess::msb_index(se_blocker);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    southeast &= mask_inclusive;
  }

  Bitboard northwest = rays.northwest;
  Bitboard nw_blocker = northwest & occ;
  if (nw_blocker) {
    int b = chess::lsb_index(nw_blocker);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    northwest &= mask_inclusive;
  }

  Bitboard southwest = rays.southwest;
  Bitboard sw_blocker = southwest & occ;
  if (sw_blocker) {
    int b = chess::msb_index(sw_blocker);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    southwest &= mask_inclusive;
  }

  return northeast | southeast | northwest | southwest;
}

} // namespace

TEST(MagicBitboardsTest, RookMatchesReference) {
  chess::init_magic_bitboards();
  std::mt19937_64 rng(0xBADC0FFEEULL);
  std::uniform_int_distribution<std::uint64_t> dist;

  for (int sq = 0; sq < 64; ++sq) {
    for (int i = 0; i < 128; ++i) {
      const Bitboard occ = dist(rng);
      EXPECT_EQ(chess::rook_attacks_magic(sq, occ), rook_attacks_slow(sq, occ));
    }
  }
}

TEST(MagicBitboardsTest, BishopMatchesReference) {
  chess::init_magic_bitboards();
  std::mt19937_64 rng(0xC0FFEE123ULL);
  std::uniform_int_distribution<std::uint64_t> dist;

  for (int sq = 0; sq < 64; ++sq) {
    for (int i = 0; i < 128; ++i) {
      const Bitboard occ = dist(rng);
      EXPECT_EQ(chess::bishop_attacks_magic(sq, occ),
                bishop_attacks_slow(sq, occ));
    }
  }
}
