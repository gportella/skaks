#pragma once
#include <cstdint>
#include <limits>
#include <string_view>

using Bitboard = std::uint64_t;

namespace chess {

enum class Piece { wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK };
constexpr std::size_t kPieceCount = static_cast<std::size_t>(Piece::bK) + 1;
enum class OccupancyType { empty, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK };
enum class PieceColor { White, Black, Both, None };
enum class SideToMove { White, Black };
enum class CastlingRights {
  NoCastling = 0,
  WhiteKingside = 1,
  WhiteQueenside = 2,
  BlackKingside = 4,
  BlackQueenside = 8
};

// clang-format off
enum class Square {
  A1, B1, C1, D1, E1, F1, G1, H1,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A8, B8, C8, D8, E8, F8, G8, H8
};
// clang-format on

constexpr Bitboard U64_MASK = std::numeric_limits<std::uint64_t>::max(); // 0xFFFFFFFFFFFFFFFF
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = FILE_A << 1;
constexpr Bitboard FILE_G = FILE_A << 6;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_3 = RANK_1 << 16;
constexpr Bitboard RANK_4 = RANK_1 << 24;
constexpr Bitboard RANK_5 = RANK_1 << 32;
constexpr Bitboard RANK_6 = RANK_1 << 40;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;
constexpr Bitboard NOT_FILE_A = ~FILE_A;
constexpr Bitboard NOT_FILE_H = ~FILE_H;
constexpr bool is_white(OccupancyType occ) {
  return static_cast<std::size_t>(occ) < static_cast<std::size_t>(OccupancyType::bP);
}

constexpr SideToMove flip_side(SideToMove side) {
  return side == SideToMove::White ? SideToMove::Black : SideToMove::White;
}

struct FenFields {
  std::string_view placement;
  std::string_view side_to_move;
  std::string_view castling;
  std::string_view en_passant;
  std::string_view halfmove_clock;
  std::string_view fullmove_number;
};

} // namespace chess