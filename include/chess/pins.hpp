#pragma once
#include "chess/attack_masks.hpp"
#include "chess/types.hpp"

#include <array>

namespace chess {

enum class MoveDirection { None, File, Rank, NE, NW, SE, SW };
struct PinnedBitBoardDirections {
  Bitboard mask;
  MoveDirection direction;
};
struct PinnedMapByPiece {
  std::array<PinnedBitBoardDirections, 64> bishop_pins;
  std::array<PinnedBitBoardDirections, 64> rook_pins;
  std::array<PinnedBitBoardDirections, 64> queen_pins;
  std::array<PinnedBitBoardDirections, 64> knight_pins;
  std::array<PinnedBitBoardDirections, 64> pawn_pins;
};

enum class Edge { Forward, Backward };

struct OrthoDir {
  Bitboard chess::RookRays::* ray; // king → direction
  Edge first_blocker_edge;         // do we walk “up” (lsb) or “down” (msb)?
  MoveDirection dir;
};

struct DiagonalDir {
  Bitboard chess::BishopRays::* ray; // king → direction
  Edge first_blocker_edge;           // do we walk “up” (lsb) or “down” (msb)?
  MoveDirection dir;
};

chess::PinnedMapByPiece build_pinned_map(const Board& board, SideToMove stm);

} // namespace chess