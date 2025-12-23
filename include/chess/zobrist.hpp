#pragma once
#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"

#include <cstdint>

namespace chess {

constexpr int kEmptyIndex = static_cast<int>(OccupancyType::empty);
constexpr int kFirstPieceIndex = static_cast<int>(OccupancyType::wP);
constexpr int kLastPieceIndex = static_cast<int>(OccupancyType::bK);

static_assert(kEmptyIndex == 0, "Expect OccupancyType::empty == 0");
static_assert(kLastPieceIndex - kFirstPieceIndex + 1 == 12,
              "Expect 12 piece types");

inline int occ_to_zidx(OccupancyType o) {
  int v = static_cast<int>(o);
  if (v == kEmptyIndex)
    return -1;
  return v - kFirstPieceIndex; // wP=0 ... bK=11
}

struct Zobrist {
  std::uint64_t piece[12][64];    // indexed by occ_to_zidx(o), square
  std::uint64_t castle[16];       // indexed by your castling mask 0..15
  std::uint64_t enPassantFile[8]; // file 0..7, only when EP capture available
  std::uint64_t sideToMove;       // XOR when Black to move
};

std::uint64_t compute_position_key(const Board& b);
void update_key_for_move(Board& b, const Undo& undo, int castle_mask_after,
                         bool ep_hash_before, bool ep_hash_after);
void init_zobrist(std::uint64_t seed);
const Zobrist& zobrist_table();

} // namespace chess