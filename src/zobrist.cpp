
#include "chess/zobrist.hpp"

#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

namespace chess {
static Zobrist Z;
bool g_inited = false;

// SplitMix64 PRNG
static inline std::uint64_t splitmix64(std::uint64_t& s) {
  s += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void init_zobrist(std::uint64_t seed = 0xA5A5A5A5A5A5A5A5ULL) {
  if (g_inited)
    return; // or assert(!g_inited) if reinit is a logic error
  std::uint64_t s = seed;
  for (int p = 0; p < 12; ++p)
    for (int sq = 0; sq < 64; ++sq)
      Z.piece[p][to_index(sq)] = splitmix64(s);

  for (int i = 0; i < 16; ++i)
    Z.castle[i] = splitmix64(s);
  for (int f = 0; f < 8; ++f)
    Z.enPassantFile[f] = splitmix64(s);
  Z.sideToMove = splitmix64(s);
  g_inited = true;
}
std::uint64_t compute_position_key(const Board& b) {
  std::uint64_t key = 0ULL;

  // Pieces
  for (int sq = 0; sq < 64; ++sq) {
    int idx = occ_to_zidx(b.pieces[to_index(sq)]);
    if (idx >= 0)
      key ^= Z.piece[idx][to_index(sq)];
  }

  // Side to move
  if (b.side_to_move == SideToMove::Black)
    key ^= Z.sideToMove;

  // Castling rights
  key ^= Z.castle[to_mask(b.castling_rights)];

  // En passant file (only if an EP capture is actually available)
  if (b.en_passant >= 0 && ep_capture_available(b)) {
    key ^= Z.enPassantFile[file_of(b.en_passant)];
  }

  return key;
}

const Zobrist& zobrist_table() {
  if (!g_inited)
    init_zobrist(); // lazy init fallback
  return Z;
}

void update_key_for_move(Board& b, const Move& m) {
  std::uint64_t key = b.position_key;

  // Remove previous EP from hash if it was hashed
  if (b.en_passant >= 0 && ep_capture_available(b)) {
    key ^= Z.enPassantFile[file_of(b.en_passant)];
  }

  // Moving piece out of 'from'
  key ^= Z.piece[occ_to_zidx(m.moving_pc)][m.from];

  // Captures
  if (flag_is_ep(m.flags)) {
    int capSq = (b.side_to_move == SideToMove::White) ? (m.to - 8) : (m.to + 8);
    OccupancyType cap = b.pieces[to_index(capSq)];
    key ^= Z.piece[occ_to_zidx(cap)][to_index(capSq)];
  } else if (m.captured_pc != OccupancyType::empty) {
    key ^= Z.piece[occ_to_zidx(m.captured_pc)][to_index(m.to)];
  }

  // Place the moving (or promoted) piece on 'to'
  if (m.promo_pc != OccupancyType::empty) {
    key ^= Z.piece[occ_to_zidx(m.promo_pc)][to_index(m.to)];
  } else {
    key ^= Z.piece[occ_to_zidx(m.moving_pc)][to_index(m.to)];
  }

  // Castling rook movement (XOR rook from/to)
  if (flag_is_castle(m.flags)) {
    // Before mutating rights:
    const int oldCR = to_mask(b.castling_rights);

    // ...apply board changes that don't affect rights yet...

    // Update rights and hash them
    const int newCR = update_castling_rights(b, m);
    if (oldCR != newCR) {
      b.position_key ^= Z.castle[oldCR];
      b.position_key ^= Z.castle[newCR];
    }
  }

  // Side to move
  key ^= Z.sideToMove;

  // Castling rights: XOR old and new masks if changed
  int oldCR = to_mask(b.castling_rights);
  // Update b.castling_rights according to the move before computing newCR
  int newCR = to_mask(b.castling_rights);
  if (oldCR != newCR) {
    key ^= Z.castle[oldCR];
    key ^= Z.castle[newCR];
  }

  // En passant: clear, then set if double push; hash only if capture available
  b.ep_square = 0;
  b.en_passant = -1;
  if (flag_is_double_push(m.flags)) {
    const int mid = (m.from + m.to) / 2;
    b.en_passant = mid;
    b.ep_square = 1ULL << mid;
    if (ep_capture_available(b)) {
      key ^= Z.enPassantFile[file_of(mid)];
    }
  }

  b.position_key = key;
}

} // namespace chess