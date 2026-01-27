// SPDX-License-Identifier: GPL-3.0-or-later
#include "chess/zobrist.hpp"

#include "chess/board.hpp"
#include "chess/casteling.hpp"
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

/// Initializes Zobrist hashing tables using a SplitMix64 RNG seeded by `seed`.
///
/// Fills piece, castling, en passant, and side-to-move keys exactly once;
/// subsequent calls return immediately if already initialized.
///
/// @param seed 64-bit seed for deterministic key generation (default is a fixed
/// constant).
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
/// \brief Computes the Zobrist hash key for a given board position.
///
/// Iterates over all squares to XOR piece-square keys, then incorporates
/// side-to-move, castling rights, and en passant file (only if a capture
/// is available) to produce the final position key.
///
/// \param b The board state to hash.
/// \return The 64-bit Zobrist key representing the position.
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

/// Updates the board's Zobrist position key after applying a move, using
/// undo data and the resulting game state.
///
/// This function:
/// - Removes any previous en passant hash contribution (if applicable).
/// - Removes the previous castling rights hash.
/// - Toggles the moving piece from its origin square.
/// - Toggles any captured piece from its capture square.
/// - Adds the piece now occupying the destination square.
/// - Handles rook movement for castling.
/// - Adds the updated castling rights hash.
/// - Adds the updated en passant hash (if applicable).
/// - Toggles the side-to-move hash.
///
/// @param b The board being updated; its position_key is modified.
/// @param undo Snapshot of pre-move state (pieces, rights, squares).
/// @param castle_mask_after Castling rights mask after the move.
/// @param ep_hash_before Whether the pre-move en passant file should be hashed.
/// @param ep_hash_after Whether the post-move en passant file should be hashed.
void update_key_for_move(Board& b, const Undo& undo, int castle_mask_after,
                         bool ep_hash_before, bool ep_hash_after) {
  const Zobrist& zob = zobrist_table();
  std::uint64_t key = undo.position_key_before;

  if (ep_hash_before && undo.en_passant_before >= 0) {
    key ^= zob.enPassantFile[file_of(undo.en_passant_before)];
  }

  key ^= zob.castle[to_mask(undo.castling_rights_before)];

  const auto from_idx = to_index(undo.from);
  const auto captured_idx_sq = to_index(undo.captured_sq);
  const auto to_idx = to_index(undo.to);

  const int moving_idx = occ_to_zidx(undo.moving_pc);
  if (moving_idx >= 0) {
    key ^= zob.piece[moving_idx][from_idx];
  }

  if (undo.captured_pc != OccupancyType::empty) {
    const int captured_idx = occ_to_zidx(undo.captured_pc);
    if (captured_idx >= 0) {
      key ^= zob.piece[captured_idx][captured_idx_sq];
    }
  }

  const OccupancyType placed_piece = b.pieces[to_idx];
  if (placed_piece != OccupancyType::empty) {
    const int placed_idx = occ_to_zidx(placed_piece);
    if (placed_idx >= 0) {
      key ^= zob.piece[placed_idx][to_idx];
    }
  }

  if (undo.was_castling) {
    const bool white_king = (undo.moving_pc == OccupancyType::wK);
    const auto& cfg = kCastlingSideConfigs[to_index(
        white_king ? SideToMove::White : SideToMove::Black)];
    const bool king_side = flag_is_castle(undo.flags);
    const Square rook_from =
        king_side ? cfg.rook_kingside_start : cfg.rook_queenside_start;
    const Square rook_to =
        king_side ? cfg.rook_kingside_target : cfg.rook_queenside_target;
    const OccupancyType rook_pc =
        white_king ? OccupancyType::wR : OccupancyType::bR;
    const int rook_idx = occ_to_zidx(rook_pc);
    key ^= zob.piece[rook_idx][to_index(rook_from)];
    key ^= zob.piece[rook_idx][to_index(rook_to)];
  }

  key ^= zob.castle[castle_mask_after];

  if (ep_hash_after && b.en_passant >= 0) {
    key ^= zob.enPassantFile[file_of(b.en_passant)];
  }

  key ^= zob.sideToMove;

  b.position_key = key;
}

} // namespace chess