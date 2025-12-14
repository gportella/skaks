#include "chess/attack_masks.hpp"

#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/casteling.hpp"
#include "chess/defaults.hpp"
#include "chess/types.hpp"

#include <assert.h>
#include <string>
#include <string_view>

namespace chess {

// Helpers
inline int poplsb(Bitboard& bb) {
  // Returns index of least significant set bit and clears it
  int idx = lsb_index(bb);
  bb &= (bb - 1);
  return idx;
}

namespace {

inline void guard_black_queen_on_c2(const Board&, std::string_view) {}

inline void guard_suspicious_move(const Board&, int, int, std::string_view) {}

inline void trace_black_queen_to_f2_debug(const Board&, int, int, Bitboard) {}

} // namespace

// First we get the attack bitboard for the piece on sq, then we mask out
// friendly pieces

// Further down we got the emission of moves for each piece type

// King attacks
Bitboard king_attack_bm(Board board, u_int8_t sq, SideToMove sidetm) {
  auto friendly = board.occupancy[to_index(sidetm)];
  auto attacks = KING_ATTACKS[static_cast<std::size_t>(sq)];
  return attacks & ~friendly;
}

/**
 * Knight attacks bitmask
 * @param[in] board struct
 * @param[in] square of interest
 * @param[in] what side are we moving
 * @return a bitboard with possible locations to attack
 */
Bitboard knight_attack_bm(Board board, u_int8_t sq, SideToMove sidetm) {
  auto friendly = board.occupancy[to_index(sidetm)];
  auto attacks = KNIGHT_ATTACKS[static_cast<std::size_t>(sq)];
  return attacks & ~friendly;
}

/**
 * Rook attacks bitmask
 * @param[in] board struct
 * @param[in] square of interest
 * @param[in] what side are we moving
 * @return a bitboard with possible locations to attack
 */
Bitboard rook_attack_bm(Board board, u_int8_t sq, SideToMove sidetm) {
  auto occ = board.occupancy[to_index(PieceColor::Both)];
  auto friendly = board.occupancy[to_index(sidetm)];
  auto attacks = rook_attacks_from(sq, occ);
  return attacks & ~friendly;
}

/**
 * Bishop attacks bitmask
 * @param[in] board struct
 * @param[in] square of interest
 * @param[in] what side are we moving
 * @return a bitboard with possible locations to attack
 */
Bitboard bishop_attack_bm(Board board, u_int8_t sq, SideToMove sidetm) {
  auto occ = board.occupancy[to_index(PieceColor::Both)];
  auto friendly = board.occupancy[to_index(sidetm)];
  auto attacks = bishop_attacks_from(sq, occ);
  return attacks & ~friendly;
}

/**
 * Queen attacks bitmask
 * @param[in] board struct
 * @param[in] square of interest
 * @param[in] what side are we moving
 * @return a bitboard with possible locations to attack
 */
Bitboard queen_attack_bm(Board board, u_int8_t sq, SideToMove sidetm) {
  return rook_attack_bm(board, sq, sidetm) | bishop_attack_bm(board, sq, sidetm);
}

/**
 * Generate pawn move masks for the given side to move
 * @param[in] board struct
 * @param[in] side to move
 * @return PawnMasks struct with bitboards for pawn moves
 */
PawnMasks gen_pawn_masks(const Board& board, SideToMove stm) {

  const Bitboard white = board.occupancy[to_index(SideToMove::White)];
  const Bitboard black = board.occupancy[to_index(SideToMove::Black)];
  const Bitboard occ = white | black;
  const Bitboard empty = ~occ;
  const Bitboard ep = board.ep_square;

  PawnMasks pm{};
  if (stm == SideToMove::White) {
    const Bitboard pawns = board.pieces_bb[static_cast<std::size_t>(Piece::wP)];

    // Single pushes
    Bitboard single = (pawns << 8) & empty;
    pm.promo_push = single & RANK_8;
    pm.single_push = single & ~RANK_8;

    // Double pushes (from rank 2, both squares empty)
    Bitboard dbl = ((pawns & RANK_2) << 8) & empty;
    pm.double_push = (dbl << 8) & empty;

    // Captures into enemy-occupied
    Bitboard capsL = ((pawns & NOT_FILE_A) << 7) & black;
    Bitboard capsR = ((pawns & NOT_FILE_H) << 9) & black;
    Bitboard caps = capsL | capsR;
    pm.promo_captures = caps & RANK_8;
    pm.captures = caps & ~RANK_8;

    // En passant captures (into ep square)
    pm.ep_captures = ((((pawns & NOT_FILE_A) << 7) | ((pawns & NOT_FILE_H) << 9)) & ep);

  } else {
    const Bitboard pawns = board.pieces_bb[static_cast<std::size_t>(Piece::bP)];

    // Single pushes
    Bitboard single = (pawns >> 8) & empty;
    pm.promo_push = single & RANK_1;
    pm.single_push = single & ~RANK_1;

    // Double pushes (from rank 7, both squares empty)
    Bitboard dbl = ((pawns & RANK_7) >> 8) & empty;
    pm.double_push = (dbl >> 8) & empty;

    // Captures into enemy-occupied
    Bitboard capsL = ((pawns & NOT_FILE_H) >> 7) & white; // from black pov "left"
    Bitboard capsR = ((pawns & NOT_FILE_A) >> 9) & white;
    Bitboard caps = capsL | capsR;
    pm.promo_captures = caps & RANK_1;
    pm.captures = caps & ~RANK_1;

    // En passant captures (into ep square)
    pm.ep_captures = ((((pawns & NOT_FILE_H) >> 7) | ((pawns & NOT_FILE_A) >> 9)) & ep);
  }

  return pm;
}

// Move generation functions

/**
 * White pawn move emission
 * @param[in] b Board struct
 * @param[in] pm PawnMasks struct
 * @param[out] out array to store encoded moves
 * @param[in,out] move_count current count of moves, will be updated
 */
void emit_white_pawn_moves(const Board& b, const PawnMasks& pm,
                           std::array<uint32_t, kMaxMovementCount>& out, uint16_t& move_count) {
  guard_black_queen_on_c2(b, "emit_white_pawn_moves");
  const Bitboard pawns = b.pieces_bb[static_cast<std::size_t>(Piece::wP)];

  // Single pushes (non-promo)
  {
    Bitboard dst = pm.single_push;
    while (dst) {
      int to = poplsb(dst);
      int from = to - 8;
      guard_suspicious_move(b, from, to, "white pawn single push");
      uint32_t m = encode_move(from, to,
                               /*piece=*/OccupancyType::wP,
                               /*captured=*/OccupancyType::empty,
                               /*promo=*/OccupancyType::empty,
                               /*flags=*/kFlagQuiet); // optional quiet flag

      assert(move_count < kMaxMovementCount);
      out[move_count++] = m;
    }
  }

  // Double pushes
  {
    Bitboard dst = pm.double_push;
    while (dst) {
      int to = poplsb(dst);
      int from = to - 16;
      guard_suspicious_move(b, from, to, "white pawn double push");
      uint32_t m = encode_move(from, to,
                               /*piece=*/OccupancyType::wP,
                               /*captured=*/OccupancyType::empty,
                               /*promo=*/OccupancyType::empty,
                               /*flags=*/kFlagDoublePush); // double-pawn-push flag
      assert(move_count < kMaxMovementCount);
      out[move_count++] = m;
    }
  }

  // Non-promo captures
  {
    Bitboard dst = pm.captures;
    while (dst) {
      int to = poplsb(dst);

      // Capture from file-right (origin to-7), guard destination not on file A
      if ((Bitboard(1) << to) & NOT_FILE_A) {
        int from = to - 7;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "white pawn capture");
          uint32_t m = encode_move(from, to,
                                   /*piece=*/OccupancyType::wP,
                                   /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                   /*promo=*/OccupancyType::empty,
                                   /*flags=*/0);
          assert(move_count < kMaxMovementCount);
          out[move_count++] = m;
        }
      }
      // Capture from file-left (origin to-9), guard destination not on file H
      if ((Bitboard(1) << to) & NOT_FILE_H) {
        int from = to - 9;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "white pawn capture");
          uint32_t m = encode_move(from, to,
                                   /*piece=*/OccupancyType::wP,
                                   /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                   /*promo=*/OccupancyType::empty,
                                   /*flags=*/0);
          assert(move_count < kMaxMovementCount);
          out[move_count++] = m;
        }
      }
    }
  }

  // Promotions by push (4 options)
  {
    Bitboard dst = pm.promo_push;
    while (dst) {
      int to = poplsb(dst);
      int from = to - 8;
      guard_suspicious_move(b, from, to, "white pawn promotion push");
      assert(move_count + 4 < kMaxMovementCount);
      out[move_count++] = encode_move(from, to, OccupancyType::wP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::wQ, 0);
      out[move_count++] = encode_move(from, to, OccupancyType::wP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::wR, 0);
      out[move_count++] = encode_move(from, to, OccupancyType::wP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::wB, 0);
      out[move_count++] = encode_move(from, to, OccupancyType::wP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::wN, 0);
    }
  }

  // Promotions by capture (4 options)
  {
    Bitboard dst = pm.promo_captures;
    while (dst) {
      int to = poplsb(dst);

      if ((Bitboard(1) << to) & NOT_FILE_A) {
        int from = to - 7;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "white pawn promotion capture");
          OccupancyType cap = b.pieces[static_cast<std::size_t>(to)];
          assert(move_count + 4 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wQ, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wR, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wB, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wN, 0);
        }
      }
      if ((Bitboard(1) << to) & NOT_FILE_H) {
        int from = to - 9;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "white pawn promotion capture");
          OccupancyType cap = b.pieces[static_cast<std::size_t>(to)];
          assert(move_count + 4 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wQ, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wR, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wB, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::wP, cap, OccupancyType::wN, 0);
        }
      }
    }
  }

  // En passant captures
  {
    Bitboard dst = pm.ep_captures;
    while (dst) {
      int to = poplsb(dst);

      if ((Bitboard(1) << to) & NOT_FILE_A) {
        int from = to - 7;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "white pawn en passant");
          assert(move_count + 1 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::wP,
                                          /*captured=*/OccupancyType::empty,
                                          /*promo=*/OccupancyType::empty,
                                          /*flags=*/kFlagEnPassant); // EP flag
        }
      }
      if ((Bitboard(1) << to) & NOT_FILE_H) {
        int from = to - 9;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "white pawn en passant");
          assert(move_count + 1 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::wP,
                                          /*captured=*/OccupancyType::empty,
                                          /*promo=*/OccupancyType::empty,
                                          /*flags=*/kFlagEnPassant); // EP flag
        }
      }
    }
  }
}

void emit_black_pawn_moves(const Board& b, const PawnMasks& pm,
                           std::array<uint32_t, kMaxMovementCount>& out, uint16_t& move_count) {
  guard_black_queen_on_c2(b, "emit_black_pawn_moves");
  const Bitboard pawns = b.pieces_bb[static_cast<std::size_t>(Piece::bP)];

  // Single pushes (non-promo)
  {
    Bitboard dst = pm.single_push;
    while (dst) {
      int to = poplsb(dst);
      int from = to + 8;
      guard_suspicious_move(b, from, to, "black pawn single push");
      uint32_t m = encode_move(from, to,
                               /*piece=*/OccupancyType::bP,
                               /*captured=*/OccupancyType::empty,
                               /*promo=*/OccupancyType::empty,
                               /*flags=*/kFlagQuiet); // optional quiet flag
      assert(move_count < kMaxMovementCount);
      out[move_count++] = m;
    }
  }

  // Double pushes
  {
    Bitboard dst = pm.double_push;
    while (dst) {
      int to = poplsb(dst);
      int from = to + 16;
      guard_suspicious_move(b, from, to, "black pawn double push");
      uint32_t m = encode_move(from, to,
                               /*piece=*/OccupancyType::bP,
                               /*captured=*/OccupancyType::empty,
                               /*promo=*/OccupancyType::empty,
                               /*flags=*/kFlagDoublePush); // double-pawn-push flag
      assert(move_count < kMaxMovementCount);
      out[move_count++] = m;
    }
  }

  // Non-promo captures
  {
    Bitboard dst = pm.captures;
    while (dst) {
      int to = poplsb(dst);

      // Capture from file-left (origin to+9), guard destination not on file H
      if ((Bitboard(1) << to) & NOT_FILE_H) {
        int from = to + 9;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "black pawn capture");
          uint32_t m = encode_move(from, to,
                                   /*piece=*/OccupancyType::bP,
                                   /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                   /*promo=*/OccupancyType::empty,
                                   /*flags=*/0);
          assert(move_count < kMaxMovementCount);
          out[move_count++] = m;
        }
      }
      // Capture from file-right (origin to+7), guard destination not on file A
      if ((Bitboard(1) << to) & NOT_FILE_A) {
        int from = to + 7;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "black pawn capture");
          uint32_t m = encode_move(from, to,
                                   /*piece=*/OccupancyType::bP,
                                   /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                   /*promo=*/OccupancyType::empty,
                                   /*flags=*/0);
          assert(move_count < kMaxMovementCount);
          out[move_count++] = m;
        }
      }
    }
  }

  // Promotions by push (4 options)
  {
    Bitboard dst = pm.promo_push;
    while (dst) {
      int to = poplsb(dst);
      int from = to + 8;
      guard_suspicious_move(b, from, to, "black pawn promotion push");
      assert(move_count + 4 < kMaxMovementCount);
      out[move_count++] = encode_move(from, to, OccupancyType::bP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::bQ, 0);
      out[move_count++] = encode_move(from, to, OccupancyType::bP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::bR, 0);
      out[move_count++] = encode_move(from, to, OccupancyType::bP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::bB, 0);
      out[move_count++] = encode_move(from, to, OccupancyType::bP, OccupancyType::empty,
                                      /*promo=*/OccupancyType::bN, 0);
    }
  }

  // Promotions by capture (4 options)
  {
    Bitboard dst = pm.promo_captures;
    while (dst) {
      int to = poplsb(dst);

      if ((Bitboard(1) << to) & NOT_FILE_H) {
        int from = to + 9;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "black pawn promotion capture");
          OccupancyType cap = b.pieces[static_cast<std::size_t>(to)];
          assert(move_count + 4 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bQ, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bR, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bB, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bN, 0);
        }
      }
      if ((Bitboard(1) << to) & NOT_FILE_A) {
        int from = to + 7;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "black pawn promotion capture");
          OccupancyType cap = b.pieces[static_cast<std::size_t>(to)];
          assert(move_count + 4 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bQ, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bR, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bB, 0);
          out[move_count++] = encode_move(from, to, OccupancyType::bP, cap, OccupancyType::bN, 0);
        }
      }
    }
  }

  // En passant captures
  {
    Bitboard dst = pm.ep_captures;
    while (dst) {
      int to = poplsb(dst);

      if ((Bitboard(1) << to) & NOT_FILE_H) {
        int from = to + 9;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "black pawn en passant");
          assert(move_count + 1 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::bP,
                                          /*captured=*/OccupancyType::empty,
                                          /*promo=*/OccupancyType::empty,
                                          /*flags=*/kFlagEnPassant); // EP flag
        }
      }
      if ((Bitboard(1) << to) & NOT_FILE_A) {
        int from = to + 7;
        if ((pawns >> from) & 1ULL) {
          guard_suspicious_move(b, from, to, "black pawn en passant");
          assert(move_count + 1 < kMaxMovementCount);
          out[move_count++] = encode_move(from, to, OccupancyType::bP,
                                          /*captured=*/OccupancyType::empty,
                                          /*promo=*/OccupancyType::empty,
                                          /*flags=*/kFlagEnPassant); // EP flag
        }
      }
    }
  }
}

/**
 * Knight move emission
 * @param[in] b Board struct
 * @param[in] stm Side to move
 * @param[out] out array to store encoded moves
 * @param[in,out] move_count current count of moves, will be updated
 */
void emit_knight_moves(const Board& b, SideToMove stm, std::array<uint32_t, kMaxMovementCount>& out,
                       uint16_t& move_count) {
  guard_black_queen_on_c2(b, "emit_knight_moves");
  const bool white = (stm == SideToMove::White);
  const Bitboard my_occ = b.occupancy[to_index(stm)];
  const Bitboard their_occ = b.occupancy[to_index(flip_side(stm))];
  const Bitboard empty = ~(my_occ | their_occ);

  const Bitboard knights = b.pieces_bb[static_cast<std::size_t>(white ? Piece::wN : Piece::bN)];

  Bitboard k = knights;
  while (k) {
    int from = poplsb(k);
    Bitboard atk = KNIGHT_ATTACKS[static_cast<std::size_t>(from)];

    // Quiet
    Bitboard q = atk & empty;
    while (q) {
      int to = poplsb(q);
      guard_suspicious_move(b, from, to, "knight quiet");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wN : OccupancyType::bN,
                                      /*captured=*/OccupancyType::empty,
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/kFlagQuiet); // optional quiet flag
    }

    // Captures
    Bitboard c = atk & their_occ;
    while (c) {
      int to = poplsb(c);
      guard_suspicious_move(b, from, to, "knight capture");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wN : OccupancyType::bN,
                                      /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/0);
    }
  }
}

/**
 * Bishop move emission
 * @param[in] b Board struct
 * @param[in] stm Side to move
 * @param[out] out array to store encoded moves
 * @param[in,out] move_count current count of moves, will be updated
 */
void emit_bishop_moves(const Board& b, SideToMove stm, std::array<uint32_t, kMaxMovementCount>& out,
                       uint16_t& move_count) {
  guard_black_queen_on_c2(b, "emit_bishop_moves");
  const bool white = (stm == SideToMove::White);
  const Bitboard my_occ = b.occupancy[to_index(stm)];
  const Bitboard their_occ = b.occupancy[to_index(flip_side(stm))];
  const Bitboard occ = b.occupancy[to_index(PieceColor::Both)];

  const Bitboard bishops = b.pieces_bb[static_cast<std::size_t>(white ? Piece::wB : Piece::bB)];

  Bitboard bs = bishops;
  while (bs) {
    int from = poplsb(bs);
    Bitboard atk = bishop_attacks_from(static_cast<u_int8_t>(from), occ);
    atk &= ~my_occ;

    // Quiet
    Bitboard q = atk & ~their_occ;
    while (q) {
      int to = poplsb(q);
      guard_suspicious_move(b, from, to, "bishop quiet");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wB : OccupancyType::bB,
                                      /*captured=*/OccupancyType::empty,
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/kFlagQuiet); // optional quiet flag
    }

    // Captures
    Bitboard c = atk & their_occ;
    while (c) {
      int to = poplsb(c);
      guard_suspicious_move(b, from, to, "bishop capture");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wB : OccupancyType::bB,
                                      /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/0);
    }
  }
}

/**
 * Rook move emission
 */
void emit_rook_moves(const Board& b, SideToMove stm, std::array<uint32_t, kMaxMovementCount>& out,
                     uint16_t& move_count) {
  guard_black_queen_on_c2(b, "emit_rook_moves");
  const bool white = (stm == SideToMove::White);
  const Bitboard their_occ = b.occupancy[to_index(flip_side(stm))];
  const Bitboard occ = b.occupancy[to_index(PieceColor::Both)];
  const Bitboard my_occ = b.occupancy[to_index(stm)];

  Bitboard rooks = b.pieces_bb[static_cast<std::size_t>(white ? Piece::wR : Piece::bR)];
  while (rooks) {
    int from = poplsb(rooks);
    Bitboard atk = rook_attacks_from(static_cast<u_int8_t>(from), occ);
    atk &= ~my_occ;

    Bitboard quiet = atk & ~their_occ;
    while (quiet) {
      int to = poplsb(quiet);
      guard_suspicious_move(b, from, to, "rook quiet");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wR : OccupancyType::bR,
                                      /*captured=*/OccupancyType::empty,
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/kFlagQuiet); // optional quiet flag
    }

    Bitboard captures = atk & their_occ;
    while (captures) {
      int to = poplsb(captures);
      guard_suspicious_move(b, from, to, "rook capture");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wR : OccupancyType::bR,
                                      /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/0);
    }
  }
}

/**
 * Queen move emission
 * @param[in] b Board struct
 * @param[in] stm Side to move
 * @param[out] out array to store encoded moves
 * @param[in,out] move_count current count of moves, will be updated
 */
void emit_queen_moves(const Board& b, SideToMove stm, std::array<uint32_t, kMaxMovementCount>& out,
                      uint16_t& move_count) {
  guard_black_queen_on_c2(b, "emit_queen_moves");
  const bool white = (stm == SideToMove::White);
  const Bitboard their_occ = b.occupancy[to_index(flip_side(stm))];
  const Bitboard occ = b.occupancy[to_index(PieceColor::Both)];
  const Bitboard my_occ = b.occupancy[to_index(stm)];
  const Bitboard queens = b.pieces_bb[static_cast<std::size_t>(white ? Piece::wQ : Piece::bQ)];
  Bitboard qs = queens;
  while (qs) {
    int from = poplsb(qs);
    Bitboard atk = rook_attacks_from(static_cast<u_int8_t>(from), occ) |
                   bishop_attacks_from(static_cast<u_int8_t>(from), occ);
    atk &= ~my_occ;

    Bitboard q = atk & ~their_occ;
    while (q) {
      int to = poplsb(q);
      guard_suspicious_move(b, from, to, "queen quiet");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wQ : OccupancyType::bQ,
                                      /*captured=*/OccupancyType::empty,
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/kFlagQuiet); // optional quiet flag
    }
    // Captures
    Bitboard c = atk & their_occ;
    while (c) {
      int to = poplsb(c);
      guard_suspicious_move(b, from, to, "queen capture");
      if (!white && to == 13) {
        trace_black_queen_to_f2_debug(b, from, to, atk);
      }
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wQ : OccupancyType::bQ,
                                      /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/0);
    }
  }
}

/**
 * King move emission
 */
void emit_king_moves(const Board& b, SideToMove stm, std::array<uint32_t, kMaxMovementCount>& out,
                     uint16_t& move_count) {
  guard_black_queen_on_c2(b, "emit_king_moves");
  const bool white = (stm == SideToMove::White);
  const Bitboard my_occ = b.occupancy[to_index(stm)];
  const Bitboard their_occ = b.occupancy[to_index(flip_side(stm))];
  const auto& castle_cfg = kCastlingSideConfigs[to_index(stm)];

  Bitboard kings = b.pieces_bb[static_cast<std::size_t>(white ? Piece::wK : Piece::bK)];
  while (kings) {
    int from = poplsb(kings);
    Bitboard atk = KING_ATTACKS[static_cast<std::size_t>(from)] & ~my_occ;

    Bitboard quiet = atk & ~their_occ;
    while (quiet) {
      int to = poplsb(quiet);
      guard_suspicious_move(b, from, to, "king quiet");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wK : OccupancyType::bK,
                                      /*captured=*/OccupancyType::empty,
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/kFlagQuiet); // optional quiet flag
    }

    Bitboard captures = atk & their_occ;
    while (captures) {
      int to = poplsb(captures);
      guard_suspicious_move(b, from, to, "king capture");
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wK : OccupancyType::bK,
                                      /*captured=*/b.pieces[static_cast<std::size_t>(to)],
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/0);
    }

    CastlingRights castle_rights = king_castle_rights(b, stm);
    // Kingside castling --> remember to also move the rook in make_move()
    if (castle_rights == (white ? WK : BK)) {
      int to = static_cast<int>(castle_cfg.king_kingside_target);
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wK : OccupancyType::bK,
                                      /*captured=*/OccupancyType::empty,
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/kFlagCastle); // kingside castle flag
    }
    // Queenside castling --> remember to also move the rook in make_move()
    if (castle_rights == (white ? WQ : BQ)) {
      int to = static_cast<int>(castle_cfg.king_queenside_target);
      assert(move_count < kMaxMovementCount);
      out[move_count++] = encode_move(from, to,
                                      /*piece=*/white ? OccupancyType::wK : OccupancyType::bK,
                                      /*captured=*/OccupancyType::empty,
                                      /*promo=*/OccupancyType::empty,
                                      /*flags=*/kFlagCastleLong); // queenside castle flag
    }
  }
}

bool is_square_attacked(const Board& board, u_int8_t sq, SideToMove attacker_side) {
  // Check for pawn attacks
  const Bitboard target = Bitboard(1) << sq;
  Bitboard pawn_attackers;
  if (attacker_side == SideToMove::White) {
    pawn_attackers = ((target & NOT_FILE_H) >> 7) | ((target & NOT_FILE_A) >> 9);
    pawn_attackers &= board.pieces_bb[static_cast<std::size_t>(Piece::wP)];
  } else {
    pawn_attackers = ((target & NOT_FILE_A) << 7) | ((target & NOT_FILE_H) << 9);
    pawn_attackers &= board.pieces_bb[static_cast<std::size_t>(Piece::bP)];
  }
  if (pawn_attackers)
    return true;

  // Check for knight attacks
  Bitboard knight_attackers = knight_attack_bm(board, sq, flip_side(attacker_side));
  knight_attackers &= board.pieces_bb[static_cast<std::size_t>(
      attacker_side == SideToMove::White ? Piece::wN : Piece::bN)];
  if (knight_attackers)
    return true;

  // Check for bishop/queen attacks
  Bitboard bishop_attackers = bishop_attack_bm(board, sq, flip_side(attacker_side));
  bishop_attackers &= board.pieces_bb[static_cast<std::size_t>(
                          attacker_side == SideToMove::White ? Piece::wB : Piece::bB)] |
                      board.pieces_bb[static_cast<std::size_t>(
                          attacker_side == SideToMove::White ? Piece::wQ : Piece::bQ)];
  if (bishop_attackers)
    return true;

  // Check for rook/queen attacks
  Bitboard rook_attackers = rook_attack_bm(board, sq, flip_side(attacker_side));
  rook_attackers &= board.pieces_bb[static_cast<std::size_t>(
                        attacker_side == SideToMove::White ? Piece::wR : Piece::bR)] |
                    board.pieces_bb[static_cast<std::size_t>(
                        attacker_side == SideToMove::White ? Piece::wQ : Piece::bQ)];
  if (rook_attackers)
    return true;
  return false;
}

CastlingRights king_castle_rights(const Board& board, SideToMove stm) {
  const SideToMove enemy = flip_side(stm);
  const auto& cfg = kCastlingSideConfigs[to_index(stm)];

  const auto mask_from = [](const auto& squares) {
    Bitboard mask = 0;
    for (Square sq : squares) {
      mask |= bit_mask(sq);
    }
    return mask;
  };

  const Bitboard occ = board.occupancy[to_index(PieceColor::Both)];
  const bool clear_king_path = (occ & mask_from(cfg.king_path)) == 0;
  const bool clear_queen_path = (occ & mask_from(cfg.queen_path)) == 0;

  const auto squares_are_safe = [&](const auto& squares) {
    for (Square sq : squares) {
      if (is_square_attacked(board, static_cast<u_int8_t>(sq), enemy)) {
        return false;
      }
    }
    return true;
  };

  const auto& kings = board.king_list[to_index(stm)];
  bool king_on_start = false;
  for (std::uint8_t i = 0; i < kings.count; ++i) {
    if (kings.squares[i] == cfg.king_start) {
      king_on_start = true;
      break;
    }
  }

  if (!king_on_start) {
    return CastlingRights::NoCastling;
  }

  const auto& rooks = board.rook_list[to_index(stm)];
  const auto has_rook_at = [&](Square target) {
    for (std::uint8_t i = 0; i < rooks.count; ++i) {
      if (rooks.squares[i] == target) {
        return true;
      }
    }
    return false;
  };

  const bool king_square_safe =
      !is_square_attacked(board, static_cast<u_int8_t>(cfg.king_start), enemy);

  CastlingRights rights = CastlingRights::NoCastling;

  if (king_square_safe && clear_king_path && squares_are_safe(cfg.king_safe) &&
      has_rook_at(cfg.rook_kingside_start)) {
    rights |= cfg.king_flag;
  }

  if (king_square_safe && clear_queen_path && squares_are_safe(cfg.queen_safe) &&
      has_rook_at(cfg.rook_queenside_start)) {
    rights |= cfg.queen_flag;
  }

  return rights;
}

int mvv_lva_score(OccupancyType captured, OccupancyType piece) {
  static const int scores[13] = {
      0,   100, 320, 330, 500, 900,  20000, // empty, P, N, B, R, Q, K
      100, 320, 330, 500, 900, 20000        // empty, p, n, b, r, q, k
  };
  return scores[static_cast<std::size_t>(captured)] * 10 - scores[static_cast<std::size_t>(piece)];
}

void sort_moves(std::array<uint32_t, kMaxMovementCount>& moves, uint16_t move_count) {
  // Simple bubble sort based on MVV-LVA heuristic
  for (uint16_t i = 0; i < move_count; ++i) {
    for (uint16_t j = 0; j < move_count - i - 1; ++j) {
      uint32_t m1 = moves[j];
      uint32_t m2 = moves[j + 1];

      OccupancyType cap1 = static_cast<OccupancyType>(move_captured(m1));
      OccupancyType cap2 = static_cast<OccupancyType>(move_captured(m2));
      OccupancyType piece1 = static_cast<OccupancyType>(move_piece(m1));
      OccupancyType piece2 = static_cast<OccupancyType>(move_piece(m2));

      int score1 = mvv_lva_score(cap1, piece1);
      int score2 = mvv_lva_score(cap2, piece2);

      if (score1 < score2) {
        std::swap(moves[j], moves[j + 1]);
      }
    }
  }
}

void emit_all_moves(const Board& board, SideToMove stm,
                    std::array<uint32_t, kMaxMovementCount>& out, uint16_t& move_count) {
  guard_black_queen_on_c2(board, "emit_all_moves START");
  PawnMasks pm = gen_pawn_masks(board, stm);
  if (stm == SideToMove::White) {
    emit_white_pawn_moves(board, pm, out, move_count);
  } else {
    emit_black_pawn_moves(board, pm, out, move_count);
  }
  emit_knight_moves(board, stm, out, move_count);
  emit_bishop_moves(board, stm, out, move_count);
  emit_rook_moves(board, stm, out, move_count);
  emit_queen_moves(board, stm, out, move_count);
  emit_king_moves(board, stm, out, move_count);
  sort_moves(out, move_count);
  guard_black_queen_on_c2(board, "emit_all_moves END");
}

} // namespace chess