#include "chess/scoring_rules.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/defaults.hpp"
#include "chess/piece_values.hpp"
#include "chess/pins.hpp"
#include "chess/pst_tables.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <array>

namespace chess {

constexpr int CHECK_PENALTY = 100;     // penalty/bonus for king in check
constexpr int PAWN_SHIELD_BONUS = 25;  // small bonus per nearby pawn to own king
constexpr int CASTLING_BONUS = 50;     // modest bonus if castled
constexpr int TEMPO_BONUS = 14;        // nudge for having the move
constexpr int THREAT_WEIGHT = 4;       // scales attacked-piece pressure
constexpr int PASSED_PAWN_BASE = 20;   // baseline reward for a passed pawn
constexpr int PASSED_PAWN_ADVANCE = 8; // scaled bonus per rank of advancement

// Helper: White-centric check queries
inline bool white_in_check(const Board& board) {
  return is_check(board, SideToMove::White);
}
inline bool black_in_check(const Board& board) {
  return is_check(board, SideToMove::Black);
}

// Material: White-centric (positive is good for White)
int evaluate_material(const Board& board) {
  int material_score = 0;
  for (const auto& occ : board.pieces) {
    material_score += piece_material_value(occ);
  }
  return material_score;
}

// Simple center presence (placeholder), White-centric and summed (no early
// return)
int evaluate_pawn_center_control(const Board& board) {
  int score = 0;
  for (auto sq : {to_index(Square::D4), to_index(Square::D5),
                  to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::wP)
      score += 10;
    else if (piece == OccupancyType::bP)
      score -= 10;
  }
  return score;
}

int evaluate_center_control(const Board& board) {
  int score = 0;
  for (auto sq : {to_index(Square::D4), to_index(Square::D5),
                  to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece != OccupancyType::empty) {
      const bool is_white_piece = static_cast<std::size_t>(piece) <
                                  static_cast<std::size_t>(OccupancyType::bP);
      score += is_white_piece ? 5 : -5;
    }
  }
  return score;
}

// King safety: White-centric, independent of side_to_move
int evaluate_king_safety(const Board& board) {
  int score = 0;

  // White pawn shield near White king
  {
    const int wking_sq = board.king_positions[to_index(PieceColor::White)];
    if (wking_sq != -1) {
      for (std::uint8_t i = 0;
           i < board.pawn_list[to_index(PieceColor::White)].count; ++i) {
        const auto pawn_sq =
            board.pawn_list[to_index(PieceColor::White)].squares[i];
        if (chebyshev_dist(to_index(pawn_sq),
                           static_cast<std::size_t>(wking_sq)) <= 2) {
          score += PAWN_SHIELD_BONUS;
        }
      }
    }
  }

  // Black pawn shield near Black king (subtract because it's good for Black)
  {
    const int bking_sq = board.king_positions[to_index(PieceColor::Black)];
    if (bking_sq != -1) {
      for (std::uint8_t i = 0;
           i < board.pawn_list[to_index(PieceColor::Black)].count; ++i) {
        const auto pawn_sq =
            board.pawn_list[to_index(PieceColor::Black)].squares[i];
        if (chebyshev_dist(to_index(pawn_sq),
                           static_cast<std::size_t>(bking_sq)) <= 2) {
          score -= PAWN_SHIELD_BONUS;
        }
      }
    }
  }

  // Castling status
  if (board.has_castled[to_index(PieceColor::White)])
    score += CASTLING_BONUS;
  if (board.has_castled[to_index(PieceColor::Black)])
    score -= CASTLING_BONUS;

  // In-check penalties/bonuses (moderate; mate handled in search)
  if (white_in_check(board))
    score -= CHECK_PENALTY;
  if (black_in_check(board))
    score += CHECK_PENALTY;

  return score;
}

// King mobility: evaluate both kings, White-centric, independent of side_to_move
int evaluate_king_mobility(const Board& board) {
  int score = 0;

  const int wking_sq = board.king_positions[to_index(PieceColor::White)];
  if (wking_sq != -1) {
    const int moves = popcount_bitboard(king_attack_bm(
        board, static_cast<u_int8_t>(wking_sq), SideToMove::White));
    score += MOBILITY_SCALING * moves;
  }

  const int bking_sq = board.king_positions[to_index(PieceColor::Black)];
  if (bking_sq != -1) {
    const int moves = popcount_bitboard(king_attack_bm(
        board, static_cast<u_int8_t>(bking_sq), SideToMove::Black));
    score -= MOBILITY_SCALING * moves;
  }

  return score;
}

// Attacking pieces: if a piece is attacked by the opponent, penalize for White
// pieces, reward for Black
int evaluate_attacking_pieces(const Board& board) {
  constexpr std::array<int, static_cast<std::size_t>(OccupancyType::bK) + 1>
      kThreatBase = {0, 12, 30, 30, 45, 180, 540, 12, 30, 30, 45, 180, 540};

  int attack_score = 0;

  for (std::size_t sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::empty)
      continue;

    const bool white_piece = static_cast<std::size_t>(piece) <
                             static_cast<std::size_t>(OccupancyType::bP);

    const int base_weight = kThreatBase[static_cast<std::size_t>(piece)];
    if (base_weight == 0)
      continue;

    // If this square is attacked by the opponent
    const SideToMove attacker =
        white_piece ? SideToMove::Black : SideToMove::White;
    if (!is_square_attacked(board, static_cast<u_int8_t>(sq), attacker))
      continue;

    // Scale by advancement (optional heuristic)
    const int rank = rank_of(static_cast<int>(sq));
    const int distance_from_home = white_piece ? rank : (7 - rank);
    const int positional_scale = 1 + distance_from_home / 2;
    const int threat_value = base_weight * positional_scale;

    attack_score += white_piece ? -threat_value : +threat_value;
  }

  return attack_score;
}

int compute_threat_pressure(const Board& board, SideToMove attacker) {
  const bool attacker_is_white = attacker == SideToMove::White;
  int pressure = 0;

  for (int sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece == OccupancyType::empty)
      continue;

    const bool piece_is_white = is_white(piece);
    if (piece_is_white == attacker_is_white)
      continue;

    if (!is_square_attacked(board, static_cast<u_int8_t>(sq), attacker))
      continue;

    const int magnitude = piece_material_magnitude(piece);
    pressure += std::max(1, magnitude / 100);
  }

  return pressure;
}

// let's git it some kick
int evaluate_initiative(const Board& board) {
  const SideToMove us = board.side_to_move;
  const SideToMove them = flip_side(us);

  const int our_pressure = compute_threat_pressure(board, us);
  const int their_pressure = compute_threat_pressure(board, them);

  const int net_pressure = our_pressure - their_pressure;
  return TEMPO_BONUS + THREAT_WEIGHT * net_pressure;
}

int evaluate_pst(const Board& board) {

  int midgame_score = 0;
  int endgame_score = 0;
  int phase = 0;

  for (int sq = 0; sq < 64; ++sq) {
    const auto piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece == OccupancyType::empty)
      continue;

    const bool white_piece = is_white(piece);
    const int type_index = (static_cast<int>(piece) - 1) % 6;
    const int oriented_sq = white_piece ? sq : mirror_rank(sq);

    const int mg_entry = kMidgamePst[static_cast<std::size_t>(type_index)]
                                    [static_cast<std::size_t>(oriented_sq)];
    const int eg_entry = kEndgamePst[static_cast<std::size_t>(type_index)]
                                    [static_cast<std::size_t>(oriented_sq)];

    midgame_score += white_piece ? mg_entry : -mg_entry;
    endgame_score += white_piece ? eg_entry : -eg_entry;

    phase += kPstPhaseWeights[static_cast<std::size_t>(type_index)];
  }

  const int mg_phase = std::min(phase, kPstPhaseMax);
  const int eg_phase = kPstPhaseMax - mg_phase;
  const int tapered =
      (midgame_score * mg_phase + endgame_score * eg_phase) / kPstPhaseMax;
  return tapered;
}

bool is_passed_pawn(const Board& board, int sq, bool white) {
  const int file = file_of(sq);
  const int rank = rank_of(sq);
  const OccupancyType enemy_pawn = white ? OccupancyType::bP : OccupancyType::wP;
  const int dir = white ? 1 : -1;

  for (int r = rank + dir; white ? (r <= 7) : (r >= 0); r += dir) {
    for (int df = -1; df <= 1; ++df) {
      const int f = file + df;
      if (f < 0 || f > 7) {
        continue;
      }
      const int idx = r * 8 + f;
      if (board.pieces[static_cast<std::size_t>(idx)] == enemy_pawn) {
        return false;
      }
    }
  }
  return true;
}

int evaluate_passed_pawns(const Board& board) {
  int score = 0;
  for (int sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece != OccupancyType::wP && piece != OccupancyType::bP) {
      continue;
    }

    const bool white = (piece == OccupancyType::wP);
    if (!is_passed_pawn(board, sq, white)) {
      continue;
    }

    const int advance = white ? rank_of(sq) : (7 - rank_of(sq));
    const int bonus = PASSED_PAWN_BASE + PASSED_PAWN_ADVANCE * advance;
    score += white ? bonus : -bonus;
  }
  return score;
}

int evaluate_pins(const Board& board) {
  int score = 0;
  auto pin_white = build_pinned_map(board, SideToMove::White);
  auto pin_black = build_pinned_map(board, SideToMove::Black);
  auto apply_penalty = [&](const PinnedBitBoardDirections& entry,
                           int base_penalty, int mobility_penalty,
                           int side_sign) {
    if (entry.mask == 0) {
      return;
    }
    const int mobility = popcount_bitboard(entry.mask);
    score += side_sign * (base_penalty + mobility_penalty * mobility);
  };
  for (std::size_t sq = 0; sq < 64; ++sq) {
    const auto piece = board.pieces[sq];
    if (piece == OccupancyType::empty) {
      continue;
    }
    const bool is_white_piece = is_white(piece);
    const auto& pin_map = is_white_piece ? pin_white : pin_black;
    const int side_sign = is_white_piece ? -1 : 1;

    switch (piece) {
    case OccupancyType::wB:
    case OccupancyType::bB: {
      const auto entry = pin_map.bishop_pins[sq];
      apply_penalty(entry, 12, 2, side_sign);
      break;
    }
    case OccupancyType::wR:
    case OccupancyType::bR: {
      const auto entry = pin_map.rook_pins[sq];
      apply_penalty(entry, 6, 1, side_sign);
      break;
    }
    case OccupancyType::wQ:
    case OccupancyType::bQ:
      break;
    case OccupancyType::wN:
    case OccupancyType::bN: {
      const auto entry = pin_map.knight_pins[sq];
      apply_penalty(entry, 15, 0, side_sign);
      break;
    }
    case OccupancyType::wP:
    case OccupancyType::bP: {
      const auto entry = pin_map.pawn_pins[sq];
      const bool is_diagonal_pin = entry.direction == MoveDirection::NE ||
                                   entry.direction == MoveDirection::NW ||
                                   entry.direction == MoveDirection::SE ||
                                   entry.direction == MoveDirection::SW;
      const int base = is_diagonal_pin ? 10 : 6;
      apply_penalty(entry, base, 2, side_sign);
      break;
    }
    default:
      break;
    }
  }
  return score;
}

// In-lieu of a book move, for now

inline bool is_knight_good_dev(int sq, bool white) {
  if (white) {
    return (sq == to_index(Square::C3) || sq == to_index(Square::F3));
  }
  return (sq == to_index(Square::C6) || sq == to_index(Square::F6));
}

inline bool is_bishop_good_dev(int sq, bool white) {
  if (white) {
    return (sq == to_index(Square::C4) || sq == to_index(Square::F4));
  }
  return (sq == to_index(Square::C5) || sq == to_index(Square::F5));
}

int opening_phase(const Board& board) {
  int phase = 0;
  for (int sq = 0; sq < 64; ++sq) {
    const auto piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece == OccupancyType::empty) {
      continue;
    }
    const int type_index = (static_cast<int>(piece) - 1) % 6;
    phase += kPstPhaseWeights[static_cast<std::size_t>(type_index)];
  }
  const int mg_phase = std::min(phase, kPstPhaseMax);
  const int opening = std::max(0, 64 - mg_phase);
  return opening; // [0..64], larger at start, shrinks as material/phase declines
}

int evaluate_opening_principles(const Board& board) {
  const int open_w = opening_phase(board);
  if (open_w == 0) {
    return 0;
  }

  int score = 0;

  bool w_back_blocked = false;
  bool b_back_blocked = false;

  for (int sq = 0; sq < 64; ++sq) {
    const auto pc = board.pieces[static_cast<std::size_t>(sq)];
    if (pc == OccupancyType::empty) {
      continue;
    }

    switch (pc) {
    case OccupancyType::wN:
      if (is_knight_good_dev(sq, true)) {
        score += (KNIGHT_DEV_BONUS * open_w) / 64;
      }
      break;
    case OccupancyType::bN:
      if (is_knight_good_dev(sq, false)) {
        score -= (KNIGHT_DEV_BONUS * open_w) / 64;
      }
      break;
    case OccupancyType::wB:
      if (is_bishop_good_dev(sq, true)) {
        score += (BISHOP_DEV_BONUS * open_w) / 64;
      }
      break;
    case OccupancyType::bB:
      if (is_bishop_good_dev(sq, false)) {
        score -= (BISHOP_DEV_BONUS * open_w) / 64;
      }
      break;
    default:
      break;
    }

    // Back-rank blockers prevent connected rooks bonus
    if (rank_of(sq) == 0) {
      if (pc == OccupancyType::wN || pc == OccupancyType::wB ||
          pc == OccupancyType::wQ) {
        w_back_blocked = true;
      }
    } else if (rank_of(sq) == 7) {
      if (pc == OccupancyType::bN || pc == OccupancyType::bB ||
          pc == OccupancyType::bQ) {
        b_back_blocked = true;
      }
    }
  }

  if (!w_back_blocked) {
    score += (CONNECT_ROOKS_BONUS * open_w) / 64;
  }
  if (!b_back_blocked) {
    score -= (CONNECT_ROOKS_BONUS * open_w) / 64;
  }

  auto add_central_pawn = [&](Square sq, OccupancyType pc) {
    if (board.pieces[static_cast<std::size_t>(sq)] == pc) {
      score += (is_white(pc) ? +1 : -1) * ((CENTRAL_PAWN_BONUS * open_w) / 64);
    }
  };
  add_central_pawn(Square::E4, OccupancyType::wP);
  add_central_pawn(Square::D4, OccupancyType::wP);
  add_central_pawn(Square::E5, OccupancyType::bP);
  add_central_pawn(Square::D5, OccupancyType::bP);

  if (board.has_castled[to_index(PieceColor::White)]) {
    score += (CASTLE_URGENCY * open_w) / 64;
  } else if (open_w > 40) {
    score -= ((open_w - 40) * 2);
  }

  if (board.has_castled[to_index(PieceColor::Black)]) {
    score -= (CASTLE_URGENCY * open_w) / 64;
  } else if (open_w > 40) {
    score += ((open_w - 40) * 2);
  }

  auto count_white_minor_developed = [&]() -> int {
    int c = 0;
    for (int sq = 0; sq < 64; ++sq) {
      const auto pc = board.pieces[static_cast<std::size_t>(sq)];
      if (pc == OccupancyType::wN || pc == OccupancyType::wB) {
        if (rank_of(sq) > 0) {
          c += 1;
        }
      }
    }
    return c;
  };
  auto count_black_minor_developed = [&]() -> int {
    int c = 0;
    for (int sq = 0; sq < 64; ++sq) {
      const auto pc = board.pieces[static_cast<std::size_t>(sq)];
      if (pc == OccupancyType::bN || pc == OccupancyType::bB) {
        if (rank_of(sq) < 7) {
          c += 1;
        }
      }
    }
    return c;
  };

  const int w_minors = count_white_minor_developed();
  const int b_minors = count_black_minor_developed();
  const bool wq_moved = board.pieces[to_index(Square::D1)] != OccupancyType::wQ;
  const bool bq_moved = board.pieces[to_index(Square::D8)] != OccupancyType::bQ;

  if (wq_moved && w_minors < 2) {
    score -= (EARLY_QUEEN_PENALTY * open_w) / 64;
  }
  if (bq_moved && b_minors < 2) {
    score += (EARLY_QUEEN_PENALTY * open_w) / 64;
  }

  auto penalize_flank_push = [&](Square start, OccupancyType pc, int dir) {
    const int s = static_cast<int>(to_index(start));
    const int t = s + dir;
    if (t < 0 || t >= 64) {
      return;
    }
    const auto from_piece = board.pieces[static_cast<std::size_t>(s)];
    const auto to_piece = board.pieces[static_cast<std::size_t>(t)];
    if (from_piece == OccupancyType::empty && to_piece == pc && open_w > 40) {
      score += is_white(pc) ? -FLANK_PAWN_PENALTY : +FLANK_PAWN_PENALTY;
    }
  };
  penalize_flank_push(Square::A2, OccupancyType::wP, +8);
  penalize_flank_push(Square::H2, OccupancyType::wP, +8);
  penalize_flank_push(Square::A7, OccupancyType::bP, -8);
  penalize_flank_push(Square::H7, OccupancyType::bP, -8);

  return score;
}

// Final evaluation: strictly White-centric; do not flip by side_to_move
int evaluate_board(const Board& board) {
  if (board.king_captured == PieceColor::White) {
    return -100000; // Black wins (bad for White)
  } else if (board.king_captured == PieceColor::Black) {
    return 100000; // White wins
  }

  int total_eval = 0;
  total_eval += evaluate_material(board);
  total_eval += evaluate_pawn_center_control(board);
  total_eval += evaluate_center_control(board);
  total_eval += evaluate_attacking_pieces(board);
  total_eval += evaluate_king_safety(board);
  total_eval += evaluate_king_mobility(board);
  total_eval += evaluate_pins(board);
  total_eval += evaluate_pst(board);
  total_eval += evaluate_passed_pawns(board);
  total_eval += evaluate_initiative(board);

  return total_eval;
}

} // namespace chess