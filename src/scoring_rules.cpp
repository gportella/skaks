#include "chess/scoring_rules.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/defaults.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <array>

namespace chess {

// Constants (adjust as needed)
constexpr int CHECK_PENALTY = 100;   // penalty/bonus for king in check
constexpr int PAWN_SHIELD_BONUS = 5; // small bonus per nearby pawn to own king
constexpr int CASTLING_BONUS = 15;   // modest bonus if castled
// constexpr int MOBILITY_SCALING = 2;  // scale for king mobility (keep small)

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
    switch (occ) {
    case OccupancyType::wP:
      material_score += 100;
      break;
    case OccupancyType::wN:
      material_score += 320;
      break;
    case OccupancyType::wB:
      material_score += 330;
      break;
    case OccupancyType::wR:
      material_score += 500;
      break;
    case OccupancyType::wQ:
      material_score += 900;
      break;
    case OccupancyType::bP:
      material_score -= 100;
      break;
    case OccupancyType::bN:
      material_score -= 320;
      break;
    case OccupancyType::bB:
      material_score -= 330;
      break;
    case OccupancyType::bR:
      material_score -= 500;
      break;
    case OccupancyType::bQ:
      material_score -= 900;
      break;
    default:
      break;
    }
  }
  return material_score;
}

// Simple center presence (placeholder), White-centric and summed (no early return)
int evaluate_pawn_center_control(const Board& board) {
  int score = 0;
  for (auto sq :
       {to_index(Square::D4), to_index(Square::D5), to_index(Square::E4), to_index(Square::E5)}) {
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
  for (auto sq :
       {to_index(Square::D4), to_index(Square::D5), to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece != OccupancyType::empty) {
      const bool is_white_piece =
          static_cast<std::size_t>(piece) < static_cast<std::size_t>(OccupancyType::bP);
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
      for (std::uint8_t i = 0; i < board.pawn_list[to_index(PieceColor::White)].count; ++i) {
        const auto pawn_sq = board.pawn_list[to_index(PieceColor::White)].squares[i];
        if (chebyshev_dist(to_index(pawn_sq), static_cast<std::size_t>(wking_sq)) <= 2) {
          score += PAWN_SHIELD_BONUS;
        }
      }
    }
  }

  // Black pawn shield near Black king (subtract because it's good for Black)
  {
    const int bking_sq = board.king_positions[to_index(PieceColor::Black)];
    if (bking_sq != -1) {
      for (std::uint8_t i = 0; i < board.pawn_list[to_index(PieceColor::Black)].count; ++i) {
        const auto pawn_sq = board.pawn_list[to_index(PieceColor::Black)].squares[i];
        if (chebyshev_dist(to_index(pawn_sq), static_cast<std::size_t>(bking_sq)) <= 2) {
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
    const int moves = popcount_bitboard(
        king_attack_bm(board, static_cast<u_int8_t>(wking_sq), SideToMove::White));
    score += MOBILITY_SCALING * moves;
  }

  const int bking_sq = board.king_positions[to_index(PieceColor::Black)];
  if (bking_sq != -1) {
    const int moves = popcount_bitboard(
        king_attack_bm(board, static_cast<u_int8_t>(bking_sq), SideToMove::Black));
    score -= MOBILITY_SCALING * moves;
  }

  return score;
}

// Attacking pieces: if a piece is attacked by the opponent, penalize for White pieces, reward for
// Black
int evaluate_attacking_pieces(const Board& board) {
  constexpr std::array<int, static_cast<std::size_t>(OccupancyType::bK) + 1> kThreatBase = {
      0, 12, 30, 30, 45, 180, 540, 12, 30, 30, 45, 180, 540};

  int attack_score = 0;

  for (std::size_t sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::empty)
      continue;

    const bool white_piece =
        static_cast<std::size_t>(piece) < static_cast<std::size_t>(OccupancyType::bP);

    const int base_weight = kThreatBase[static_cast<std::size_t>(piece)];
    if (base_weight == 0)
      continue;

    // If this square is attacked by the opponent
    const SideToMove attacker = white_piece ? SideToMove::Black : SideToMove::White;
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

  return total_eval; // White-centric score
}

} // namespace chess