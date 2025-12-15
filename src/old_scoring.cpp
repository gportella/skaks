#include "chess/scoring_rules.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/defaults.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <array>

namespace chess {

int evaluate_king_safety(const Board& board) {
  // Placeholder for king safety evaluation
  // In a real implementation, this would analyze pawn structure, enemy piece proximity, etc.
  constexpr int kPawnShieldBonus = 5;
  constexpr int kCastlingBonus = 15;

  auto king_safety_for = [&](PieceColor color) {
    int score = 0;
    const auto color_index = to_index(color);
    const int side_sign = (color == PieceColor::White) ? 1 : -1;
    const int king_sq = board.king_positions[color_index];
    if (king_sq == -1) {
      return score;
    }

    // reward nearby pawn shield around the king
    for (std::uint8_t i = 0; i < board.pawn_list[color_index].count; ++i) {
      const auto pawn_sq = board.pawn_list[color_index].squares[i];
      if (chebyshev_dist(to_index(pawn_sq), static_cast<std::size_t>(king_sq)) <= 2) {
        score += side_sign * kPawnShieldBonus;
      }
    }

    if (board.has_castled[color_index]) {
      score += side_sign * kCastlingBonus; // modest bonus for successful castling
    }

    if (is_check(board, board.side_to_move)) {
      score -= side_sign * MATE_SCORE; // heavy penalty if king is in check
    }

    return score;
  };

  int safety_score = 0;
  safety_score += king_safety_for(PieceColor::White);
  safety_score += king_safety_for(PieceColor::Black);

  for (const auto& occ : board.pieces) {
    switch (occ) {
    case OccupancyType::wP:
      safety_score += 1;
      break;
    case OccupancyType::bP:
      safety_score -= 1;
      break;
    default:
      break;
    }
  }
  return safety_score;
}

int evaluate_pawn_center_control(const Board& board) {
  // Placeholder for pawn center control evaluation
  for (auto sq :
       {to_index(Square::D4), to_index(Square::D5), to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::wP) {
      return 10; // White controls center
    } else if (piece == OccupancyType::bP) {
      return -10; // Black controls center
    }
  }
  return 0;
}
int evaluate_center_control(const Board& board) {
  // Placeholder for center control evaluation
  for (auto sq :
       {to_index(Square::D4), to_index(Square::D5), to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece != OccupancyType::empty) {
      if (static_cast<std::size_t>(piece) < static_cast<std::size_t>(OccupancyType::bP)) {
        return 5; // White piece controls center
      } else {
        return -5; // Black piece controls center
      }
    }
  }
  return 0;
}

int evaluate_king_mobility(const Board& board) {
  // Placeholder for king mobility evaluation
  if (board.side_to_move == SideToMove::White) {
    return MOBILITY_SCALING *
           popcount_bitboard(king_attack_bm(board, static_cast<u_int8_t>(board.king_positions[0]),
                                            SideToMove::White));

  } else {
    return -MOBILITY_SCALING *
           popcount_bitboard(king_attack_bm(board, static_cast<u_int8_t>(board.king_positions[1]),
                                            SideToMove::Black));
  }
}

int evaluate_attacking_pieces(const Board& board) {
  constexpr std::array<int, static_cast<std::size_t>(OccupancyType::bK) + 1> kThreatBase = {
      0, 12, 30, 30, 45, 180, 540, 12, 30, 30, 45, 180, 540};

  int attack_score = 0;
  for (std::size_t sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::empty) {
      continue;
    }

    const bool is_white_piece =
        static_cast<std::size_t>(piece) < static_cast<std::size_t>(OccupancyType::bP);
    const SideToMove defender = is_white_piece ? SideToMove::White : SideToMove::Black;
    if (!is_square_attacked(board, static_cast<u_int8_t>(sq), flip_side(defender))) {
      continue;
    }

    const int base_weight = kThreatBase[static_cast<std::size_t>(piece)];
    if (base_weight == 0) {
      continue;
    }

    const int rank = rank_of(static_cast<int>(sq));
    const int distance_from_home = is_white_piece ? rank : (7 - rank);
    const int positional_scale = 1 + distance_from_home / 2;
    const int threat_value = base_weight * positional_scale;

    attack_score += (defender == SideToMove::White) ? -threat_value : threat_value;
  }

  return attack_score;
}

int evaluate_material(const Board& board) {
  // Placeholder for material evaluation
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

int evaluate_board(const Board& board) {
  // Placeholder for board evaluation function
  int total_eval = 0;
  if (board.king_captured == PieceColor::White) {
    return -100000; // Black wins
  } else if (board.king_captured == PieceColor::Black) {
    return 100000; // White wins
  }
  total_eval += evaluate_pawn_center_control(board);
  total_eval += evaluate_center_control(board);
  total_eval += evaluate_material(board);
  total_eval += evaluate_attacking_pieces(board);
  total_eval += evaluate_king_safety(board);
  total_eval += evaluate_king_mobility(board);
  return total_eval;
  // if (board.side_to_move == SideToMove::White) {
  //   return total_eval; // Evaluation for White
  // } else {
  //   return -total_eval; // Evaluation for Black
  // }
}

} // namespace chess