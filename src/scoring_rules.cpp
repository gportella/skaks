#include "chess/scoring_rules.hpp"

#include "chess/board.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

namespace chess {

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
  total_eval += evaluate_pawn_center_control(board);
  total_eval += evaluate_center_control(board);
  total_eval += evaluate_material(board);
  if (board.side_to_move == SideToMove::White) {
    return total_eval; // Evaluation for White
  } else {
    return -total_eval; // Evaluation for Black
  }
}

} // namespace chess