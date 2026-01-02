#pragma once

#include "chess/eval_terms.hpp"
#include "chess/types.hpp"

#include <array>

namespace chess {

struct PinPenalty {
  int base = 0;
  int mobility = 0;
};

struct EvaluationParams {
  int check_penalty = 0;
  int pawn_shield_bonus = 0;
  int castling_bonus = 0;
  int tempo_bonus = 0;
  int threat_weight = 0;
  int passed_pawn_base = 0;
  int passed_pawn_advance = 0;
  int hanging_divisor = 1;
  int hanging_min_penalty = 0;
  int king_ring_base = 0;
  int king_ring_defended_scale = 1;
  int king_ring_enemy_occupier = 0;
  int king_ring_enemy_piece_material_scale = 0;
  int bishop_pair_bonus = 0;
  int rook_open_file_bonus = 0;
  int rook_semi_open_file_bonus = 0;
  int mobility_scaling = 0;
  int knight_dev_bonus = 0;
  int bishop_dev_bonus = 0;
  int connect_rooks_bonus = 0;
  int central_pawn_bonus = 0;
  int castle_urgency = 0;
  int early_queen_penalty = 0;
  int flank_pawn_penalty = 0;
  int knight_mobility_scale = 0;
  int bishop_mobility_scale = 0;
  int rook_mobility_scale = 0;
  int queen_mobility_scale = 0;
  int doubled_pawn_penalty = 0;
  int isolated_pawn_penalty = 0;
  int backward_pawn_penalty = 0;
  std::array<int, kPieceCount> king_attack_weights{};
  std::array<int, static_cast<std::size_t>(OccupancyType::bK) + 1> threat_base{};
  PinPenalty bishop_pin_penalty{};
  PinPenalty rook_pin_penalty{};
  PinPenalty knight_pin_penalty{};
  PinPenalty pawn_pin_straight_penalty{};
  PinPenalty pawn_pin_diagonal_penalty{};
  std::array<double, static_cast<std::size_t>(OccupancyType::bK) + 1>
      reserved{}; // placeholder to preserve layout
  std::array<float, static_cast<std::size_t>(TermId::Count)> phase_weights_mg{};
  std::array<float, static_cast<std::size_t>(TermId::Count)> phase_weights_eg{};
};

inline EvaluationParams default_evaluation_params() {
  EvaluationParams params{};
  params.check_penalty = 100;
  params.pawn_shield_bonus = 25;
  params.castling_bonus = 50;
  params.tempo_bonus = 14;
  params.threat_weight = 4;
  params.passed_pawn_base = 20;
  params.passed_pawn_advance = 8;
  params.hanging_divisor = 4;
  params.hanging_min_penalty = 18;
  params.king_ring_base = 6;
  params.king_ring_defended_scale = 2;
  params.king_ring_enemy_occupier = 14;
  params.king_ring_enemy_piece_material_scale = 14;
  params.bishop_pair_bonus = 28;
  params.rook_open_file_bonus = 34;
  params.rook_semi_open_file_bonus = 18;
  params.mobility_scaling = 15;
  params.knight_dev_bonus = 12;
  params.bishop_dev_bonus = 10;
  params.connect_rooks_bonus = 10;
  params.central_pawn_bonus = 12;
  params.castle_urgency = 20;
  params.early_queen_penalty = 16;
  params.flank_pawn_penalty = 8;
  params.knight_mobility_scale = 4;
  params.bishop_mobility_scale = 5;
  params.rook_mobility_scale = 3;
  params.queen_mobility_scale = 1;
  params.doubled_pawn_penalty = 12;
  params.isolated_pawn_penalty = 16;
  params.backward_pawn_penalty = 10;
  params.king_attack_weights = {14, 32, 30, 44, 74, 20, 14, 32, 30, 44, 74, 20};
  params.threat_base = {0, 12, 30, 30, 45, 180, 540, 12, 30, 30, 45, 180, 540};
  params.bishop_pin_penalty = {12, 2};
  params.rook_pin_penalty = {6, 1};
  params.knight_pin_penalty = {15, 0};
  params.pawn_pin_straight_penalty = {6, 2};
  params.pawn_pin_diagonal_penalty = {10, 2};
  for (std::size_t i = 0; i < params.phase_weights_mg.size(); ++i) {
    params.phase_weights_mg[i] = 1.0f;
    params.phase_weights_eg[i] = 1.0f;
  }
  return params;
}

inline EvaluationParams& mutable_evaluation_params() {
  static EvaluationParams params = default_evaluation_params();
  return params;
}

inline const EvaluationParams& evaluation_params() {
  return mutable_evaluation_params();
}

inline void set_evaluation_params(const EvaluationParams& params) {
  mutable_evaluation_params() = params;
}

inline void reset_evaluation_params() {
  mutable_evaluation_params() = default_evaluation_params();
}

} // namespace chess
