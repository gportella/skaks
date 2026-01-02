#include "chess/params_loader.hpp"

#include "chess/evaluation_params.hpp"
#include "chess/search_params.hpp"

#include <sstream>
#include <string>
#include <type_traits>
#include <yaml-cpp/yaml.h>

namespace {

bool parse_int(const YAML::Node& node, const char* key, int& out,
               std::string& error) {
  if (!node[key]) {
    return true; // optional
  }
  if (!node[key].IsScalar()) {
    std::ostringstream oss;
    oss << "Expected scalar for '" << key << "'";
    error = oss.str();
    return false;
  }
  try {
    out = node[key].as<int>();
    return true;
  } catch (const YAML::BadConversion& ex) {
    std::ostringstream oss;
    oss << "Invalid integer for '" << key << "': " << ex.what();
    error = oss.str();
    return false;
  }
}

bool parse_pin_penalty(const YAML::Node& node, const char* key,
                       chess::PinPenalty& out, std::string& error) {
  const auto child = node[key];
  if (!child) {
    return true; // optional
  }
  if (!child.IsMap()) {
    std::ostringstream oss;
    oss << "Expected map for '" << key << "'";
    error = oss.str();
    return false;
  }
  if (!parse_int(child, "base", out.base, error)) {
    return false;
  }
  if (!parse_int(child, "mobility", out.mobility, error)) {
    return false;
  }
  return true;
}

template <typename ArrayT>
bool parse_int_array(const YAML::Node& node, const char* key, ArrayT& target,
                     std::string& error) {
  const auto child = node[key];
  if (!child) {
    return true; // optional
  }
  if (!child.IsSequence()) {
    std::ostringstream oss;
    oss << "Expected sequence for '" << key << "'";
    error = oss.str();
    return false;
  }
  if (child.size() != target.size()) {
    std::ostringstream oss;
    oss << "Expected " << target.size() << " entries for '" << key << "'";
    error = oss.str();
    return false;
  }
  for (std::size_t i = 0; i < target.size(); ++i) {
    if (!child[i].IsScalar()) {
      std::ostringstream oss;
      oss << "Non-scalar entry at index " << i << " for '" << key << "'";
      error = oss.str();
      return false;
    }
    try {
      target[i] = child[i].as<int>();
    } catch (const YAML::BadConversion& ex) {
      std::ostringstream oss;
      oss << "Invalid integer at index " << i << " for '" << key
          << "': " << ex.what();
      error = oss.str();
      return false;
    }
  }
  return true;
}

bool parse_evaluation(const YAML::Node& eval_node, chess::EvaluationParams& eval,
                      std::string& error) {
  auto parse_field = [&](const char* key, int& dest) -> bool {
    return parse_int(eval_node, key, dest, error);
  };

  if (!parse_field("check_penalty", eval.check_penalty))
    return false;
  if (!parse_field("pawn_shield_bonus", eval.pawn_shield_bonus))
    return false;
  if (!parse_field("castling_bonus", eval.castling_bonus))
    return false;
  if (!parse_field("tempo_bonus", eval.tempo_bonus))
    return false;
  if (!parse_field("threat_weight", eval.threat_weight))
    return false;
  if (!parse_field("passed_pawn_base", eval.passed_pawn_base))
    return false;
  if (!parse_field("passed_pawn_advance", eval.passed_pawn_advance))
    return false;
  if (!parse_field("hanging_divisor", eval.hanging_divisor))
    return false;
  if (eval.hanging_divisor == 0) {
    error = "hanging_divisor must be non-zero";
    return false;
  }
  if (!parse_field("hanging_min_penalty", eval.hanging_min_penalty))
    return false;
  if (!parse_field("king_ring_base", eval.king_ring_base))
    return false;
  if (!parse_field("king_ring_defended_scale", eval.king_ring_defended_scale))
    return false;
  if (!parse_field("king_ring_enemy_occupier", eval.king_ring_enemy_occupier))
    return false;
  if (!parse_field("king_ring_enemy_piece_material_scale",
                   eval.king_ring_enemy_piece_material_scale))
    return false;
  if (!parse_field("bishop_pair_bonus", eval.bishop_pair_bonus))
    return false;
  if (!parse_field("rook_open_file_bonus", eval.rook_open_file_bonus))
    return false;
  if (!parse_field("rook_semi_open_file_bonus", eval.rook_semi_open_file_bonus))
    return false;
  if (!parse_field("mobility_scaling", eval.mobility_scaling))
    return false;
  if (!parse_field("knight_dev_bonus", eval.knight_dev_bonus))
    return false;
  if (!parse_field("bishop_dev_bonus", eval.bishop_dev_bonus))
    return false;
  if (!parse_field("connect_rooks_bonus", eval.connect_rooks_bonus))
    return false;
  if (!parse_field("central_pawn_bonus", eval.central_pawn_bonus))
    return false;
  if (!parse_field("castle_urgency", eval.castle_urgency))
    return false;
  if (!parse_field("early_queen_penalty", eval.early_queen_penalty))
    return false;
  if (!parse_field("flank_pawn_penalty", eval.flank_pawn_penalty))
    return false;
  if (!parse_field("knight_mobility_scale", eval.knight_mobility_scale))
    return false;
  if (!parse_field("bishop_mobility_scale", eval.bishop_mobility_scale))
    return false;
  if (!parse_field("rook_mobility_scale", eval.rook_mobility_scale))
    return false;
  if (!parse_field("queen_mobility_scale", eval.queen_mobility_scale))
    return false;
  if (!parse_field("doubled_pawn_penalty", eval.doubled_pawn_penalty))
    return false;
  if (!parse_field("isolated_pawn_penalty", eval.isolated_pawn_penalty))
    return false;
  if (!parse_field("backward_pawn_penalty", eval.backward_pawn_penalty))
    return false;

  if (!parse_int_array(eval_node, "king_attack_weights",
                       eval.king_attack_weights, error))
    return false;
  if (!parse_int_array(eval_node, "threat_base", eval.threat_base, error))
    return false;

  if (!parse_pin_penalty(eval_node, "bishop_pin_penalty",
                         eval.bishop_pin_penalty, error))
    return false;
  if (!parse_pin_penalty(eval_node, "rook_pin_penalty", eval.rook_pin_penalty,
                         error))
    return false;
  if (!parse_pin_penalty(eval_node, "knight_pin_penalty",
                         eval.knight_pin_penalty, error))
    return false;
  if (!parse_pin_penalty(eval_node, "pawn_pin_straight_penalty",
                         eval.pawn_pin_straight_penalty, error))
    return false;
  if (!parse_pin_penalty(eval_node, "pawn_pin_diagonal_penalty",
                         eval.pawn_pin_diagonal_penalty, error))
    return false;

  return true;
}

bool parse_search(const YAML::Node& search_node, chess::SearchParams& search,
                  std::string& error) {
  auto parse_field = [&](const char* key, int& dest) -> bool {
    return parse_int(search_node, key, dest, error);
  };

  if (!parse_field("aspiration_window_initial",
                   search.aspiration_window_initial))
    return false;
  if (!parse_field("aspiration_window_max", search.aspiration_window_max))
    return false;
  if (search.aspiration_window_initial <= 0 ||
      search.aspiration_window_max <= 0) {
    error = "aspiration windows must be positive";
    return false;
  }
  if (!parse_field("quiescence_delta_margin", search.quiescence_delta_margin))
    return false;
  if (!parse_field("quiescence_max_ply", search.quiescence_max_ply))
    return false;
  if (!parse_field("quiescence_max_noisy_moves",
                   search.quiescence_max_noisy_moves))
    return false;
  if (!parse_field("quiescence_zero_gain_skip_index",
                   search.quiescence_zero_gain_skip_index))
    return false;
  if (!parse_field("null_move_reduction", search.null_move_reduction))
    return false;
  if (!parse_field("null_move_min_depth", search.null_move_min_depth))
    return false;
  if (search.null_move_reduction < 0 || search.null_move_min_depth < 0) {
    error = "null-move parameters must be non-negative";
    return false;
  }
  return true;
}

} // namespace

namespace chess {

bool load_engine_params_from_file(const std::string& path, EngineParams& params,
                                  std::string& error) {
  EngineParams working = params;
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::BadFile& ex) {
    std::ostringstream oss;
    oss << "Failed to open params file: " << ex.what();
    error = oss.str();
    return false;
  } catch (const YAML::ParserException& ex) {
    std::ostringstream oss;
    oss << "Failed to parse params file: " << ex.what();
    error = oss.str();
    return false;
  }

  if (!root || root.IsNull()) {
    params = working;
    return true; // empty file is treated as no overrides
  }
  if (!root.IsMap()) {
    error = "Root YAML node must be a map";
    return false;
  }

  if (const auto eval_node = root["evaluation"]) {
    if (!eval_node.IsMap()) {
      error = "'evaluation' must be a map";
      return false;
    }
    if (!parse_evaluation(eval_node, working.evaluation, error)) {
      return false;
    }
  }

  if (const auto search_node = root["search"]) {
    if (!search_node.IsMap()) {
      error = "'search' must be a map";
      return false;
    }
    if (!parse_search(search_node, working.search, error)) {
      return false;
    }
  }

  params = working;
  return true;
}

} // namespace chess
