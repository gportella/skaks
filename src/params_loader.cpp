#include "chess/params_loader.hpp"

#include "chess/search_params.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
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

bool parse_double(const YAML::Node& node, const char* key, double& out,
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
    out = node[key].as<double>();
    return true;
  } catch (const YAML::BadConversion& ex) {
    std::ostringstream oss;
    oss << "Invalid float for '" << key << "': " << ex.what();
    error = oss.str();
    return false;
  }
}

bool parse_search(const YAML::Node& search_node, chess::SearchParams& search,
                  std::string& error) {
  auto parse_field = [&](const char* key, int& dest) -> bool {
    return parse_int(search_node, key, dest, error);
  };
  auto parse_field_double = [&](const char* key, double& dest) -> bool {
    return parse_double(search_node, key, dest, error);
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
  if (!parse_field("quiescence_max_quiet_checks",
                   search.quiescence_max_quiet_checks))
    return false;
  if (!parse_field("null_move_reduction", search.null_move_reduction))
    return false;
  if (!parse_field("null_move_reduction_divisor",
                   search.null_move_reduction_divisor))
    return false;
  if (!parse_field("null_move_min_depth", search.null_move_min_depth))
    return false;
  if (search.null_move_reduction < 0 || search.null_move_min_depth < 0 ||
      search.null_move_reduction_divisor <= 0) {
    error = "null-move parameters must be non-negative; divisor > 0";
    return false;
  }
  if (!parse_field_double("lmr_intercept", search.lmr_intercept))
    return false;
  if (!parse_field_double("lmr_divisor", search.lmr_divisor))
    return false;
  if (!parse_field_double("lmr_history_divisor", search.lmr_history_divisor))
    return false;
  if (!parse_field_double("lmr_pv_offset", search.lmr_pv_offset))
    return false;
  if (search.lmr_divisor <= 0.0) {
    error = "lmr_divisor must be positive";
    return false;
  }
  if (search.lmr_history_divisor <= 0.0) {
    error = "lmr_history_divisor must be positive";
    return false;
  }
  if (search.lmr_pv_offset < 0.0) {
    error = "lmr_pv_offset must be non-negative";
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

  if (const auto search_node = root["search"]) {
    if (!search_node.IsMap()) {
      error = "'search' must be a map";
      return false;
    }
    if (!parse_search(search_node, working.search, error)) {
      return false;
    }
  }

  if (const auto search_nnue_node = root["search_nnue"]) {
    if (!search_nnue_node.IsMap()) {
      error = "'search_nnue' must be a map";
      return false;
    }
    if (!parse_search(search_nnue_node, working.search_nnue, error)) {
      return false;
    }
    working.use_search_nnue = true;
  }

  params = working;
  return true;
}

} // namespace chess
