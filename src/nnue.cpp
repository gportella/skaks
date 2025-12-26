#include "chess/nnue.hpp"

#include "chess/types_io.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace chess {
namespace {
constexpr std::size_t kPiecesPerKing = kNnuePieceKinds * kNnueSquares;
std::shared_ptr<NnueNetwork> g_active_nnue;

std::size_t king_bucket_index(bool white_king) {
  return white_king ? 0 : 1;
}

std::size_t feature_offset(bool white_king, std::size_t piece_idx,
                           std::size_t sq) {
  return king_bucket_index(white_king) * kPiecesPerKing +
         piece_idx * kNnueSquares + sq;
}

int locate_king(const Board& board, PieceColor color) {
  const int cached = board.king_positions[static_cast<std::size_t>(color)];
  if (cached >= 0 && cached < 64) {
    return cached;
  }
  for (int sq = 0; sq < 64; ++sq) {
    const auto occ = board.pieces[static_cast<std::size_t>(sq)];
    if ((color == PieceColor::White && occ == OccupancyType::wK) ||
        (color == PieceColor::Black && occ == OccupancyType::bK)) {
      return sq;
    }
  }
  return -1;
}

bool parse_float_sequence(const YAML::Node& node, const char* key,
                          std::vector<float>& out, std::string& error,
                          std::optional<std::size_t> expected_size,
                          bool required = true) {
  const auto child = node[key];
  if (!child) {
    if (required) {
      std::ostringstream oss;
      oss << "Missing required key '" << key << "'";
      error = oss.str();
      return false;
    }
    return true;
  }
  if (!child.IsSequence()) {
    std::ostringstream oss;
    oss << "Expected sequence for '" << key << "'";
    error = oss.str();
    return false;
  }
  if (expected_size && child.size() != *expected_size) {
    std::ostringstream oss;
    oss << "Expected " << *expected_size << " entries for '" << key << "'";
    error = oss.str();
    return false;
  }
  out.resize(child.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (!child[i].IsScalar()) {
      std::ostringstream oss;
      oss << "Non-scalar entry at index " << i << " for '" << key << "'";
      error = oss.str();
      return false;
    }
    try {
      out[i] = child[i].as<float>();
    } catch (const YAML::BadConversion& ex) {
      std::ostringstream oss;
      oss << "Invalid float at index " << i << " for '" << key
          << "': " << ex.what();
      error = oss.str();
      return false;
    }
  }
  return true;
}

bool parse_float_scalar(const YAML::Node& node, const char* key, float& out,
                        std::string& error) {
  const auto child = node[key];
  if (!child || child.IsNull()) {
    std::ostringstream oss;
    oss << "Missing required key '" << key << "'";
    error = oss.str();
    return false;
  }
  if (!child.IsScalar()) {
    std::ostringstream oss;
    oss << "Expected scalar for '" << key << "'";
    error = oss.str();
    return false;
  }
  try {
    out = child.as<float>();
  } catch (const YAML::BadConversion& ex) {
    std::ostringstream oss;
    oss << "Invalid float for '" << key << "': " << ex.what();
    error = oss.str();
    return false;
  }
  return true;
}
} // namespace

std::size_t nnue_piece_index(OccupancyType occ) {
  const auto idx = static_cast<int>(occ) - static_cast<int>(OccupancyType::wP);
  if (idx < 0 || idx >= static_cast<int>(kNnuePieceKinds)) {
    throw std::invalid_argument("invalid occupancy for NNUE mapping");
  }
  return static_cast<std::size_t>(idx);
}

NnueFeatures make_nnue_features(const Board& board) {
  NnueFeatures feat{};
  feat.values.fill(0);

  const int white_king_sq = locate_king(board, PieceColor::White);
  const int black_king_sq = locate_king(board, PieceColor::Black);

  const bool has_white_king = white_king_sq >= 0;
  const bool has_black_king = black_king_sq >= 0;

  for (std::size_t sq = 0; sq < 64; ++sq) {
    const auto occ = board.pieces[sq];
    if (occ == OccupancyType::empty) {
      continue;
    }
    const std::size_t piece_idx = nnue_piece_index(occ);
    if (has_white_king) {
      const auto idx = feature_offset(true, piece_idx, sq);
      feat.values[idx] = 1;
    }
    if (has_black_king) {
      const auto idx = feature_offset(false, piece_idx, sq);
      feat.values[idx] = 1;
    }
  }

  // Side-to-move bit at the end.
  feat.values[kNnueInputs - 1] =
      (board.side_to_move == SideToMove::White) ? 1 : 0;
  return feat;
}

float NnueNetwork::forward(const NnueFeatures& feat) const {
  const std::size_t hidden = hidden_size();
  if (w1.size() != hidden * input_size() || w2.size() != hidden) {
    throw std::runtime_error("NNUE weights have inconsistent dimensions");
  }
  float out = b2;
  for (std::size_t h = 0; h < hidden; ++h) {
    float sum = b1[h];
    const float* w_row = w1.data() + h * input_size();
    for (std::size_t i = 0; i < input_size(); ++i) {
      sum += w_row[i] * static_cast<float>(feat.values[i]);
    }
    const float act = std::max(0.0f, sum);
    out += w2[h] * act;
  }
  return out;
}

bool load_nnue_from_file(const std::string& path, NnueNetwork& out,
                         std::string& error) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::BadFile& ex) {
    std::ostringstream oss;
    oss << "Failed to open NNUE file: " << ex.what();
    error = oss.str();
    return false;
  } catch (const YAML::ParserException& ex) {
    std::ostringstream oss;
    oss << "Failed to parse NNUE file: " << ex.what();
    error = oss.str();
    return false;
  }

  const YAML::Node nnue = root["nnue"] ? root["nnue"] : root;
  if (!nnue || !nnue.IsMap()) {
    error = "NNUE file must contain a map (optionally under 'nnue')";
    return false;
  }

  std::vector<float> w1;
  std::vector<float> b1;
  std::vector<float> w2;
  float b2 = 0.0f;

  if (!parse_float_sequence(nnue, "w1", w1, error, std::nullopt)) {
    return false;
  }
  if (!parse_float_sequence(nnue, "b1", b1, error, std::nullopt)) {
    return false;
  }
  if (!parse_float_sequence(nnue, "w2", w2, error, std::nullopt)) {
    return false;
  }
  if (!parse_float_scalar(nnue, "b2", b2, error)) {
    return false;
  }

  if (b1.empty()) {
    error = "'b1' must not be empty";
    return false;
  }

  std::size_t hidden = b1.size();
  if (const auto hidden_node = nnue["hidden"]) {
    try {
      const std::size_t declared = hidden_node.as<std::size_t>();
      if (declared != hidden) {
        std::ostringstream oss;
        oss << "Hidden size mismatch: declared " << declared << " but b1 has "
            << hidden;
        error = oss.str();
        return false;
      }
    } catch (const YAML::BadConversion& ex) {
      std::ostringstream oss;
      oss << "Invalid integer for 'hidden': " << ex.what();
      error = oss.str();
      return false;
    }
  }

  if (w1.size() != hidden * kNnueInputs) {
    std::ostringstream oss;
    oss << "w1 length " << w1.size()
        << " does not match hidden*input: " << hidden * kNnueInputs;
    error = oss.str();
    return false;
  }

  if (w2.size() != hidden) {
    std::ostringstream oss;
    oss << "w2 length " << w2.size() << " does not match hidden " << hidden;
    error = oss.str();
    return false;
  }

  out.w1 = std::move(w1);
  out.b1 = std::move(b1);
  out.w2 = std::move(w2);
  out.b2 = b2;
  return true;
}

void set_active_nnue(std::shared_ptr<NnueNetwork> net) {
  g_active_nnue = std::move(net);
}

std::shared_ptr<const NnueNetwork> active_nnue() {
  return g_active_nnue;
}

} // namespace chess
