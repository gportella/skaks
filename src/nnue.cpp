#include "chess/nnue.hpp"

#include "chess/types_io.hpp"
#include "sf_nnue/nnue.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace chess {
namespace {
constexpr std::size_t kPiecesPerKing = kNnuePieceKinds * kNnueSquares;
std::shared_ptr<NnueNetwork> g_active_nnue;
constexpr int kQuantMin = -127;
constexpr int kQuantMax = 127;
bool g_sf_nnue_loaded = false;

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

int8_t quantize_int8(float v) {
  const int iv = static_cast<int>(std::lround(v));
  return static_cast<int8_t>(std::clamp(iv, kQuantMin, kQuantMax));
}

int32_t quantize_int32(float v) {
  const long long iv = static_cast<long long>(std::llround(v));
  const long long clamped = std::clamp(iv, -2147483648LL, 2147483647LL);
  return static_cast<int32_t>(clamped);
}

int sf_piece_code(OccupancyType occ) {
  switch (occ) {
  case OccupancyType::wK:
    return 1;
  case OccupancyType::wQ:
    return 2;
  case OccupancyType::wR:
    return 3;
  case OccupancyType::wB:
    return 4;
  case OccupancyType::wN:
    return 5;
  case OccupancyType::wP:
    return 6;
  case OccupancyType::bK:
    return 7;
  case OccupancyType::bQ:
    return 8;
  case OccupancyType::bR:
    return 9;
  case OccupancyType::bB:
    return 10;
  case OccupancyType::bN:
    return 11;
  case OccupancyType::bP:
    return 12;
  default:
    return 0;
  }
}

bool build_sf_arrays(const Board& board, std::array<int, 33>& pieces,
                     std::array<int, 33>& squares, std::string& error) {
  pieces.fill(0);
  squares.fill(0);

  // Ensure kings are first, as expected by the bundled SF/sunfish NNUE code.
  std::size_t idx = 0;
  const int wking_sq = locate_king(board, PieceColor::White);
  const int bking_sq = locate_king(board, PieceColor::Black);
  if (wking_sq < 0 || bking_sq < 0) {
    error = "Missing king when building NNUE features";
    return false;
  }
  pieces[idx] = sf_piece_code(OccupancyType::wK);
  squares[idx] = wking_sq;
  ++idx;
  pieces[idx] = sf_piece_code(OccupancyType::bK);
  squares[idx] = bking_sq;
  ++idx;

  for (std::size_t sq = 0; sq < 64; ++sq) {
    if (sq == static_cast<std::size_t>(wking_sq) ||
        sq == static_cast<std::size_t>(bking_sq)) {
      continue;
    }
    const auto occ = board.pieces[sq];
    if (occ == OccupancyType::empty) {
      continue;
    }
    const int code = sf_piece_code(occ);
    if (code == 0) {
      std::ostringstream oss;
      oss << "Unsupported occupancy for SF NNUE mapping at sq " << sq;
      error = oss.str();
      return false;
    }
    if (idx + 1 >= pieces.size()) {
      error = "Too many pieces for SF NNUE input";
      return false;
    }
    pieces[idx] = code;
    squares[idx] = static_cast<int>(sq);
    ++idx;
  }

  pieces[idx] = 0;
  squares[idx] = 0;
  return true;
}

int eval_sf_backend(const Board& board) {
  std::array<int, 33> pieces{};
  std::array<int, 33> squares{};
  std::string error;
  if (!build_sf_arrays(board, pieces, squares, error)) {
    throw std::runtime_error(error);
  }
  int player = (board.side_to_move == SideToMove::White) ? 0 : 1;

  // sunfish/SF NNUE expect the piece list from the POV of the player argument
  // ("friend" pieces first). If Black to move, swap colors and mirror the
  // board so the net sees its own pieces as White on the near ranks.
  if (player == 1) {
    auto swap_code = [](int code) {
      if (code >= 1 && code <= 6)
        return code + 6; // white piece -> black code
      if (code >= 7 && code <= 12)
        return code - 6; // black piece -> white code
      return code;
    };
    for (std::size_t i = 0; i < pieces.size() && pieces[i] != 0; ++i) {
      pieces[i] = swap_code(pieces[i]);
      const int sq = squares[i];
      const int file = sq & 7;
      const int rank = sq >> 3;
      const int mirrored_rank = 7 - rank;
      squares[i] = mirrored_rank * 8 + file; // flip ranks only
    }
    player = 0; // after transform, present as "white to move"
  }

  return nnue_evaluate(player, pieces.data(), squares.data());
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

void NnueNetwork::build_accumulator(const NnueFeatures& feat,
                                    NnueAccumulator& acc) const {
  const std::size_t hidden = hidden_size();
  if (w1.size() != hidden * input_size() || w2.size() != hidden) {
    throw std::runtime_error("NNUE weights have inconsistent dimensions");
  }
  acc.activations.assign(hidden, 0);
  for (std::size_t h = 0; h < hidden; ++h) {
    int32_t sum = b1[h];
    const int8_t* w_row = w1.data() + h * input_size();
    for (std::size_t i = 0; i < input_size(); ++i) {
      sum +=
          static_cast<int32_t>(w_row[i]) * static_cast<int32_t>(feat.values[i]);
    }
    acc.activations[h] = std::max<int32_t>(0, sum);
  }
}

float NnueNetwork::forward(const NnueAccumulator& acc) const {
  if (acc.activations.size() != hidden_size()) {
    throw std::runtime_error("Accumulator size mismatch");
  }
  int64_t out = static_cast<int64_t>(b2);
  for (std::size_t h = 0; h < hidden_size(); ++h) {
    out +=
        static_cast<int64_t>(w2[h]) * static_cast<int64_t>(acc.activations[h]);
  }
  return static_cast<float>(out) * output_scale;
}

float NnueNetwork::forward(const NnueFeatures& feat) const {
  NnueAccumulator acc;
  build_accumulator(feat, acc);
  return forward(acc);
}

bool load_nnue_from_file(const std::string& path, NnueNetwork& out,
                         std::string& error) {
  g_sf_nnue_loaded = false;
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
  float scale = 1.0f;

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
  if (const auto scale_node = nnue["scale"]) {
    try {
      scale = scale_node.as<float>();
    } catch (const YAML::BadConversion& ex) {
      std::ostringstream oss;
      oss << "Invalid float for 'scale': " << ex.what();
      error = oss.str();
      return false;
    }
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

  out.w1.resize(w1.size());
  for (std::size_t i = 0; i < w1.size(); ++i) {
    out.w1[i] = quantize_int8(w1[i]);
  }
  out.b1.resize(b1.size());
  for (std::size_t i = 0; i < b1.size(); ++i) {
    out.b1[i] = quantize_int32(b1[i]);
  }
  out.w2.resize(w2.size());
  for (std::size_t i = 0; i < w2.size(); ++i) {
    out.w2[i] = quantize_int8(w2[i]);
  }
  out.b2 = quantize_int32(b2);
  out.output_scale = scale;
  return true;
}

void set_active_nnue(std::shared_ptr<NnueNetwork> net) {
  g_sf_nnue_loaded = false;
  g_active_nnue = std::move(net);
}

std::shared_ptr<const NnueNetwork> active_nnue() {
  return g_active_nnue;
}

bool load_sf_nnue(const std::string& path, std::string& error) {
  g_active_nnue.reset();
  g_sf_nnue_loaded = false;
  const auto p = std::filesystem::path(path);
  if (!std::filesystem::exists(p)) {
    error = "NNUE file not found";
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::is_regular_file(p, ec)) {
    error = "NNUE path is not a regular file";
    return false;
  }
  const auto file_size = std::filesystem::file_size(p, ec);
  if (file_size < 1024 || ec) {
    error = "NNUE file is too small or unreadable";
    return false;
  }

  // Basic header sanity check to fail fast on git-lfs pointers or wrong files.
  // Accept either Stockfish-style magic "NNUE" or the sunfishNNUE variant
  // that starts with NnueVersion (0x7AF32F16 little-endian).
  std::ifstream fin(path, std::ios::binary);
  char magic[4] = {0, 0, 0, 0};
  fin.read(magic, 4);
  const uint32_t hdr_u32 =
      static_cast<uint32_t>(static_cast<unsigned char>(magic[0])) |
      (static_cast<uint32_t>(static_cast<unsigned char>(magic[1])) << 8) |
      (static_cast<uint32_t>(static_cast<unsigned char>(magic[2])) << 16) |
      (static_cast<uint32_t>(static_cast<unsigned char>(magic[3])) << 24);
  const bool sf_magic = std::string_view(magic, 4) == "NNUE";
  const bool sunfish_magic =
      hdr_u32 == 0x7AF32F16u; // matches sunfishNNUE NnueVersion
  if (!sf_magic && !sunfish_magic) {
    error = "NNUE file header invalid (expected 'NNUE' or 0x7AF32F16)";
    return false;
  }
  try {
    nnue_init(path.c_str());
    g_sf_nnue_loaded = true;
    return true;
  } catch (const std::exception& ex) {
    error = ex.what();
    return false;
  }
}

bool sf_nnue_active() {
  return g_sf_nnue_loaded;
}

int evaluate_sf_nnue(const Board& board) {
  if (!g_sf_nnue_loaded) {
    throw std::runtime_error("SF NNUE backend not loaded");
  }
  const int stm_cp = eval_sf_backend(board); // centipawns from side-to-move POV
  return (board.side_to_move == SideToMove::White) ? stm_cp : -stm_cp;
}

} // namespace chess
