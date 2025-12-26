#pragma once

#include "chess/board.hpp"
#include "chess/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace chess {

// Simple king-relative (HalfKP-style) feature layout.
// For each king bucket (white king, black king) we one-hot encode piece on
// square: 12 piece types (white then black) * 64 squares * 2 king buckets = 1536
// bits, plus one side-to-move bit.
constexpr std::size_t kNnuePieceKinds = 12; // wP..wK, bP..bK
constexpr std::size_t kNnueSquares = 64;
constexpr std::size_t kNnueKingBuckets = 2; // white king, black king
constexpr std::size_t kNnueInputs =
    kNnuePieceKinds * kNnueSquares * kNnueKingBuckets + 1;

struct NnueFeatures {
  std::array<std::int8_t, kNnueInputs> values{}; // 0/1 values
};

// Forward declaration
struct NnueNetwork;

// Map OccupancyType to [0, kNnuePieceKinds).
std::size_t nnue_piece_index(OccupancyType occ);

// Extract NNUE input features from a Board.
NnueFeatures make_nnue_features(const Board& board);

// Load NNUE weights from a YAML file. Expected schema (root or under 'nnue'):
// hidden: <int> (optional, inferred from b1 if absent)
// w1: [hidden * 1537 floats]
// b1: [hidden floats]
// w2: [hidden floats]
// b2: <float>
// On success, 'out' is populated and returns true; on failure, returns false
// and populates 'error'.
bool load_nnue_from_file(const std::string& path, NnueNetwork& out,
                         std::string& error);

// Manage the active NNUE network used by evaluation.
void set_active_nnue(std::shared_ptr<NnueNetwork> net);
std::shared_ptr<const NnueNetwork> active_nnue();

// Minimal two-layer dense network (ReLU hidden) for embedding small NNUE models.
struct NnueNetwork {
  std::vector<float> w1; // [hidden, input]
  std::vector<float> b1; // [hidden]
  std::vector<float> w2; // [hidden]
  float b2{0.0f};
  std::size_t hidden_size() const {
    return b1.size();
  }
  std::size_t input_size() const {
    return kNnueInputs;
  }
  float forward(const NnueFeatures& feat) const;
};

} // namespace chess
