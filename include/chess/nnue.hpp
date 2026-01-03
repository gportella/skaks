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

// Simple accumulator holding post-ReLU hidden activations for the current
// features. This is a per-position cache; incremental update is a future step.
struct NnueAccumulator {
  std::vector<int32_t> activations;
};

// Map OccupancyType to [0, kNnuePieceKinds).
std::size_t nnue_piece_index(OccupancyType occ);

// Extract NNUE input features from a Board.
NnueFeatures make_nnue_features(const Board& board);

// Map engine occupancy to Stockfish NNUE piece codes.
int sf_nnue_piece_code(OccupancyType occ);

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

// Load Stockfish-style binary NNUE (.nnue). On success, evaluation switches to
// the SF backend; YAML/PT networks remain available via set_active_nnue.
bool load_sf_nnue(const std::string& path, std::string& error);
bool sf_nnue_active();
int evaluate_sf_nnue(const Board& board);

// Manage the active NNUE network used by evaluation.
void set_active_nnue(std::shared_ptr<NnueNetwork> net);
std::shared_ptr<const NnueNetwork> active_nnue();

// Minimal two-layer dense network (ReLU hidden) with int8 quantized weights.
// Hidden size is typically 256 in this build; weights are assumed row-major.
struct NnueNetwork {
  std::vector<int8_t> w1;  // [hidden, input]
  std::vector<int32_t> b1; // [hidden]
  std::vector<int8_t> w2;  // [hidden]
  int32_t b2{0};
  float output_scale{1.0f}; // scale int output to cp
  std::size_t hidden_size() const {
    return b1.size();
  }
  std::size_t input_size() const {
    return kNnueInputs;
  }
  // Build activations (post-ReLU) from features.
  void build_accumulator(const NnueFeatures& feat, NnueAccumulator& acc) const;
  // Forward using either a prebuilt accumulator or raw features.
  float forward(const NnueFeatures& feat) const;
  float forward(const NnueAccumulator& acc) const;
};

} // namespace chess
