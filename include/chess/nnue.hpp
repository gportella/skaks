#pragma once

#include "chess/board.hpp"
#include "chess/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace chess {

constexpr std::size_t kNnueSquares = 64;
constexpr std::size_t kNnuePieceKinds = 12;
constexpr std::size_t kNnueInputs = kNnueSquares * kNnuePieceKinds * 2 + 1;

struct NnueFeatures {
  std::array<int8_t, kNnueInputs> values{};
};

struct NnueAccumulator {
  std::vector<int32_t> activations;
};

struct NnueNetwork {
  std::vector<int8_t> w1;
  std::vector<int32_t> b1;
  std::vector<int8_t> w2;
  int32_t b2 = 0;
  float output_scale = 1.0f;

  [[nodiscard]] constexpr std::size_t input_size() const {
    return kNnueInputs;
  }

  [[nodiscard]] std::size_t hidden_size() const {
    return b1.size();
  }

  void build_accumulator(const NnueFeatures& feat, NnueAccumulator& acc) const;
  float forward(const NnueAccumulator& acc) const;
  float forward(const NnueFeatures& feat) const;
};

int sf_nnue_piece_code(OccupancyType occ);
std::size_t nnue_piece_index(OccupancyType occ);
NnueFeatures make_nnue_features(const Board& board);
bool load_nnue_from_file(const std::string& path, NnueNetwork& out,
                         std::string& error);
void set_active_nnue(std::shared_ptr<NnueNetwork> net);
std::shared_ptr<const NnueNetwork> active_nnue();
bool load_sf_nnue(const std::string& path, std::string& error);
bool sf_nnue_active();
int evaluate_sf_nnue(const Board& board);

} // namespace chess
