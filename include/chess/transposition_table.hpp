#pragma once

#include "chess/defaults.hpp"
#include "chess/moves.hpp"
#include "chess/score.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {

enum class TranspositionFlag { None, Exact, LowerBound, UpperBound };

struct TranspositionEntry {
  std::uint64_t key = 0;
  Move best_move{};
  int depth = -1;
  int score = 0;
  TranspositionFlag flag = TranspositionFlag::None;
};

class TranspositionTable {
public:
  static constexpr std::size_t kClusterSize = 4;

  struct Slot {
    std::atomic<std::uint64_t> key{0};
    Move best_move{};
    int depth{-1};
    int score{0};
    TranspositionFlag flag{TranspositionFlag::None};
  };

  struct alignas(64) Cluster {
    std::array<Slot, kClusterSize> slots{};
  };

  explicit TranspositionTable(std::size_t entry_count = 1 << 16);

  void resize(std::size_t entry_count);
  void resize_mb(std::size_t megabytes);
  void clear();
  bool probe(std::uint64_t key, TranspositionEntry& out) const;
  void store(std::uint64_t key, int depth, int score, TranspositionFlag flag,
             const Move& move, int ply);

  static int decode_score(int stored_score, int ply);
  static int encode_score(int score, int ply);

private:
  std::size_t index(std::uint64_t key) const {
    return static_cast<std::size_t>(key) & mask_;
  }

  std::vector<Cluster> clusters_;
  std::size_t mask_ = 0;
};

} // namespace chess
