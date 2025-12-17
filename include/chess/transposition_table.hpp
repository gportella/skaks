#pragma once

#include "chess/defaults.hpp"
#include "chess/moves.hpp"
#include "chess/score.hpp"

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
  explicit TranspositionTable(std::size_t entry_count = 1 << 16) {
    resize(entry_count);
  }

  void resize(std::size_t entry_count) {
    if (entry_count == 0) {
      entry_count = 1;
    }
    std::size_t pow_two = 1;
    while (pow_two < entry_count) {
      pow_two <<= 1U;
    }
    entries_.assign(pow_two, {});
    mask_ = pow_two - 1U;
  }

  void clear() {
    for (auto& entry : entries_) {
      entry = {};
    }
  }

  bool probe(std::uint64_t key, TranspositionEntry& out) const {
    const auto& slot = entries_[index(key)];
    if (slot.depth >= 0 && slot.key == key) {
      out = slot;
      return true;
    }
    return false;
  }

  void store(std::uint64_t key, int depth, int score, TranspositionFlag flag, const Move& move,
             int ply) {
    TranspositionEntry& slot = entries_[index(key)];
    if (slot.depth > depth && slot.key == key) {
      return;
    }
    slot.key = key;
    slot.depth = depth;
    const int normalized = normalize_mate_score(score, ply);
    slot.score = encode_score(normalized, ply);
    slot.flag = flag;
    slot.best_move = move;
  }

  static int decode_score(int stored_score, int ply) {
    return decode_mate_score(stored_score, ply);
  }

  static int encode_score(int score, int ply) {
    return encode_mate_score(score, ply);
  }

private:
  std::size_t index(std::uint64_t key) const {
    return static_cast<std::size_t>(key) & mask_;
  }

  std::vector<TranspositionEntry> entries_;
  std::size_t mask_ = 0;
};

} // namespace chess
