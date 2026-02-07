#include "chess/transposition_table.hpp"

#include "chess/score.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace chess {

namespace {

std::size_t next_pow2(std::size_t value) {
  if (value == 0) {
    return 1;
  }
  std::size_t pow_two = 1;
  while (pow_two < value) {
    pow_two <<= 1U;
  }
  return pow_two;
}

std::size_t clusters_for_entries(std::size_t entry_count) {
  if (entry_count == 0) {
    entry_count = 1;
  }
  const std::size_t clusters =
      (entry_count + TranspositionTable::kClusterSize - 1) /
      TranspositionTable::kClusterSize;
  return next_pow2(clusters);
}

std::size_t entries_for_megabytes(std::size_t megabytes) {
  if (megabytes == 0) {
    return 1;
  }
  const std::size_t bytes = megabytes * 1024ULL * 1024ULL;
  const std::size_t entry_size = sizeof(TranspositionTable::Slot);
  return std::max<std::size_t>(1, bytes / entry_size);
}

} // namespace

TranspositionTable::TranspositionTable(std::size_t entry_count) {
  resize(entry_count);
}

void TranspositionTable::resize(std::size_t entry_count) {
  const std::size_t cluster_count = clusters_for_entries(entry_count);
  clusters_ = std::vector<Cluster>(cluster_count);
  mask_ = cluster_count - 1U;
  clear();
}

void TranspositionTable::resize_mb(std::size_t megabytes) {
  resize(entries_for_megabytes(megabytes));
}

void TranspositionTable::clear() {
  for (auto& cluster : clusters_) {
    for (auto& slot : cluster.slots) {
      slot.key.store(0, std::memory_order_relaxed);
      slot.depth = -1;
      slot.score = 0;
      slot.flag = TranspositionFlag::None;
      slot.best_move = Move{};
    }
  }
}

bool TranspositionTable::probe(std::uint64_t key,
                               TranspositionEntry& out) const {
  const auto& cluster = clusters_[index(key)];
  for (const auto& slot : cluster.slots) {
    const std::uint64_t slot_key = slot.key.load(std::memory_order_acquire);
    if (slot_key == key && slot.depth >= 0) {
      out.key = slot_key;
      out.best_move = slot.best_move;
      out.depth = slot.depth;
      out.score = slot.score;
      out.flag = slot.flag;
      return true;
    }
  }
  return false;
}

void TranspositionTable::store(std::uint64_t key, int depth, int score,
                               TranspositionFlag flag, const Move& move,
                               int ply) {
  auto& cluster = clusters_[index(key)];
  Slot* replacement = nullptr;
  int lowest_depth = std::numeric_limits<int>::max();

  for (auto& slot : cluster.slots) {
    const std::uint64_t slot_key = slot.key.load(std::memory_order_relaxed);
    if (slot_key == key) {
      if (slot.depth > depth) {
        return;
      }
      replacement = &slot;
      break;
    }
    if (slot.depth < lowest_depth) {
      lowest_depth = slot.depth;
      replacement = &slot;
    }
  }

  if (!replacement) {
    return;
  }

  // Prevent overwriting a deeper search result with a shallower one (e.g.
  // QSearch overwriting PV) unless we are updating the same position.
  if (replacement->key.load(std::memory_order_relaxed) != key &&
      replacement->depth > depth) {
    return;
  }

  const int normalized = normalize_mate_score(score, ply);
  replacement->best_move = move;
  replacement->depth = depth;
  replacement->score = encode_score(normalized, ply);
  replacement->flag = flag;
  replacement->key.store(key, std::memory_order_release);
}

int TranspositionTable::decode_score(int stored_score, int ply) {
  return decode_mate_score(stored_score, ply);
}

int TranspositionTable::encode_score(int score, int ply) {
  return encode_mate_score(score, ply);
}

} // namespace chess
