#pragma once

#include "chess/moves.hpp"

#include <array>
#include <cstdint>

namespace chess {

constexpr int kHistoryMax = 1'000'000;
constexpr int kHistoryDecayTrigger = kHistoryMax / 2;
constexpr int kHistoryDecayFactor = 2;

inline bool is_quiet_move(const Move& move) {
  const bool is_capture = move.captured_pc != OccupancyType::empty;
  const bool is_promo = move.promo_pc != OccupancyType::empty;
  const bool is_ep = flag_is_ep(move.flags);
  const bool is_castle = flag_is_castle(move.flags) || flag_is_long_castle(move.flags);
  return !is_capture && !is_promo && !is_ep && !is_castle;
}

class HistoryTable {
public:
  using Matrix = std::array<std::array<int, 64>, 64>;

  void clear();
  void update(const Move& move, int depth, bool success);
  int score(const Move& move) const;
  const Matrix& matrix() const;
  Matrix& matrix();
  void maybe_decay();

private:
  void apply_delta(const Move& move, int delta);
  Matrix entries_{};
  int max_value_ = 0;
};

class CounterMoveTable {
public:
  static constexpr std::size_t kPieceSlots = static_cast<std::size_t>(OccupancyType::bK) + 1;
  using Table = std::array<std::array<uint32_t, 64>, kPieceSlots>;

  void clear();
  void store(const Move& previous, const Move& reply);
  uint32_t lookup(const Move& previous) const;

private:
  Table entries_{};
};

class MoveOrderingTables {
public:
  void reset();
  void record_history(const Move& move, int depth, bool success);
  int history_score(const Move& move) const;
  const HistoryTable::Matrix* history_matrix() const;
  void record_counter(const Move& parent_move, const Move& reply_move);
  uint32_t counter_move(const Move& parent_move) const;
  void maybe_decay_history();

private:
  HistoryTable history_;
  CounterMoveTable counter_;
};

} // namespace chess
