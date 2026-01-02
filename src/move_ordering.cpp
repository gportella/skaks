#include "chess/move_ordering.hpp"

#include <algorithm>

namespace chess {

void HistoryTable::clear() {
  for (auto& row : entries_) {
    row.fill(0);
  }
  max_value_ = 0;
}

void HistoryTable::apply_delta(const Move& move, int delta) {
  const std::size_t from = static_cast<std::size_t>(move.from);
  const std::size_t to = static_cast<std::size_t>(move.to);
  int& entry = entries_[from][to];
  entry = std::clamp(entry + delta, 0, kHistoryMax);
  max_value_ = std::max(max_value_, entry);
}

void HistoryTable::update(const Move& move, int depth, bool success) {
  if (!is_quiet_move(move) || depth <= 0) {
    return;
  }
  const int delta = depth * depth;
  apply_delta(move, success ? delta : -delta);
}

int HistoryTable::score(const Move& move) const {
  if (!is_quiet_move(move)) {
    return 0;
  }
  const std::size_t from = static_cast<std::size_t>(move.from);
  const std::size_t to = static_cast<std::size_t>(move.to);
  return entries_[from][to];
}

const HistoryTable::Matrix& HistoryTable::matrix() const {
  return entries_;
}

HistoryTable::Matrix& HistoryTable::matrix() {
  return entries_;
}

void HistoryTable::maybe_decay() {
  if (max_value_ < kHistoryDecayTrigger) {
    return;
  }
  max_value_ = 0;
  for (auto& row : entries_) {
    for (auto& value : row) {
      value /= kHistoryDecayFactor;
      max_value_ = std::max(max_value_, value);
    }
  }
}

void CounterMoveTable::clear() {
  for (auto& row : entries_) {
    row.fill(0);
  }
}

void CounterMoveTable::store(const Move& previous, const Move& reply) {
  if (!is_quiet_move(reply)) {
    return;
  }
  const auto piece_index = static_cast<std::size_t>(previous.moving_pc);
  if (piece_index == 0 || piece_index >= entries_.size()) {
    return;
  }
  const auto to_index = static_cast<std::size_t>(previous.to);
  entries_[piece_index][to_index] = encode_move(reply.from, reply.to,
                                                reply.moving_pc, reply.captured_pc,
                                                reply.promo_pc, reply.flags);
}

uint32_t CounterMoveTable::lookup(const Move& previous) const {
  const auto piece_index = static_cast<std::size_t>(previous.moving_pc);
  if (piece_index == 0 || piece_index >= entries_.size()) {
    return 0;
  }
  const auto to_index = static_cast<std::size_t>(previous.to);
  return entries_[piece_index][to_index];
}

void MoveOrderingTables::reset() {
  history_.clear();
  counter_.clear();
}

void MoveOrderingTables::record_history(const Move& move, int depth, bool success) {
  history_.update(move, depth, success);
}

int MoveOrderingTables::history_score(const Move& move) const {
  return history_.score(move);
}

const HistoryTable::Matrix* MoveOrderingTables::history_matrix() const {
  return &history_.matrix();
}

void MoveOrderingTables::record_counter(const Move& parent_move,
                                        const Move& reply_move) {
  if (!is_quiet_move(reply_move)) {
    return;
  }
  if (parent_move.moving_pc == OccupancyType::empty) {
    return;
  }
  counter_.store(parent_move, reply_move);
}

uint32_t MoveOrderingTables::counter_move(const Move& parent_move) const {
  if (parent_move.moving_pc == OccupancyType::empty) {
    return 0;
  }
  return counter_.lookup(parent_move);
}

void MoveOrderingTables::maybe_decay_history() {
  history_.maybe_decay();
}

} // namespace chess
