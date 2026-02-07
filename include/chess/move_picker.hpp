#pragma once

#include "chess/move_ordering.hpp"
#include "chess/moves.hpp"

#include <array>
#include <cstdint>

namespace chess {

class MovePicker {

public:
  enum class Stage {
    kPlayTT,
    kGenCaptures,
    kPlayGoodCaptures,
    kPlayKiller,
    kPlayCounter,
    kGenQuiets,
    kPlayQuiets,
    kPlayBadCaptures,
    DONE
  };

  MovePicker(const Board& board, SideToMove stm, Move* move,
             const KillerTable* killers,
             const HistoryTable::Matrix* history_matrix, int ply,
             uint32_t counter_code,
             const ContinuationHistoryTable* continuation_table,
             const Move* parent_move, bool in_check, bool qsearch = false);

  Move nextMove();

private:
  static constexpr uint16_t kInvalidIndex = static_cast<uint16_t>(-1);

  uint32_t tt_code_ = 0;
  Move* tt_move_{};
  bool tt_returned_ = false;
  const Board* board_ = nullptr;
  SideToMove stm_ = SideToMove::White;
  bool in_check_ = false;
  bool qsearch_ = false;
  const KillerTable* killers_ = nullptr;
  const HistoryTable::Matrix* history_matrix_ = nullptr;
  int ply_ = -1;
  uint32_t counter_code_ = 0;
  const ContinuationHistoryTable* continuation_table_ = nullptr;
  const Move* parent_move_ = nullptr;

  std::array<uint32_t, kMaxMovementCount> good_captures_{};
  std::array<uint32_t, kMaxMovementCount> bad_captures_{};
  std::array<int, kMaxMovementCount> good_capture_keys_{};
  std::array<int, kMaxMovementCount> bad_capture_keys_{};
  uint16_t good_capture_count_ = 0;
  uint16_t bad_capture_count_ = 0;
  uint16_t good_capture_index_ = 0;
  uint16_t bad_capture_index_ = 0;
  bool captures_generated_ = false;

  std::array<uint32_t, kMaxMovementCount> quiets_{};
  uint16_t quiet_count_ = 0;
  uint16_t quiet_index_ = 0;
  bool quiets_generated_ = false;
  bool quiets_sorted_ = false;
  bool quiet_cache_valid_ = false;
  uint16_t quiet_primary_idx_ = kInvalidIndex;
  uint16_t quiet_secondary_idx_ = kInvalidIndex;
  uint16_t quiet_counter_idx_ = kInvalidIndex;

  void ensure_quiets_generated();
  void ensure_quiets_sorted();
  void remove_tt_from_quiets();
  void sort_quiets();

  Stage move_stage = Stage::kPlayTT;
};

} // namespace chess
