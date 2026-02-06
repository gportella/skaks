#include "chess/move_picker.hpp"

#include "chess/exchange.hpp"

#include <algorithm>
#include <limits>

namespace chess {

MovePicker::MovePicker(const Board& board, SideToMove stm, Move* tt_move,
                       const KillerTable* killers,
                       const HistoryTable::Matrix* history_matrix, int ply,
                       uint32_t counter_code,
                       const ContinuationHistoryTable* continuation_table,
                       const Move* parent_move, bool in_check, bool qsearch)
    : tt_move_(tt_move), board_(&board), stm_(stm), in_check_(in_check),
      qsearch_(qsearch), killers_(killers), history_matrix_(history_matrix),
      ply_(ply), counter_code_(counter_code),
      continuation_table_(continuation_table), parent_move_(parent_move) {

  if (tt_move_ != nullptr) {
    tt_code_ =
        encode_move(tt_move_->from, tt_move_->to, tt_move_->moving_pc,
                    tt_move_->captured_pc, tt_move_->promo_pc, tt_move_->flags);
  }
}

void MovePicker::remove_tt_from_quiets() {
  if (tt_code_ == 0) {
    return;
  }
  for (uint16_t i = 0; i < quiet_count_; ++i) {
    if (quiets_[i] == tt_code_) {
      quiets_[i] = 0;
      break;
    }
  }
}

void MovePicker::sort_quiets() {
  struct QuietKey {
    uint32_t code;
    int key;
    uint16_t order;
  };
  std::array<QuietKey, kMaxMovementCount> keys{};

  for (uint16_t i = 0; i < quiet_count_; ++i) {
    const uint32_t m = quiets_[i];
    if (m == 0) {
      keys[i] = {m, std::numeric_limits<int>::min(), i};
      continue;
    }
    int key = 100'000;
    if (history_matrix_ != nullptr) {
      const auto from = static_cast<std::size_t>(move_from(m));
      const auto to = static_cast<std::size_t>(move_to(m));
      key += (*history_matrix_)[from][to];
    }
    if (continuation_table_ != nullptr && parent_move_ != nullptr &&
        parent_move_->moving_pc != OccupancyType::empty) {
      key += continuation_table_->score(*parent_move_, decode_move(m));
    }
    if (counter_code_ != 0 && m == counter_code_) {
      key += 30'000;
    }
    keys[i] = {m, key, i};
  }

  std::sort(keys.begin(), keys.begin() + quiet_count_,
            [](const QuietKey& a, const QuietKey& b) {
              if (a.key == b.key) {
                return a.order < b.order;
              }
              return a.key > b.key;
            });

  for (uint16_t i = 0; i < quiet_count_; ++i) {
    quiets_[i] = keys[i].code;
  }
}

namespace {
inline int mvv_lva_score(OccupancyType captured, OccupancyType piece) {
  static const int scores[13] = {0,   100, 320, 330, 500, 900,  20000,
                                 100, 320, 330, 500, 900, 20000};
  return scores[static_cast<size_t>(captured)] * 10 -
         scores[static_cast<size_t>(piece)];
}

void sort_capture_list_with_keys(std::array<uint32_t, kMaxMovementCount>& list,
                                 std::array<int, kMaxMovementCount>& keys,
                                 uint16_t count) {
  struct CapKey {
    uint32_t code;
    int key;
    uint16_t order;
  };
  std::array<CapKey, kMaxMovementCount> cap_keys{};

  for (uint16_t i = 0; i < count; ++i) {
    cap_keys[i] = {list[i], keys[i], i};
  }

  std::sort(cap_keys.begin(), cap_keys.begin() + count,
            [](const CapKey& a, const CapKey& b) {
              if (a.key == b.key) {
                return a.order < b.order;
              }
              return a.key > b.key;
            });

  for (uint16_t i = 0; i < count; ++i) {
    list[i] = cap_keys[i].code;
  }
}
} // namespace

void MovePicker::ensure_quiets_generated() {
  if (in_check_ || quiets_generated_) {
    return;
  }
  quiets_ = generate_legal_moves(const_cast<Board&>(*board_), stm_, quiet_count_,
                                 MoveGenType::QuietMovesOnly);
  quiet_index_ = 0;
  remove_tt_from_quiets();
  quiets_generated_ = true;
  quiets_sorted_ = false;
}

void MovePicker::ensure_quiets_sorted() {
  if (in_check_ || !quiets_generated_ || quiets_sorted_) {
    return;
  }
  sort_quiets();
  quiets_sorted_ = true;
}

Move MovePicker::nextMove() {
  while (true) {
    switch (move_stage) {
    case Stage::kPlayTT:
      move_stage = Stage::kGenCaptures;
      if (tt_move_ != nullptr && !tt_returned_) {
        tt_returned_ = true;
        return *tt_move_;
      }
      continue;

    case Stage::kGenCaptures:
      if (!captures_generated_) {
        std::array<uint32_t, kMaxMovementCount> caps{};
        uint16_t cap_count = 0;
        const auto type =
            in_check_ ? MoveGenType::AllMoves : MoveGenType::CapturesOnly;
        caps = generate_legal_moves(const_cast<Board&>(*board_), stm_, cap_count,
                                    type);

        good_capture_count_ = 0;
        bad_capture_count_ = 0;
        good_capture_index_ = 0;
        bad_capture_index_ = 0;

        for (uint16_t i = 0; i < cap_count; ++i) {
          const uint32_t m = caps[i];
          // let's skip the TT if it shows up, should have
          // proposed this one before
          if (tt_code_ != 0 && m == tt_code_) {
            continue;
          }
          // all moves in check are potentially good to play
          // so we'll defer the SEE with a continue
          if (in_check_) {
            good_captures_[good_capture_count_++] = m;
            continue;
          }
          const auto cap = static_cast<OccupancyType>(move_captured(m));
          const auto pc = static_cast<OccupancyType>(move_piece(m));
          const int mvv = mvv_lva_score(cap, pc);
          const int see = static_exchange_eval(*board_, decode_move(m));
          const int key = 1'000'000 + mvv + see * 100;
          if (see >= 0) {
            good_captures_[good_capture_count_] = m;
            good_capture_keys_[good_capture_count_++] = key;
          } else {
            bad_captures_[bad_capture_count_] = m;
            bad_capture_keys_[bad_capture_count_++] = key;
          }
        }
        if (!in_check_) {
          sort_capture_list_with_keys(good_captures_, good_capture_keys_,
                                      good_capture_count_);
          sort_capture_list_with_keys(bad_captures_, bad_capture_keys_,
                                      bad_capture_count_);
        }
        captures_generated_ = true;
      }
      move_stage = Stage::kPlayGoodCaptures;
      continue;

    case Stage::kPlayGoodCaptures:
      if (good_capture_index_ < good_capture_count_) {
        return decode_move(good_captures_[good_capture_index_++]);
      }
      // if we're in check, we have to play all captures before moving on
      if (in_check_) {
        move_stage = Stage::DONE;
      } else if (qsearch_) {
        move_stage = Stage::kPlayBadCaptures;
      } else {
        move_stage = Stage::kPlayKiller;
      }
      continue;

    case Stage::kPlayKiller:
      if (!in_check_) {
        // TODO: Optimization: check killer/counter legality directly without
        // generating quiets; only fall back to quiet generation if needed.
        ensure_quiets_generated();

        if (killers_ != nullptr && ply_ >= 0 && ply_ < MAX_PLY) {
          const uint32_t primary =
              killers_->primary[static_cast<std::size_t>(ply_)];
          if (primary != 0 && primary != tt_code_) {
            for (uint16_t i = 0; i < quiet_count_; ++i) {
              if (quiets_[i] == primary) {
                quiets_[i] = 0;
                return decode_move(primary);
              }
            }
          }
          const uint32_t secondary =
              killers_->secondary[static_cast<std::size_t>(ply_)];
          if (secondary != 0 && secondary != tt_code_) {
            for (uint16_t i = 0; i < quiet_count_; ++i) {
              if (quiets_[i] == secondary) {
                quiets_[i] = 0;
                return decode_move(secondary);
              }
            }
          }
        }
      }
      move_stage = Stage::kPlayCounter;
      continue;

    case Stage::kPlayCounter:
      if (!in_check_) {
        ensure_quiets_generated();

        if (counter_code_ != 0 && counter_code_ != tt_code_) {
          for (uint16_t i = 0; i < quiet_count_; ++i) {
            if (quiets_[i] == counter_code_) {
              quiets_[i] = 0;
              return decode_move(counter_code_);
            }
          }
        }
      }
      move_stage = Stage::kGenQuiets;
      continue;

    case Stage::kGenQuiets:
      // we could also do SEE here
      ensure_quiets_generated();
      ensure_quiets_sorted();
      move_stage = Stage::kPlayQuiets;
      continue;

    case Stage::kPlayQuiets:
      if (!in_check_) {
        while (quiet_index_ < quiet_count_) {
          const uint32_t m = quiets_[quiet_index_++];
          if (m == 0) {
            continue;
          }
          return decode_move(m);
        }
      }
      move_stage = Stage::kPlayBadCaptures;
      continue;

    case Stage::kPlayBadCaptures:
      if (bad_capture_index_ < bad_capture_count_) {
        return decode_move(bad_captures_[bad_capture_index_++]);
      }
      move_stage = Stage::DONE;
      continue;

    case Stage::DONE:
      return Move{};
    }
  }
}

} // namespace chess
