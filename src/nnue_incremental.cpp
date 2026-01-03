#include "chess/nnue_incremental.hpp"

#include "chess/board.hpp"
#include "chess/casteling.hpp"
#include "chess/moves.hpp"
#include "chess/nnue.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "sf_nnue/nnue.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace chess {
namespace {

constexpr int kInvalidSquare = 64;

inline bool is_king_piece(OccupancyType occ) {
  return occ == OccupancyType::wK || occ == OccupancyType::bK;
}

struct DirtyEntry {
  int pc = 0;
  int from = kInvalidSquare;
  int to = kInvalidSquare;
};

void clear_dirty_piece(DirtyPiece& dp) {
  dp.dirtyNum = 0;
  for (int i = 0; i < 3; ++i) {
    dp.pc[i] = 0;
    dp.from[i] = kInvalidSquare;
    dp.to[i] = kInvalidSquare;
  }
}

void clear_accumulator(Accumulator& acc) {
  for (int perspective = 0; perspective < 2; ++perspective) {
    for (int i = 0; i < 256; ++i) {
      acc.accumulation[perspective][i] = 0;
    }
  }
  acc.computedAccumulation = 0;
}

void clear_entry_impl(NNUEdata& data) {
  clear_accumulator(data.accumulator);
  clear_dirty_piece(data.dirtyPiece);
}

DirtyEntry make_piece_entry(OccupancyType occ, int from, int to) {
  DirtyEntry entry;
  entry.pc = sf_nnue_piece_code(occ);
  entry.from = from;
  entry.to = to;
  return entry;
}

OccupancyType rook_for_side(SideToMove side) {
  return (side == SideToMove::White) ? OccupancyType::wR : OccupancyType::bR;
}

} // namespace

SfNnueStack::SfNnueStack() {
  reset();
}

void SfNnueStack::reset() {
  top_ = 0;
  for (std::size_t i = 0; i < stack_.size(); ++i) {
    clear_entry_impl(stack_[i]);
    force_refresh_[i] = false;
  }
}

void SfNnueStack::push_move(const Board& board, const Move& move,
                            const Undo& undo) {
  if (top_ + 1 >= stack_.size()) {
    throw std::runtime_error("NNUE stack overflow");
  }
  ++top_;
  clear_entry_impl(stack_[top_]);
  force_refresh_[top_] = false;
  DirtyPiece& dirty = stack_[top_].dirtyPiece;

  std::array<DirtyEntry, 3> entries{};
  std::size_t count = 0;

  const bool is_promotion = move.promo_pc != OccupancyType::empty;
  const bool is_capture = undo.captured_pc != OccupancyType::empty;
  const bool is_castle = undo.was_castling;

  const int from_sq = static_cast<int>(move.from);
  const int to_sq = static_cast<int>(move.to);

  const SideToMove mover = flip_side(board.side_to_move);

  auto push_entry = [&](DirtyEntry entry) {
    if (count < entries.size() && entry.pc != 0) {
      entries[count++] = entry;
    }
  };

  if (is_king_piece(move.moving_pc)) {
    push_entry(make_piece_entry(move.moving_pc, from_sq, to_sq));
  }

  if (is_promotion) {
    push_entry(make_piece_entry(move.moving_pc, from_sq, kInvalidSquare));
    push_entry(make_piece_entry(move.promo_pc, kInvalidSquare, to_sq));
  } else if (!is_king_piece(move.moving_pc)) {
    push_entry(make_piece_entry(move.moving_pc, from_sq, to_sq));
  }

  if (is_capture) {
    const int captured_sq = static_cast<int>(undo.captured_sq);
    push_entry(make_piece_entry(undo.captured_pc, captured_sq, kInvalidSquare));
  }

  if (is_castle) {
    const auto& cfg = kCastlingSideConfigs[to_index(mover)];
    const bool kingside = flag_is_castle(move.flags);
    const Square rook_from =
        kingside ? cfg.rook_kingside_start : cfg.rook_queenside_start;
    const Square rook_to =
        kingside ? cfg.rook_kingside_target : cfg.rook_queenside_target;
    const int rook_from_idx = static_cast<int>(to_index(rook_from));
    const int rook_to_idx = static_cast<int>(to_index(rook_to));
    push_entry(
        make_piece_entry(rook_for_side(mover), rook_from_idx, rook_to_idx));
  }

  dirty.dirtyNum = static_cast<int>(count);
  for (std::size_t i = 0; i < count; ++i) {
    dirty.pc[i] = entries[i].pc;
    dirty.from[i] = entries[i].from;
    dirty.to[i] = entries[i].to;
  }
  for (std::size_t i = count; i < entries.size(); ++i) {
    dirty.pc[i] = 0;
    dirty.from[i] = kInvalidSquare;
    dirty.to[i] = kInvalidSquare;
  }
}

void SfNnueStack::push_null() {
  if (top_ + 1 >= stack_.size()) {
    throw std::runtime_error("NNUE stack overflow");
  }
  ++top_;
  clear_entry_impl(stack_[top_]);
  force_refresh_[top_] = true;
}

void SfNnueStack::pop() {
  if (top_ == 0) {
    clear_entry_impl(stack_[0]);
    force_refresh_[0] = false;
    top_ = 0;
    return;
  }
  clear_entry_impl(stack_[top_]);
  force_refresh_[top_] = false;
  --top_;
}

int SfNnueStack::depth() const {
  return static_cast<int>(top_);
}

NNUEdata& SfNnueStack::current_data() {
  return stack_[top_];
}

const NNUEdata& SfNnueStack::current_data() const {
  return stack_[top_];
}

std::array<NNUEdata*, 3> SfNnueStack::pointer_triplet() {
  NNUEdata* current = &stack_[top_];
  NNUEdata* prev = current;
  NNUEdata* prev_prev = current;
  if (top_ > 0) {
    prev = &stack_[top_ - 1];
    prev_prev = (top_ > 1) ? &stack_[top_ - 2] : prev;
  }
  if (force_refresh_[top_]) {
    force_refresh_[top_] = false;
    clear_entry_impl(*current);
    prev = current;
    prev_prev = current;
  }
  return {current, prev, prev_prev};
}

namespace {
thread_local SfNnueStack* g_tls_nnue_stack = nullptr;
} // namespace

SfNnueStack* current_thread_nnue_stack() {
  return g_tls_nnue_stack;
}

void set_thread_nnue_stack(SfNnueStack* stack) {
  g_tls_nnue_stack = stack;
}

ScopedNnueThreadContext::ScopedNnueThreadContext(SfNnueStack* stack) {
  previous_ = g_tls_nnue_stack;
  g_tls_nnue_stack = stack;
}

ScopedNnueThreadContext::~ScopedNnueThreadContext() {
  g_tls_nnue_stack = previous_;
}

} // namespace chess
