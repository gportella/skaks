// SPDX-License-Identifier: GPL-3.0-or-later
#include "chess/casteling.hpp"
#include "chess/nnue_sf.hpp"
#include "chess/types_io.hpp"
#include "nnue/adapter_position.h"
#include "nnue/evaluate_nnue.h"
#include "probe.h"

#include <algorithm>
#include <cmath>
#include <deque>

namespace chess {
namespace {

Stockfish::Piece to_sf_piece(OccupancyType occ) {
  switch (occ) {
  case OccupancyType::wP:
    return Stockfish::W_PAWN;
  case OccupancyType::wN:
    return Stockfish::W_KNIGHT;
  case OccupancyType::wB:
    return Stockfish::W_BISHOP;
  case OccupancyType::wR:
    return Stockfish::W_ROOK;
  case OccupancyType::wQ:
    return Stockfish::W_QUEEN;
  case OccupancyType::wK:
    return Stockfish::W_KING;
  case OccupancyType::bP:
    return Stockfish::B_PAWN;
  case OccupancyType::bN:
    return Stockfish::B_KNIGHT;
  case OccupancyType::bB:
    return Stockfish::B_BISHOP;
  case OccupancyType::bR:
    return Stockfish::B_ROOK;
  case OccupancyType::bQ:
    return Stockfish::B_QUEEN;
  case OccupancyType::bK:
    return Stockfish::B_KING;
  case OccupancyType::empty:
    return Stockfish::NO_PIECE;
  }
  return Stockfish::NO_PIECE;
}

/// Builds a Stockfish::DirtyPiece describing all piece movements/removals for a
/// move.
///
/// Populates the dirty piece entries for special moves (castling, promotion, en
/// passant) and captures, mapping from the engine's move representation to
/// Stockfish squares/pieces. For castling, both king and rook are added. For
/// promotions, the pawn is removed and the promoted piece is added (and captured
/// piece removed if applicable). For en passant, the captured pawn is removed
/// from its actual square.
///
/// @param move The move to translate into dirty piece updates.
/// @return A Stockfish::DirtyPiece containing the affected pieces and their
/// from/to squares.
Stockfish::DirtyPiece build_dirty_piece(const Move& move) {
  Stockfish::DirtyPiece dp{};
  auto add = [&](Stockfish::Piece piece, Stockfish::Square from,
                 Stockfish::Square to) {
    const int idx = dp.dirty_num++;
    dp.piece[idx] = piece;
    dp.from[idx] = from;
    dp.to[idx] = to;
  };

  const bool is_ep = flag_is_ep(move.flags);
  const bool is_promo = move.promo_pc != OccupancyType::empty;
  const bool is_castle =
      flag_is_castle(move.flags) || flag_is_long_castle(move.flags);
  const bool is_white_move = is_white(move.moving_pc);

  const auto from_sq = static_cast<Stockfish::Square>(move.from);
  const auto to_sq = static_cast<Stockfish::Square>(move.to);
  const auto moving_piece = to_sf_piece(move.moving_pc);

  if (is_castle) {
    add(moving_piece, from_sq, to_sq);
    const auto side = is_white_move ? SideToMove::White : SideToMove::Black;
    const auto& cfg = kCastlingSideConfigs[to_index(side)];
    const bool is_kingside = flag_is_castle(move.flags);
    const auto rook_from = static_cast<Stockfish::Square>(static_cast<int>(
        is_kingside ? cfg.rook_kingside_start : cfg.rook_queenside_start));
    const auto rook_to = static_cast<Stockfish::Square>(static_cast<int>(
        is_kingside ? cfg.rook_kingside_target : cfg.rook_queenside_target));
    const auto rook_piece =
        is_white_move ? Stockfish::W_ROOK : Stockfish::B_ROOK;
    add(rook_piece, rook_from, rook_to);
    return dp;
  }

  if (is_promo) {
    const auto pawn_piece =
        is_white_move ? Stockfish::W_PAWN : Stockfish::B_PAWN;
    const auto promo_piece = to_sf_piece(move.promo_pc);
    if (move.captured_pc != OccupancyType::empty) {
      const auto captured_piece = to_sf_piece(move.captured_pc);
      add(captured_piece, to_sq, Stockfish::SQ_NONE);
    }
    add(pawn_piece, from_sq, Stockfish::SQ_NONE);
    add(promo_piece, Stockfish::SQ_NONE, to_sq);
    return dp;
  }

  add(moving_piece, from_sq, to_sq);

  if (is_ep) {
    const int captured_sq = is_white_move ? (move.to - 8) : (move.to + 8);
    const auto captured_piece =
        is_white_move ? Stockfish::B_PAWN : Stockfish::W_PAWN;
    add(captured_piece, static_cast<Stockfish::Square>(captured_sq),
        Stockfish::SQ_NONE);
  } else if (move.captured_pc != OccupancyType::empty) {
    const auto captured_piece = to_sf_piece(move.captured_pc);
    add(captured_piece, to_sq, Stockfish::SQ_NONE);
  }

  return dp;
}

int simple_eval_adapter(const Stockfish::Eval::NNUE::AdapterPosition& pos,
                        Stockfish::Color c) {
  return Stockfish::PawnValue * (pos.template count<Stockfish::PAWN>(c) -
                                 pos.template count<Stockfish::PAWN>(~c)) +
         (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}

} // namespace

struct NnueAdapter::Impl {
  const Board* board = nullptr;
  std::deque<Stockfish::StateInfo> states{};
  Stockfish::StateInfo* current = nullptr;

  void reset(const Board& b) {
    board = &b;
    states.clear();
    states.emplace_back(Stockfish::StateInfo{});
    current = &states.back();
    current->previous = nullptr;
    current->dirtyPiece.dirty_num = 0;
  }

  void push_move(const Move& move) {
    if (!board)
      return;
    states.emplace_back(Stockfish::StateInfo{});
    auto* st = &states.back();
    st->previous = current;
    st->dirtyPiece = build_dirty_piece(move);
    current = st;
  }

  void push_null() {
    if (!board)
      return;
    states.emplace_back(Stockfish::StateInfo{});
    auto* st = &states.back();
    st->previous = current;
    st->dirtyPiece.dirty_num = 0;
    current = st;
  }

  void pop_move() {
    if (states.size() <= 1)
      return;
    states.pop_back();
    current = &states.back();
  }

  void pop_null() {
    pop_move();
  }
};

NnueAdapter::NnueAdapter() : impl_(std::make_unique<Impl>()) {}

NnueAdapter::NnueAdapter(const Board& board) : impl_(std::make_unique<Impl>()) {
  impl_->reset(board);
}

NnueAdapter::~NnueAdapter() = default;

NnueAdapter::NnueAdapter(NnueAdapter&&) noexcept = default;
NnueAdapter& NnueAdapter::operator=(NnueAdapter&&) noexcept = default;

void NnueAdapter::reset(const Board& board) {
  impl_->reset(board);
}

void NnueAdapter::push_move(const Move& move) {
  impl_->push_move(move);
}

void NnueAdapter::push_null() {
  impl_->push_null();
}

void NnueAdapter::pop_move() {
  impl_->pop_move();
}

void NnueAdapter::pop_null() {
  impl_->pop_null();
}

/// Evaluates the current position using Stockfish NNUE in incremental mode,
/// blending NNUE output with a simple evaluation and material/shuffling factors.
///
/// @param adapter   Adapter containing the board and current state.
/// @param adjusted  Whether to use adjusted NNUE evaluation.
/// @param complexity Optional output pointer to receive NNUE complexity.
/// @return Signed evaluation score from the side-to-move perspective.
int evaluate_nnue_stockfish_incremental(const NnueAdapter& adapter,
                                        bool adjusted, int* complexity) {
  if (!adapter.impl_ || !adapter.impl_->board || !adapter.impl_->current) {
    if (complexity)
      *complexity = 0;
    return 0;
  }

  Stockfish::Eval::NNUE::AdapterPosition pos(adapter.impl_->board,
                                             adapter.impl_->current);
  const auto stm = pos.side_to_move();
  const int simple_eval = simple_eval_adapter(pos, stm);
  const bool small_net = std::abs(simple_eval) > 1050;

  int nnue_complexity = 0;
  const auto net_size =
      small_net ? Stockfish::Eval::NNUE::Small : Stockfish::Eval::NNUE::Big;
  Stockfish::Value nnue = Stockfish::Eval::NNUE::evaluate_adapter(
      pos, net_size, adjusted, &nnue_complexity);

  if (complexity)
    *complexity = nnue_complexity;

  nnue -= nnue * (nnue_complexity + std::abs(simple_eval - nnue)) / 32768;

  const int npm = static_cast<int>(pos.non_pawn_material()) / 64;
  int v =
      (nnue * (915 + npm + 9 * pos.template count<Stockfish::PAWN>())) / 1024;

  const int shuffling = pos.rule50_count();
  v = v * (200 - shuffling) / 214;

  v = std::clamp(v, Stockfish::VALUE_TB_LOSS_IN_MAX_PLY + 1,
                 Stockfish::VALUE_TB_WIN_IN_MAX_PLY - 1);

  if (stm == Stockfish::BLACK)
    v = -v;
  return v;
}

} // namespace chess
