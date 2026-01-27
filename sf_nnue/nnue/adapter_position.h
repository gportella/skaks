// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NNUE_ADAPTER_POSITION_H_INCLUDED
#define NNUE_ADAPTER_POSITION_H_INCLUDED

#include "../position.h"
#include "../types.h"
#include "chess/board.hpp"
#include "chess/types_io.hpp"

namespace Stockfish::Eval::NNUE {

namespace detail {
inline Piece to_sf_piece(chess::OccupancyType occ) {
  switch (occ) {
  case chess::OccupancyType::wP:
    return W_PAWN;
  case chess::OccupancyType::wN:
    return W_KNIGHT;
  case chess::OccupancyType::wB:
    return W_BISHOP;
  case chess::OccupancyType::wR:
    return W_ROOK;
  case chess::OccupancyType::wQ:
    return W_QUEEN;
  case chess::OccupancyType::wK:
    return W_KING;
  case chess::OccupancyType::bP:
    return B_PAWN;
  case chess::OccupancyType::bN:
    return B_KNIGHT;
  case chess::OccupancyType::bB:
    return B_BISHOP;
  case chess::OccupancyType::bR:
    return B_ROOK;
  case chess::OccupancyType::bQ:
    return B_QUEEN;
  case chess::OccupancyType::bK:
    return B_KING;
  case chess::OccupancyType::empty:
    return NO_PIECE;
  }
  return NO_PIECE;
}

inline Color to_sf_color(chess::SideToMove stm) {
  return stm == chess::SideToMove::White ? WHITE : BLACK;
}
} // namespace detail

class AdapterPosition {
public:
  AdapterPosition(const chess::Board* board, StateInfo* state)
      : board_(board), state_(state) {}

  Bitboard pieces() const {
    return board_
               ? static_cast<Bitboard>(
                     board_->occupancy[chess::to_index(chess::PieceColor::Both)])
               : 0ULL;
  }

  Piece piece_on(Square s) const {
    if (!board_)
      return NO_PIECE;
    const auto idx = static_cast<std::size_t>(s);
    return detail::to_sf_piece(board_->pieces[idx]);
  }

  template <PieceType Pt> int count() const {
    if (!board_)
      return 0;
    int total = 0;
    for (int sq = 0; sq < 64; ++sq) {
      const auto occ = board_->pieces[static_cast<std::size_t>(sq)];
      if (occ == chess::OccupancyType::empty)
        continue;
      const int type_index = (static_cast<int>(occ) - 1) % 6;
      const auto pt = static_cast<PieceType>(type_index + 1);
      if constexpr (Pt == ALL_PIECES)
        ++total;
      else if (pt == Pt)
        ++total;
    }
    return total;
  }

  template <PieceType Pt> int count(Color c) const {
    if (!board_)
      return 0;
    int total = 0;
    for (int sq = 0; sq < 64; ++sq) {
      const auto occ = board_->pieces[static_cast<std::size_t>(sq)];
      if (occ == chess::OccupancyType::empty)
        continue;
      const bool is_white_piece = chess::is_white(occ);
      const bool match_color = (c == WHITE) == is_white_piece;
      if (!match_color)
        continue;
      const int type_index = (static_cast<int>(occ) - 1) % 6;
      const auto pt = static_cast<PieceType>(type_index + 1);
      if constexpr (Pt == ALL_PIECES)
        ++total;
      else if (pt == Pt)
        ++total;
    }
    return total;
  }

  template <PieceType Pt> Square square(Color c) const {
    static_assert(Pt == KING, "Only square<KING> is supported");
    if (!board_)
      return SQ_NONE;
    const std::size_t idx = c == WHITE ? 0U : 1U;
    const int sq = board_->king_positions[idx];
    return sq < 0 ? SQ_NONE : static_cast<Square>(sq);
  }

  Color side_to_move() const {
    return board_ ? detail::to_sf_color(board_->side_to_move) : WHITE;
  }

  StateInfo* state() const {
    return state_;
  }

  int rule50_count() const {
    return board_ ? board_->fifty_move_counter : 0;
  }

  Value non_pawn_material(Color c) const {
    if (!board_)
      return 0;
    Value total = 0;
    for (int sq = 0; sq < 64; ++sq) {
      const auto occ = board_->pieces[static_cast<std::size_t>(sq)];
      if (occ == chess::OccupancyType::empty)
        continue;
      const bool is_white_piece = chess::is_white(occ);
      const bool match_color = (c == WHITE) == is_white_piece;
      if (!match_color)
        continue;
      const auto sf_piece = detail::to_sf_piece(occ);
      if (sf_piece == NO_PIECE)
        continue;
      const auto pt = type_of(sf_piece);
      if (pt == PAWN || pt == KING)
        continue;
      total += PieceValue[sf_piece];
    }
    return total;
  }

  Value non_pawn_material() const {
    return non_pawn_material(WHITE) + non_pawn_material(BLACK);
  }

private:
  const chess::Board* board_ = nullptr;
  StateInfo* state_ = nullptr;
};

} // namespace Stockfish::Eval::NNUE

#endif // NNUE_ADAPTER_POSITION_H_INCLUDED
