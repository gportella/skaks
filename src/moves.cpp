#include "chess/moves.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/casteling.hpp"
#include "chess/history.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "chess/zobrist.hpp"

#include <algorithm>
#include <array>

namespace chess {

std::array<uint32_t, kMaxMovementCount> generate_all_moves(const Board& board, SideToMove stm,
                                                           uint16_t& move_count) {
  std::array<uint32_t, kMaxMovementCount> moves{};
  move_count = 0;
  emit_all_moves(board, stm, moves, move_count);
  return moves;
}

std::array<uint32_t, kMaxMovementCount> generate_legal_moves(Board& board, SideToMove stm,
                                                             uint16_t& move_count) {
  std::array<uint32_t, kMaxMovementCount> legal_moves{};
  uint16_t legal_move_count = 0;
  auto pseudo_moves = generate_all_moves(board, stm, move_count);

  for (uint16_t i = 0; i < move_count; ++i) {
    Move m = decode_move(pseudo_moves[i]);
    Undo u = make_move(board, m);
    if (!is_check(board, stm)) {
      legal_moves[legal_move_count++] = pseudo_moves[i];
    }
    undo_move(board, u);
  }
  move_count = legal_move_count;
  return legal_moves;
}

inline Undo make_move(Board& b, const Move& m) {
  const bool white = (b.side_to_move == SideToMove::White);
  const Square king_rook = kCastlingSideConfigs[to_index(b.side_to_move)].rook_kingside_start;
  const Square queen_rook = kCastlingSideConfigs[to_index(b.side_to_move)].rook_queenside_start;
  const Square king_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].king_kingside_target; // used for short castle
  const Square queen_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].king_queenside_target; // used for long castle
  const Square rook_kingside_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_kingside_target; // used for short castle
  const Square rook_queenside_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_queenside_target;
  Undo undo;
  if (b.side_to_move == SideToMove::Black) {
    ++b.ply_count;
  }
  undo.position_key_before = b.position_key;
  undo.fifty_move_counter_before = b.fifty_move_counter;
  undo.en_passant_before = b.en_passant;
  undo.ep_square_before = b.ep_square;
  undo.captured_pc = m.captured_pc;
  undo.castling_rights_before = b.castling_rights;
  undo.moving_pc = m.moving_pc;
  undo.promo_pc = m.promo_pc;
  undo.was_en_passant = flag_is_ep(m.flags);
  undo.was_castling = flag_is_castle(m.flags) || flag_is_long_castle(m.flags);
  undo.from = m.from;
  undo.to = m.to;
  undo.flags = m.flags;
  undo.captured_sq = undo.was_en_passant ? (white ? (m.to - 8) : (m.to + 8)) : m.to;
  if (undo.was_en_passant) {
    undo.captured_pc = white ? OccupancyType::bP : OccupancyType::wP;
  }
  std::copy(std::begin(b.occupancy), std::end(b.occupancy), std::begin(undo.occupancy));
  undo.rook_list_before = {b.rook_list[0], b.rook_list[1]};
  undo.king_list_before = {b.king_list[0], b.king_list[1]};
  undo.pawn_list_before = {b.pawn_list[0], b.pawn_list[1]};
  undo.king_positions_before = b.king_positions;
  undo.castled_before = b.has_castled;
  undo.king_captured_before = b.king_captured;

  const auto mover_index = to_index(b.side_to_move);
  const auto enemy_index = to_index(flip_side(b.side_to_move));
  const Square from_sq = static_cast<Square>(m.from);
  const Square to_sq = static_cast<Square>(m.to);
  const auto from_idx = to_index(m.from);
  const auto to_idx = to_index(m.to);
  const auto captured_idx = to_index(undo.captured_sq);

  b.pieces[from_idx] = OccupancyType::empty;
  if (undo.was_en_passant) {
    b.pieces[captured_idx] = OccupancyType::empty;
  }
  b.pieces[to_idx] = (m.promo_pc != OccupancyType::empty) &&
                             (m.moving_pc == (white ? OccupancyType::wP : OccupancyType::bP))
                         ? m.promo_pc
                         : m.moving_pc;

  auto update_piece_square = [](PieceList& list, Square from, Square to) {
    for (std::uint8_t i = 0; i < list.count; ++i) {
      if (list.squares[i] == from) {
        list.squares[i] = to;
        return;
      }
    }
  };

  auto remove_piece_square = [](PieceList& list, Square sq) {
    for (std::uint8_t i = 0; i < list.count; ++i) {
      if (list.squares[i] == sq) {
        list.squares[i] = list.squares[list.count - 1];
        --list.count;
        return;
      }
    }
  };

  auto add_piece_square = [](PieceList& list, Square sq) {
    if (list.count < list.squares.size()) {
      list.squares[list.count++] = sq;
    }
  };

  if (m.moving_pc == OccupancyType::wK || m.moving_pc == OccupancyType::bK) {
    b.king_positions[mover_index] = m.to;
    update_piece_square(b.king_list[mover_index], from_sq, to_sq);
  }
  if (m.moving_pc == OccupancyType::wR || m.moving_pc == OccupancyType::bR) {
    update_piece_square(b.rook_list[mover_index], from_sq, to_sq);
  }
  if (m.moving_pc == OccupancyType::wP || m.moving_pc == OccupancyType::bP) {
    if (m.promo_pc == OccupancyType::empty) {
      update_piece_square(b.pawn_list[mover_index], from_sq, to_sq);
    }
  }

  // normal capture, could be en passant or regular capture
  if ((m.promo_pc == OccupancyType::empty) && (undo.captured_pc != OccupancyType::empty)) {
    // 0 is the moving piece, 1 is the captured piece or the rook in castling
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];

    // bitboard updates
    // remove bit from the captured piece, remove from the "from" square, add to the "to" square
    clear_bit(b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1], undo.captured_sq);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.to);

    remove_piece_square(b.rook_list[enemy_index], static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.pawn_list[enemy_index], static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.king_list[enemy_index], static_cast<Square>(undo.captured_sq));

    if (undo.captured_pc == OccupancyType::wK || undo.captured_pc == OccupancyType::bK) {
      b.king_positions[enemy_index] = -1;
      b.king_captured =
          (undo.captured_pc == OccupancyType::wK) ? PieceColor::White : PieceColor::Black;
    }

  }
  // castling 0 is king, 1 is the rook
  else if ((flag_is_castle(m.flags) || flag_is_long_castle(m.flags)) &&
           (m.captured_pc == OccupancyType::empty)) // redundant check but ok...
  {
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[1] =
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) -
                    1]; // Assuming rook is involved

    // bitboard updates
    // remove king and rook from their original squares, add to their new squares
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    clear_bit(
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) - 1],
        flag_is_castle(m.flags) ? king_rook : queen_rook);

    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1],
            flag_is_castle(m.flags) ? king_target : queen_target);
    set_bit(
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) - 1],
        flag_is_castle(m.flags) ? (rook_kingside_target) : (rook_queenside_target));

    const Square rook_from = flag_is_castle(m.flags) ? king_rook : queen_rook;
    const Square rook_to = flag_is_castle(m.flags) ? rook_kingside_target : rook_queenside_target;
    b.pieces[to_index(rook_from)] = OccupancyType::empty;
    b.pieces[to_index(rook_to)] = white ? OccupancyType::wR : OccupancyType::bR;

    // update rook list to reflect the single rook that moved
    update_piece_square(b.rook_list[mover_index], rook_from, rook_to);

    b.has_castled[mover_index] = true;

  }
  // promotion with no capture, 0 is the pawn, 1 is the promoted piece
  else if ((m.promo_pc != OccupancyType::empty) && (m.captured_pc == OccupancyType::empty)) {
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);

    remove_piece_square(b.pawn_list[mover_index], from_sq);
    if (m.promo_pc == OccupancyType::wR || m.promo_pc == OccupancyType::bR) {
      add_piece_square(b.rook_list[mover_index], to_sq);
    }

  }
  // promotion with capture
  else if (m.promo_pc != OccupancyType::empty) {
    // 0 is the pawn being promoted, 1 capture piece, 2 is the promoted piece
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[2] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1], undo.captured_sq);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);

    remove_piece_square(b.pawn_list[mover_index], from_sq);
    remove_piece_square(b.rook_list[enemy_index], static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.pawn_list[enemy_index], static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.king_list[enemy_index], static_cast<Square>(undo.captured_sq));
    if (undo.captured_pc == OccupancyType::wK || undo.captured_pc == OccupancyType::bK) {
      b.king_positions[enemy_index] = -1;
      b.king_captured =
          (undo.captured_pc == OccupancyType::wK) ? PieceColor::White : PieceColor::Black;
    }
    if (m.promo_pc == OccupancyType::wR || m.promo_pc == OccupancyType::bR) {
      add_piece_square(b.rook_list[mover_index], to_sq);
    }
  }
  //
  else {
    // quite move
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.to);

    b.occupancy[mover_index] &= ~bit_mask(m.from);
    b.occupancy[mover_index] |= bit_mask(m.to);
  }

  b.occupancy[to_index(PieceColor::White)] = calculate_occupancy(b, PieceColor::White);
  b.occupancy[to_index(PieceColor::Black)] = calculate_occupancy(b, PieceColor::Black);
  b.occupancy[to_index(PieceColor::Both)] =
      b.occupancy[to_index(PieceColor::White)] | b.occupancy[to_index(PieceColor::Black)];

  // Update position key and board state
  //   change to incremental update XOR later
  b.position_key = compute_position_key(b); // Recompute for simplicity

  b.fifty_move_counter = (m.moving_pc == OccupancyType::wP || m.moving_pc == OccupancyType::bP ||
                          m.captured_pc != OccupancyType::empty)
                             ? 0
                             : b.fifty_move_counter + 1;
  b.en_passant = -1; // Reset en passant square
  b.ep_square = 0;
  update_castling_rights(b, m);

  // Switch side to move
  b.side_to_move = flip_side(b.side_to_move);

  return undo;
}

int update_castling_rights(Board& b, const Move& m) {
  int mask = to_mask(b.castling_rights);

  // 1) King moves: lose both rights for that side
  if (m.moving_pc == OccupancyType::wK) {
    mask &= ~to_mask(CastlingRights::WhiteKingside);
    mask &= ~to_mask(CastlingRights::WhiteQueenside);
  }
  if (m.moving_pc == OccupancyType::bK) {
    mask &= ~to_mask(CastlingRights::BlackKingside);
    mask &= ~to_mask(CastlingRights::BlackQueenside);
  }

  // 2) Rook moves from home squares: lose that side's relevant right
  if (m.moving_pc == OccupancyType::wR) {
    if (m.from == to_index(Square::H1))
      mask &= ~to_mask(CastlingRights::WhiteKingside);
    if (m.from == to_index(Square::A1))
      mask &= ~to_mask(CastlingRights::WhiteQueenside);
  }
  if (m.moving_pc == OccupancyType::bR) {
    if (m.from == to_index(Square::H8))
      mask &= ~to_mask(CastlingRights::BlackKingside);
    if (m.from == to_index(Square::A8))
      mask &= ~to_mask(CastlingRights::BlackQueenside);
  }

  // (EP cannot capture a rook, so no special-case needed)
  if (m.captured_pc == OccupancyType::wR) {
    if (m.to == to_index(Square::H1))
      mask &= ~to_mask(CastlingRights::WhiteKingside);
    if (m.to == to_index(Square::A1))
      mask &= ~to_mask(CastlingRights::WhiteQueenside);
  }
  if (m.captured_pc == OccupancyType::bR) {
    if (m.to == to_index(Square::H8))
      mask &= ~to_mask(CastlingRights::BlackKingside);
    if (m.to == to_index(Square::A8))
      mask &= ~to_mask(CastlingRights::BlackQueenside);
  }

  b.castling_rights = from_mask(mask);
  return mask;
}

inline void undo_move(Board& b, const Undo& u) {
  // Revert the move on the board (not fully implemented here)
  // ...

  // Switch side to move back
  b.side_to_move = flip_side(b.side_to_move);
  if (b.side_to_move == SideToMove::Black) {
    --b.ply_count;
  }
  const bool white = (b.side_to_move == SideToMove::White);
  const Square king_rook = kCastlingSideConfigs[to_index(b.side_to_move)].rook_kingside_start;
  const Square queen_rook = kCastlingSideConfigs[to_index(b.side_to_move)].rook_queenside_start;
  const Square king_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].king_kingside_target; // used for short castle
  const Square queen_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].king_queenside_target; // used for long castle
  const Square rook_kingside_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_kingside_target; // used for short castle
  const Square rook_queenside_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_queenside_target;

  // Restore position key and board state
  b.position_key = u.position_key_before;
  b.fifty_move_counter = u.fifty_move_counter_before;
  b.en_passant = u.en_passant_before;
  b.ep_square = u.ep_square_before;
  b.castling_rights = u.castling_rights_before;
  b.has_castled = u.castled_before;
  std::copy(std::begin(u.occupancy), std::end(u.occupancy), std::begin(b.occupancy));
  b.rook_list[0] = u.rook_list_before[0];
  b.rook_list[1] = u.rook_list_before[1];
  b.king_list[0] = u.king_list_before[0];
  b.king_list[1] = u.king_list_before[1];
  b.pawn_list[0] = u.pawn_list_before[0];
  b.pawn_list[1] = u.pawn_list_before[1];
  b.king_positions = u.king_positions_before;
  b.king_captured = u.king_captured_before;
  if (u.moving_pc != OccupancyType::empty) {
    b.pieces[u.from] = u.moving_pc;
  }

  if (u.was_en_passant) {
    b.pieces[u.captured_sq] = u.captured_pc;
    b.pieces[u.to] = OccupancyType::empty;
  } else {
    b.pieces[u.to] = u.captured_pc;
  }
  // bring back bitboards
  // normal capture, could be en passant or regular capture
  if ((u.promo_pc == OccupancyType::empty) && (u.captured_pc != OccupancyType::empty)) {
    // 0 is the moving piece, 1 is the captured piece or the rook in castling

    b.pieces_bb[static_cast<std::size_t>(u.captured_pc) - 1] = u.pieces_bb[0];
    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[1];

    // bitboard updates
    // set bit to the from, remove from the "to" square, add back to the captured square
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.to);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.captured_pc) - 1], u.captured_sq);

  }
  // castling 0 is king, 1 is the rook
  else if ((flag_is_castle(u.flags) || flag_is_long_castle(u.flags)) &&
           (u.captured_pc == OccupancyType::empty)) // redundant check but ok...
  {

    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[0];
    b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) - 1] =
        u.pieces_bb[1];

    // bitboard updates
    // set king and rook back to their original squares, remove from their new squares
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1],
              flag_is_castle(u.flags) ? king_target : queen_target);
    clear_bit(
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) - 1],
        flag_is_castle(u.flags) ? rook_kingside_target : rook_queenside_target);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
    set_bit(
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) - 1],
        flag_is_castle(u.flags) ? king_rook : queen_rook);

    const Square rook_from = flag_is_castle(u.flags) ? king_rook : queen_rook;
    const Square rook_to = flag_is_castle(u.flags) ? rook_kingside_target : rook_queenside_target;
    b.pieces[to_index(rook_to)] = OccupancyType::empty;
    b.pieces[to_index(rook_from)] = white ? OccupancyType::wR : OccupancyType::bR;

  }
  // promotion with no capture, 0 is the pawn, 1 is the promoted piece
  else if ((u.promo_pc != OccupancyType::empty) && (u.captured_pc == OccupancyType::empty)) {
    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[0];
    b.pieces_bb[static_cast<std::size_t>(u.promo_pc) - 1] = u.pieces_bb[1];

    // reset promoted piece back to a pawn
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.promo_pc) - 1], u.to);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);

  }
  // promotion with capture
  else if (u.promo_pc != OccupancyType::empty) {
    // 0 is the pawn being promoted, 1 capture piece, 2 is the promoted piece
    b.pieces_bb[static_cast<std::size_t>(u.captured_pc) - 1] = u.pieces_bb[0];
    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[1];
    b.pieces_bb[static_cast<std::size_t>(u.promo_pc) - 1] = u.pieces_bb[2];

    // bitboard updates
    // set bit to the from pawn, set bit to the captured piece, remove promoted piece from to
    // square
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.promo_pc) - 1], u.to);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.captured_pc) - 1], u.captured_sq);
  }
  //
  else {
    // quite move
    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[0];
    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.to);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
  }

  // Update position key and board state
  //   change to incremental update XOR later
  b.position_key = compute_position_key(b); // Recompute for simplicity
}
} // namespace chess