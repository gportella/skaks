#include "chess/moves.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/casteling.hpp"
#include "chess/history.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "chess/zobrist.hpp"

#include <array>
#include <cstdint>

namespace chess {

std::array<uint32_t, kMaxMovementCount> generate_all_moves(const Board& board, SideToMove stm,
                                                           uint16_t& move_count) {
  std::array<uint32_t, kMaxMovementCount> moves{};
  move_count = 0;
  std::cout << "Start with  " << move_count << " moves.\n" << std::endl;
  emit_all_moves(board, stm, moves, move_count);
  std::cout << "Generated " << move_count << " moves.\n" << std::endl;
  return moves;
}

int evaluate_pawn_center_control(const Board& board) {
  // Placeholder for pawn center control evaluation
  for (auto sq :
       {to_index(Square::D4), to_index(Square::D5), to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::wP) {
      return 10; // White controls center
    } else if (piece == OccupancyType::bP) {
      return -10; // Black controls center
    }
  }
  return 0;
}

int evaluate_center_control(const Board& board) {
  // Placeholder for center control evaluation
  for (auto sq :
       {to_index(Square::D4), to_index(Square::D5), to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece != OccupancyType::empty) {
      if (static_cast<std::size_t>(piece) < static_cast<std::size_t>(OccupancyType::bP)) {
        return 5; // White piece controls center
      } else {
        return -5; // Black piece controls center
      }
    }
  }
  return 0;
}

int evaluate_material(const Board& board) {
  // Placeholder for material evaluation
  int material_score = 0;
  for (const auto& occ : board.pieces) {
    switch (occ) {
    case OccupancyType::wP:
      material_score += 100;
      break;
    case OccupancyType::wN:
      material_score += 320;
      break;
    case OccupancyType::wB:
      material_score += 330;
      break;
    case OccupancyType::wR:
      material_score += 500;
      break;
    case OccupancyType::wQ:
      material_score += 900;
      break;
    case OccupancyType::bP:
      material_score -= 100;
      break;
    case OccupancyType::bN:
      material_score -= 320;
      break;
    case OccupancyType::bB:
      material_score -= 330;
      break;
    case OccupancyType::bR:
      material_score -= 500;
      break;
    case OccupancyType::bQ:
      material_score -= 900;
      break;
    default:
      break;
    }
  }
  return material_score;
}

int evaluate_board(const Board& board) {
  // Placeholder for board evaluation function
  int total_eval = 0;
  total_eval += evaluate_pawn_center_control(board);
  total_eval += evaluate_center_control(board);
  total_eval += evaluate_material(board);
  if (board.side_to_move == SideToMove::White) {
    return total_eval; // Evaluation for White
  } else {
    return -total_eval; // Evaluation for Black
  }
}

SearchResult alphabeta_minimax(Board& board, int depth, int alpha, int beta, SideToMove stm) {
  // Placeholder for minimax implementation
  if (depth == 0 || board.is_terminal()) {
    return {evaluate_board(board), Move{}};
  }
  uint16_t move_count = 0;
  auto moves = generate_all_moves(board, stm, move_count);
  SearchResult best = {(stm == SideToMove::White) ? -INF : INF, Move{}};

  for (uint16_t i = 0; i < move_count; ++i) {
    Move move = decode_move(moves[i]);
    Undo undo = make_move(board, move);
    int score = alphabeta_minimax(board, depth - 1, alpha, beta, flip_side(stm)).score;
    // todo
    undo_move(board, undo);

    if (stm == SideToMove::White) {
      if (score > alpha) {
        alpha = score;
        best = {score, move};
      }
      if (alpha >= beta) {
        break;
      }
    } else {
      if (score < beta) {
        beta = score;
        best = {score, move};
      }
      if (beta <= alpha) {
        break;
      }
    }
  }

  return best;
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
  undo.position_key_before = b.position_key;
  undo.fifty_move_counter_before = b.fifty_move_counter;
  undo.en_passant_before = b.en_passant;
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

  // We will first store the previous state in undo
  // Then we will update the board according to the move
  //   TODO Probably not finished
  //    still need to do occupancy
  b.pieces[to_index(m.to)] = m.promo_pc != OccupancyType::empty ? m.promo_pc : m.moving_pc;
  b.pieces[to_index(undo.captured_sq)] = OccupancyType::empty;

  // normal capture, could be en passant or regular capture
  if ((m.promo_pc == OccupancyType::empty) && (m.captured_pc != OccupancyType::empty)) {
    // 0 is the moving piece, 1 is the captured piece or the rook in castling
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.captured_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];

    // bitboard updates
    // remove bit from the captured piece, remove from the "from" square, add to the "to" square
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.captured_pc) - 1], undo.captured_sq);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.to);

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

  }
  // promotion with no capture, 0 is the pawn, 1 is the promoted piece
  else if ((m.promo_pc != OccupancyType::empty) && (m.captured_pc == OccupancyType::empty)) {
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);

  }
  // promotion with capture
  else if (m.promo_pc != OccupancyType::empty) {
    // 0 is the pawn being promoted, 1 capture piece, 2 is the promoted piece
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.captured_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[2] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.captured_pc) - 1], undo.captured_sq);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);
  }
  //
  else {
    // quite move
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.to);
  }

  // Update position key and board state
  //   change to incremental update XOR later
  b.position_key = compute_position_key(b); // Recompute for simplicity
  b.fifty_move_counter = (m.moving_pc == OccupancyType::wP || m.moving_pc == OccupancyType::bP ||
                          m.captured_pc != OccupancyType::empty)
                             ? 0
                             : b.fifty_move_counter + 1;
  b.en_passant = -1; // Reset en passant square
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
  const bool white = (b.side_to_move == SideToMove::White);
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
  b.castling_rights = u.castling_rights_before;
  b.pieces[u.from] = u.moving_pc;

  if (u.was_en_passant) {
    int capSq = (b.side_to_move == SideToMove::White) ? (u.to - 8) : (u.to + 8);
    b.pieces[to_index(capSq)] = u.captured_pc;
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

    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    clear_bit(
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) - 1],
        flag_is_castle(m.flags) ? king_rook : queen_rook);

    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1],
            flag_is_castle(m.flags) ? king_target : queen_target);
    set_bit(
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR : OccupancyType::bR) - 1],
        flag_is_castle(m.flags) ? (rook_kingside_target) : (rook_queenside_target));

  }
  // promotion with no capture, 0 is the pawn, 1 is the promoted piece
  else if ((m.promo_pc != OccupancyType::empty) && (m.captured_pc == OccupancyType::empty)) {
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);

  }
  // promotion with capture
  else if (m.promo_pc != OccupancyType::empty) {
    // 0 is the pawn being promoted, 1 capture piece, 2 is the promoted piece
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.captured_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[2] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.captured_pc) - 1], undo.captured_sq);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);
  }
  //
  else {
    // quite move
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.to);
  }

  // Update position key and board state
  //   change to incremental update XOR later
  b.position_key = compute_position_key(b); // Recompute for simplicity
  b.fifty_move_counter = (m.moving_pc == OccupancyType::wP || m.moving_pc == OccupancyType::bP ||
                          m.captured_pc != OccupancyType::empty)
                             ? 0
                             : b.fifty_move_counter + 1;
  b.en_passant = -1; // Reset en passant square
  update_castling_rights(b, m);

  // Switch side to move
  b.side_to_move = flip_side(b.side_to_move);
}
} // namespace chess