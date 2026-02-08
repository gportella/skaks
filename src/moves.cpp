#include "chess/moves.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/casteling.hpp"
#include "chess/exchange.hpp"
#include "chess/history.hpp"
#include "chess/move_ordering.hpp"
#include "chess/piece_values.hpp"
#include "chess/pins.hpp"
#include "chess/pst_tables.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "chess/zobrist.hpp"

#include <algorithm>
#include <array>
#include <iostream>

namespace chess {

struct MoveKey {
  uint32_t code;
  int key;        // higher is better
  uint16_t order; // original emission order for deterministic ties
};

inline int killer_priority(uint32_t code, const KillerTable* killers, int ply) {
  if (killers == nullptr || ply < 0 || ply >= MAX_PLY) {
    return 0;
  }

  const uint32_t primary = killers->primary[static_cast<std::size_t>(ply)];
  if (code == primary) {
    return 40'000;
  }
  const uint32_t secondary = killers->secondary[static_cast<std::size_t>(ply)];
  if (code == secondary) {
    return 20'000;
  }
  return 0;
}

inline int mvv_lva_score(OccupancyType captured, OccupancyType piece) {
  static const int scores[13] = {0,   100, 320, 330, 500, 900,  20000,
                                 100, 320, 330, 500, 900, 20000};
  return scores[static_cast<size_t>(captured)] * 10 -
         scores[static_cast<size_t>(piece)];
}

// Order: ttMove first, then captures by MVV-LVA + SEE, then quiets (killer
// quiets boosted)
void sort_moves(const Board& board,
                std::array<uint32_t, kMaxMovementCount>& moves,
                uint16_t move_count, uint32_t tt_code,
                const KillerTable* killers,
                const std::array<std::array<int, 64>, 64>* history_heuristic,
                int ply, uint32_t counter_move_code,
                const ContinuationHistoryTable* continuation_heuristic,
                const Move* parent_move) {
  std::array<MoveKey, kMaxMovementCount> keys;

  for (uint16_t i = 0; i < move_count; ++i) {
    uint32_t m = moves[i];
    int key = 0;

    if (tt_code != 0 && m == tt_code) {
      key = 2'000'000; // force first
    } else {
      const uint16_t cap_raw = move_captured(m);
      const bool is_capture =
          cap_raw != static_cast<uint8_t>(OccupancyType::empty);
      if (is_capture) {
        auto cap = static_cast<OccupancyType>(cap_raw);
        auto pc = static_cast<OccupancyType>(move_piece(m));
        const int capture_score = mvv_lva_score(cap, pc);
        const int see_gain = static_exchange_eval(board, decode_move(m));
        if (see_gain <= 0) {
          key = 80'000 + std::min(capture_score, 15'000);
        } else {
          key = 1'000'000 + capture_score + see_gain * 100;
        }
      } else {
        key = 100'000;
        key += killer_priority(m, killers,
                               ply); // quiet killers ahead of other quiets
        if (history_heuristic != nullptr) {
          const auto from = static_cast<std::size_t>(move_from(m));
          const auto to = static_cast<std::size_t>(move_to(m));
          key += (*history_heuristic)[from][to];
        }
        if (continuation_heuristic != nullptr && parent_move != nullptr &&
            parent_move->moving_pc != OccupancyType::empty) {
          key += continuation_heuristic->score(*parent_move, decode_move(m));
        }
        if (counter_move_code != 0 && m == counter_move_code) {
          key += 30'000;
        }
      }
    }

    keys[i] = {m, key, i};
  }

  std::sort(keys.begin(), keys.begin() + move_count,
            [](const MoveKey& a, const MoveKey& b) {
              if (a.key == b.key) {
                return a.order < b.order;
              }
              return a.key > b.key;
            });

  for (uint16_t i = 0; i < move_count; ++i)
    moves[i] = keys[i].code;
}

std::array<uint32_t, kMaxMovementCount>
generate_all_moves(const Board& board, SideToMove stm, uint16_t& move_count,
                   MoveGenType::Type type) {
  std::array<uint32_t, kMaxMovementCount> moves{};
  move_count = 0;
  emit_all_moves(board, stm, moves, move_count, type);
  return moves;
}

std::array<uint32_t, kMaxMovementCount>
generate_legal_moves(Board& board, SideToMove stm, uint16_t& move_count,
                     MoveGenType::Type type) {
  std::array<uint32_t, kMaxMovementCount> legal_moves{};
  uint16_t legal_move_count = 0;
  auto pseudo_moves = generate_all_moves(board, stm, move_count, type);
  const bool in_check = is_check(board, stm);
  PinnedMapByPiece pinned_map{};
  bool use_pin_fastpath = false;
  if (!in_check) {
    const char* enable_pin = std::getenv("SKAKS_ENABLE_PIN_FASTPATH");
    const auto side_idx = to_index(stm);
    if (enable_pin && *enable_pin && board.king_list[side_idx].count > 0 &&
        board.king_positions[side_idx] >= 0) {
      build_pinned_map_into(board, stm, pinned_map);
      use_pin_fastpath = true;
    }
  }

  for (uint16_t i = 0; i < move_count; ++i) {
    Move m = decode_move(pseudo_moves[i]);
    if (use_pin_fastpath) {
      const OccupancyType king_piece =
          (stm == SideToMove::White) ? OccupancyType::wK : OccupancyType::bK;
      const bool is_king_move = (m.moving_pc == king_piece);
      const bool is_ep = flag_is_ep(m.flags);
      if (!is_king_move && !is_ep) {
        const auto side_idx = to_index(stm);
        const int king_sq = board.king_positions[side_idx];
        if (king_sq >= 0) {
          const Bitboard from_mask = Bitboard(1) << m.from;
          const auto& rook_rays = ROOK_RAYS[static_cast<std::size_t>(king_sq)];
          const auto& bishop_rays =
              BISHOP_RAYS[static_cast<std::size_t>(king_sq)];
          const Bitboard rook_lines = rook_rays.north | rook_rays.south |
                                      rook_rays.east | rook_rays.west;
          const Bitboard bishop_lines =
              bishop_rays.northeast | bishop_rays.northwest |
              bishop_rays.southeast | bishop_rays.southwest;
          if ((from_mask & (rook_lines | bishop_lines)) != 0) {
            // Moving a piece that sits on a king ray can expose check.
            // Fall back to full legality check.
            goto full_legality_check;
          }
        }
        const std::size_t from = static_cast<std::size_t>(m.from);
        bool pinned = false;
        Bitboard pin_mask = 0;

        switch (m.moving_pc) {
        case OccupancyType::wB:
        case OccupancyType::bB: {
          const auto& entry = pinned_map.bishop_pins[from];
          pinned = entry.mask != 0;
          pin_mask = entry.mask;
          break;
        }
        case OccupancyType::wR:
        case OccupancyType::bR: {
          const auto& entry = pinned_map.rook_pins[from];
          pinned = entry.mask != 0;
          pin_mask = entry.mask;
          break;
        }
        case OccupancyType::wQ:
        case OccupancyType::bQ: {
          const auto& entry = pinned_map.queen_pins[from];
          pinned = entry.mask != 0;
          pin_mask = entry.mask;
          break;
        }
        case OccupancyType::wP:
        case OccupancyType::bP: {
          const auto& entry = pinned_map.pawn_pins[from];
          pinned = entry.mask != 0;
          pin_mask = entry.mask;
          break;
        }
        case OccupancyType::wN:
        case OccupancyType::bN: {
          const auto& entry = pinned_map.knight_pins[from];
          pinned = (entry.mask == 0);
          pin_mask = entry.mask;
          break;
        }
        default:
          break;
        }

        if (pinned) {
          const Bitboard to_mask = Bitboard(1) << m.to;
          if ((pin_mask & to_mask) == 0) {
            continue;
          }
        }

        legal_moves[legal_move_count++] = pseudo_moves[i];
        continue;
      }
    }

  full_legality_check:
    Undo u = make_move(board, m);
    if (!is_check(board, stm)) {
      legal_moves[legal_move_count++] = pseudo_moves[i];
    }
    undo_move(board, u);
  }
  move_count = legal_move_count;
#ifndef NDEBUG
  {
    std::string reason;
    if (!validate_board(board, &reason)) {
      std::cerr << "validate_board failed after generate_legal_moves: " << reason
                << " fen=" << board_to_fen(board) << "\n";
      std::abort();
    }
  }
#endif
  return legal_moves;
}

bool is_quiet_position(Board& board, SideToMove stm) {
  if (is_check(board, stm)) {
    return false; // in check is never quiet
  }

  uint16_t move_count = 0;
  auto moves = generate_legal_moves(board, stm, move_count);
  for (uint16_t i = 0; i < move_count; ++i) {
    const Move m = decode_move(moves[i]);
    const bool tactical = (m.captured_pc != OccupancyType::empty) ||
                          (m.promo_pc != OccupancyType::empty) ||
                          flag_is_ep(m.flags) || flag_is_castle(m.flags) ||
                          flag_is_long_castle(m.flags);
    if (tactical) {
      return false;
    }

    // Check if the move gives check; if so, position has tactical tension.
    const Undo u = make_move(board, m);
    const bool gives_check = is_check(board, board.side_to_move);
    undo_move(board, u);
    if (gives_check) {
      return false;
    }
  }

  return true;
}

Undo make_move(Board& b, const Move& m) {
  const bool white = (b.side_to_move == SideToMove::White);
  const Square king_rook =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_kingside_start;
  const Square queen_rook =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_queenside_start;
  const Square king_target = kCastlingSideConfigs[to_index(b.side_to_move)]
                                 .king_kingside_target; // used for short castle
  const Square queen_target = kCastlingSideConfigs[to_index(b.side_to_move)]
                                  .king_queenside_target; // used for long castle
  const Square rook_kingside_target =
      kCastlingSideConfigs[to_index(b.side_to_move)]
          .rook_kingside_target; // used for short castle
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
  undo.captured_sq =
      undo.was_en_passant ? (white ? (m.to - 8) : (m.to + 8)) : m.to;
  if (undo.was_en_passant) {
    undo.captured_pc = white ? OccupancyType::bP : OccupancyType::wP;
  }
  std::copy(std::begin(b.occupancy), std::end(b.occupancy),
            std::begin(undo.occupancy));
  undo.rook_list_before = {b.rook_list[0], b.rook_list[1]};
  undo.king_list_before = {b.king_list[0], b.king_list[1]};
  undo.pawn_list_before = {b.pawn_list[0], b.pawn_list[1]};
  undo.king_positions_before = b.king_positions;
  undo.castled_before = b.has_castled;
  undo.king_captured_before = b.king_captured;

  undo.ep_hash_before = (b.en_passant >= 0 && ep_capture_available(b));

#if SKAKS_ENABLE_HCE
  // Incremental score updates
  undo.material_score_before = b.material_score;
  undo.pst_midgame_score_before = b.pst_midgame_score;
  undo.pst_endgame_score_before = b.pst_endgame_score;
  undo.phase_before = b.phase;

  const auto mover_index = to_index(b.side_to_move);
  const auto enemy_index = to_index(flip_side(b.side_to_move));
  const Square from_sq = static_cast<Square>(m.from);
  const Square to_sq = static_cast<Square>(m.to);
  const auto from_idx = to_index(m.from);
  const auto to_idx = to_index(m.to);
  const auto captured_idx = to_index(undo.captured_sq);
  const auto& mg_tables = midgame_pst();
  const auto& eg_tables = endgame_pst();

  auto update_score = [&](OccupancyType pc, int sq, bool add) {
    int sign = add ? 1 : -1;
    b.material_score += sign * piece_material_value(pc);

    const bool white_piece = is_white(pc);
    const int type_index = (static_cast<int>(pc) - 1) % 6;
    const int oriented_sq = white_piece ? sq : mirror_rank(sq);

    const int mg = mg_tables[static_cast<std::size_t>(type_index)]
                            [static_cast<std::size_t>(oriented_sq)];
    const int eg = eg_tables[static_cast<std::size_t>(type_index)]
                            [static_cast<std::size_t>(oriented_sq)];

    b.pst_midgame_score += sign * (white_piece ? mg : -mg);
    b.pst_endgame_score += sign * (white_piece ? eg : -eg);

    b.phase += sign * kPstPhaseWeights[static_cast<std::size_t>(type_index)];
  };

  // 1. Remove moving piece from 'from'
  update_score(m.moving_pc, m.from, false);

  // 2. Add moving piece to 'to' (or promoted piece)
  OccupancyType placed_pc =
      (m.promo_pc != OccupancyType::empty) ? m.promo_pc : m.moving_pc;
  update_score(placed_pc, m.to, true);

  // 3. Handle Capture
  if (undo.captured_pc != OccupancyType::empty) {
    update_score(undo.captured_pc, undo.captured_sq, false);
  }

  // 4. Handle Castling (Rook move)
  if (undo.was_castling) {
    Square rook_from = flag_is_castle(m.flags) ? king_rook : queen_rook;
    Square rook_to =
        flag_is_castle(m.flags) ? rook_kingside_target : rook_queenside_target;
    OccupancyType rook = white ? OccupancyType::wR : OccupancyType::bR;

    update_score(rook, static_cast<int>(to_index(rook_from)), false);
    update_score(rook, static_cast<int>(to_index(rook_to)), true);
  }
#else
  const auto mover_index = to_index(b.side_to_move);
  const auto enemy_index = to_index(flip_side(b.side_to_move));
  const Square from_sq = static_cast<Square>(m.from);
  const Square to_sq = static_cast<Square>(m.to);
  const auto from_idx = to_index(m.from);
  const auto to_idx = to_index(m.to);
  const auto captured_idx = to_index(undo.captured_sq);
#endif

  b.pieces[from_idx] = OccupancyType::empty;
  if (undo.was_en_passant) {
    b.pieces[captured_idx] = OccupancyType::empty;
  }
  b.pieces[to_idx] =
      (m.promo_pc != OccupancyType::empty) &&
              (m.moving_pc == (white ? OccupancyType::wP : OccupancyType::bP))
          ? m.promo_pc
          : m.moving_pc;

  // TODO: move bitboards updates to functions
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
  if ((m.promo_pc == OccupancyType::empty) &&
      (undo.captured_pc != OccupancyType::empty)) {
    // 0 is the moving piece, 1 is the captured piece or the rook in castling
    undo.pieces_bb[0] =
        b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];

    // bitboard updates, including occupancy updates
    // remove bit from the captured piece, remove from the "from" square, add to
    // the "to" square
    clear_bit(b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1],
              undo.captured_sq);
    clear_bit(b.occupancy[enemy_index], undo.captured_sq);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    clear_bit(b.occupancy[mover_index], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.to);
    set_bit(b.occupancy[mover_index], m.to);

    remove_piece_square(b.rook_list[enemy_index],
                        static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.pawn_list[enemy_index],
                        static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.king_list[enemy_index],
                        static_cast<Square>(undo.captured_sq));

    if (undo.captured_pc == OccupancyType::wK ||
        undo.captured_pc == OccupancyType::bK) {
      b.king_positions[enemy_index] = -1;
      b.king_captured = (undo.captured_pc == OccupancyType::wK)
                            ? PieceColor::White
                            : PieceColor::Black;
    }

  }
  // castling 0 is king, 1 is the rook
  else if ((flag_is_castle(m.flags) || flag_is_long_castle(m.flags)) &&
           (m.captured_pc == OccupancyType::empty)) // redundant check but ok...
  {
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[1] =
        b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR
                                                   : OccupancyType::bR) -
                    1]; // Assuming rook is involved

    // bitboard updates, including occupancy updates
    // remove king and rook from their original squares, add to their new squares
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    clear_bit(b.occupancy[mover_index], m.from);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR
                                                         : OccupancyType::bR) -
                          1],
              flag_is_castle(m.flags) ? king_rook : queen_rook);
    clear_bit(b.occupancy[mover_index],
              flag_is_castle(m.flags) ? king_rook : queen_rook);

    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1],
            flag_is_castle(m.flags) ? king_target : queen_target);
    set_bit(b.occupancy[mover_index],
            flag_is_castle(m.flags) ? king_target : queen_target);
    set_bit(b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR
                                                       : OccupancyType::bR) -
                        1],
            flag_is_castle(m.flags) ? (rook_kingside_target)
                                    : (rook_queenside_target));
    set_bit(b.occupancy[mover_index], flag_is_castle(m.flags)
                                          ? (rook_kingside_target)
                                          : (rook_queenside_target));

    const Square rook_from = flag_is_castle(m.flags) ? king_rook : queen_rook;
    const Square rook_to =
        flag_is_castle(m.flags) ? rook_kingside_target : rook_queenside_target;
    b.pieces[to_index(rook_from)] = OccupancyType::empty;
    b.pieces[to_index(rook_to)] = white ? OccupancyType::wR : OccupancyType::bR;

    // update rook list to reflect the single rook that moved
    update_piece_square(b.rook_list[mover_index], rook_from, rook_to);

    b.has_castled[mover_index] = true;

  }
  // promotion with no capture, 0 is the pawn, 1 is the promoted piece
  else if ((m.promo_pc != OccupancyType::empty) &&
           (m.captured_pc == OccupancyType::empty)) {
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    clear_bit(b.occupancy[mover_index], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);
    set_bit(b.occupancy[mover_index], m.to);
    remove_piece_square(b.pawn_list[mover_index], from_sq);
    if (m.promo_pc == OccupancyType::wR || m.promo_pc == OccupancyType::bR) {
      add_piece_square(b.rook_list[mover_index], to_sq);
    }

  }
  // promotion with capture
  else if (m.promo_pc != OccupancyType::empty) {
    // 0 is the pawn being promoted, 1 capture piece, 2 is the promoted piece
    undo.pieces_bb[0] =
        b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1];
    undo.pieces_bb[1] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    undo.pieces_bb[2] = b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1];

    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(undo.captured_pc) - 1],
              undo.captured_sq);
    clear_bit(b.occupancy[enemy_index], undo.captured_sq);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    clear_bit(b.occupancy[mover_index], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.promo_pc) - 1], m.to);
    set_bit(b.occupancy[mover_index], m.to);
    remove_piece_square(b.pawn_list[mover_index], from_sq);
    remove_piece_square(b.rook_list[enemy_index],
                        static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.pawn_list[enemy_index],
                        static_cast<Square>(undo.captured_sq));
    remove_piece_square(b.king_list[enemy_index],
                        static_cast<Square>(undo.captured_sq));
    if (undo.captured_pc == OccupancyType::wK ||
        undo.captured_pc == OccupancyType::bK) {
      b.king_positions[enemy_index] = -1;
      b.king_captured = (undo.captured_pc == OccupancyType::wK)
                            ? PieceColor::White
                            : PieceColor::Black;
    }
    if (m.promo_pc == OccupancyType::wR || m.promo_pc == OccupancyType::bR) {
      add_piece_square(b.rook_list[mover_index], to_sq);
    }
  }
  // quiet move
  else {
    undo.pieces_bb[0] = b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1];
    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.from);
    clear_bit(b.occupancy[mover_index], m.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(m.moving_pc) - 1], m.to);
    set_bit(b.occupancy[mover_index], m.to);
  }

  b.occupancy[to_index(PieceColor::Both)] =
      b.occupancy[to_index(PieceColor::White)] |
      b.occupancy[to_index(PieceColor::Black)];

  b.fifty_move_counter =
      (m.moving_pc == OccupancyType::wP || m.moving_pc == OccupancyType::bP ||
       m.captured_pc != OccupancyType::empty)
          ? 0
          : b.fifty_move_counter + 1;
  b.en_passant = -1;
  b.ep_square = 0;
  if (flag_is_double_push(m.flags)) {
    const int mid = (m.from + m.to) / 2;
    b.en_passant = mid;
    b.ep_square = bb_of(mid);
  }
  const int castle_mask_after = update_castling_rights(b, m);

  // Switch side to move
  b.side_to_move = flip_side(b.side_to_move);
  const bool ep_hash_after = (b.en_passant >= 0 && ep_capture_available(b));
  update_key_for_move(b, undo, castle_mask_after, undo.ep_hash_before,
                      ep_hash_after);

#ifndef NDEBUG
  {
    std::string reason;
    if (!validate_board(b, &reason)) {
      std::cerr << "validate_board failed after make_move: " << reason
                << " fen=" << board_to_fen(b) << "\n";
      std::abort();
    }
  }
#endif

  return undo;
}

bool allow_null_move(Board& b, int depth, int min_depth) {
  if (depth < min_depth) {
    return false;
  }

  if (b.king_captured != PieceColor::None) {
    return false;
  }

  if (is_check(b, b.side_to_move)) {
    return false;
  }

  if (b.fifty_move_counter < 2) {
    return false;
  }

  const std::size_t pawn_index =
      (b.side_to_move == SideToMove::White)
          ? static_cast<std::size_t>(OccupancyType::wP) - 1
          : static_cast<std::size_t>(OccupancyType::bP) - 1;
  const bool has_pawn = b.pieces_bb[pawn_index] != 0ULL;
  if (!has_pawn) {
    return false;
  }

  Bitboard non_pawn_material = 0ULL;
  if (b.side_to_move == SideToMove::White) {
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::wN) - 1];
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::wB) - 1];
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::wR) - 1];
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::wQ) - 1];
  } else {
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::bN) - 1];
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::bB) - 1];
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::bR) - 1];
    non_pawn_material |=
        b.pieces_bb[static_cast<std::size_t>(OccupancyType::bQ) - 1];
  }

  if (non_pawn_material == 0ULL) {
    return false;
  }

  return true;
}

UndoNull make_null_move(Board& b) {
  UndoNull undo;
  undo.position_key_before = b.position_key;
  undo.fifty_move_counter_before = b.fifty_move_counter;
  undo.en_passant_before = b.en_passant;

  if (b.side_to_move == SideToMove::Black) {
    ++b.ply_count;
  }

  const bool ep_hash_before = (b.en_passant >= 0 && ep_capture_available(b));
  const int ep_file_before = b.en_passant;
  const Zobrist& zobrist = zobrist_table();
  std::uint64_t key = undo.position_key_before;
  if (ep_hash_before && ep_file_before >= 0) {
    key ^= zobrist.enPassantFile[file_of(ep_file_before)];
  }

  b.en_passant = -1;
  b.ep_square = 0;
  b.fifty_move_counter += 1;

  // Switch side to move
  b.side_to_move = flip_side(b.side_to_move);
  key ^= zobrist.sideToMove;

  b.position_key = key;

  return undo;
}

void undo_null_move(Board& b, const UndoNull& u) {
  // Restore position key and board state
  b.position_key = u.position_key_before;
  b.fifty_move_counter = u.fifty_move_counter_before;
  b.en_passant = u.en_passant_before;

  // Switch side to move back
  b.side_to_move = flip_side(b.side_to_move);
  if (b.side_to_move == SideToMove::Black) {
    --b.ply_count;
  }
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

void undo_move(Board& b, const Undo& u) {
  // Switch side to move back
  b.side_to_move = flip_side(b.side_to_move);
  if (b.side_to_move == SideToMove::Black) {
    --b.ply_count;
  }
  const bool white = (b.side_to_move == SideToMove::White);
  const Square king_rook =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_kingside_start;
  const Square queen_rook =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_queenside_start;
  const Square king_target = kCastlingSideConfigs[to_index(b.side_to_move)]
                                 .king_kingside_target; // used for short castle
  const Square queen_target = kCastlingSideConfigs[to_index(b.side_to_move)]
                                  .king_queenside_target; // used for long castle
  const Square rook_kingside_target =
      kCastlingSideConfigs[to_index(b.side_to_move)]
          .rook_kingside_target; // used for short castle
  const Square rook_queenside_target =
      kCastlingSideConfigs[to_index(b.side_to_move)].rook_queenside_target;

  // Restore position key and board state
  b.position_key = u.position_key_before;
  b.fifty_move_counter = u.fifty_move_counter_before;
  b.en_passant = u.en_passant_before;
  b.ep_square = u.ep_square_before;
  b.castling_rights = u.castling_rights_before;
  b.has_castled = u.castled_before;

#if SKAKS_ENABLE_HCE
  // Restore incremental scores
  b.material_score = u.material_score_before;
  b.pst_midgame_score = u.pst_midgame_score_before;
  b.pst_endgame_score = u.pst_endgame_score_before;
  b.phase = u.phase_before;
#endif

  std::copy(std::begin(u.occupancy), std::end(u.occupancy),
            std::begin(b.occupancy));
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
  if ((u.promo_pc == OccupancyType::empty) &&
      (u.captured_pc != OccupancyType::empty)) {
    // 0 is the moving piece, 1 is the captured piece or the rook in castling

    b.pieces_bb[static_cast<std::size_t>(u.captured_pc) - 1] = u.pieces_bb[0];
    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[1];

    // bitboard updates
    // set bit to the from, remove from the "to" square, add back to the captured
    // square
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.to);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.captured_pc) - 1],
            u.captured_sq);

  }
  // castling 0 is king, 1 is the rook
  else if ((flag_is_castle(u.flags) || flag_is_long_castle(u.flags)) &&
           (u.captured_pc == OccupancyType::empty)) // redundant check but ok...
  {

    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[0];
    b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR
                                               : OccupancyType::bR) -
                1] = u.pieces_bb[1];

    // bitboard updates
    // set king and rook back to their original squares, remove from their new
    // squares
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1],
              flag_is_castle(u.flags) ? king_target : queen_target);
    clear_bit(b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR
                                                         : OccupancyType::bR) -
                          1],
              flag_is_castle(u.flags) ? rook_kingside_target
                                      : rook_queenside_target);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(white ? OccupancyType::wR
                                                       : OccupancyType::bR) -
                        1],
            flag_is_castle(u.flags) ? king_rook : queen_rook);

    const Square rook_from = flag_is_castle(u.flags) ? king_rook : queen_rook;
    const Square rook_to =
        flag_is_castle(u.flags) ? rook_kingside_target : rook_queenside_target;
    b.pieces[to_index(rook_to)] = OccupancyType::empty;
    b.pieces[to_index(rook_from)] =
        white ? OccupancyType::wR : OccupancyType::bR;

  }
  // promotion with no capture, 0 is the pawn, 1 is the promoted piece
  else if ((u.promo_pc != OccupancyType::empty) &&
           (u.captured_pc == OccupancyType::empty)) {
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
    // set bit to the from pawn, set bit to the captured piece, remove promoted
    // piece from to square
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.promo_pc) - 1], u.to);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.captured_pc) - 1],
            u.captured_sq);
  }
  //
  else {
    // quite move
    b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1] = u.pieces_bb[0];
    // bitboard updates
    clear_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.to);
    set_bit(b.pieces_bb[static_cast<std::size_t>(u.moving_pc) - 1], u.from);
  }

#ifndef NDEBUG
  {
    std::string reason;
    if (!validate_board(b, &reason)) {
      std::cerr << "validate_board failed after undo_move: " << reason
                << " fen=" << board_to_fen(b) << "\n";
      std::abort();
    }
  }
#endif
}
} // namespace chess