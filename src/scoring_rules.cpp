#include "chess/scoring_rules.hpp"

#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/board_arithmetic.hpp"
#include "chess/evaluation_params.hpp"
#include "chess/piece_values.hpp"
#include "chess/pins.hpp"
#include "chess/pst_tables.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

// AVX is opt-in so generic builds remain portable. Define SKAKS_FORCE_ENABLE_AVX
// when compiling with -mavx (or higher) to restore the vector path.
#if defined(SKAKS_FORCE_DISABLE_AVX)
#define SKAKS_HAVE_AVX_INTRINSICS 0
#elif defined(SKAKS_FORCE_ENABLE_AVX)
#define SKAKS_HAVE_AVX_INTRINSICS 1
#elif defined(__AVX__)
#define SKAKS_HAVE_AVX_INTRINSICS 1
#else
#define SKAKS_HAVE_AVX_INTRINSICS 0
#endif

#if SKAKS_HAVE_AVX_INTRINSICS
#include <immintrin.h>
#endif
// Conditionally include ARM NEON only when available and safe to include.
#if defined(__aarch64__)
#if defined(__has_include)
#if __has_include(<arm_neon.h>)
#include <arm_neon.h>
#define SKAKS_HAVE_NEON 1
#else
#define SKAKS_HAVE_NEON 0
#endif
#else
#define SKAKS_HAVE_NEON 0
#endif
#else
#define SKAKS_HAVE_NEON 0
#endif
#include <cmath>

namespace {
constexpr std::size_t kTermCount =
    static_cast<std::size_t>(chess::TermId::Count);

chess::PhaseWeights& phase_weight_store() {
  static chess::PhaseWeights w = [] {
    chess::PhaseWeights init{};
    for (std::size_t i = 0; i < kTermCount; ++i) {
      init.mg[i] = 1.0f;
      init.eg[i] = 1.0f;
    }
    return init;
  }();
  return w;
}
} // namespace

namespace chess {

PhaseWeights& mutable_phase_weights() {
  return phase_weight_store();
}

const PhaseWeights& phase_weights() {
  return phase_weight_store();
}

void set_phase_weights(const PhaseWeights& w) {
  phase_weight_store() = w;
}

void reset_phase_weights() {
  PhaseWeights w{};
  for (std::size_t i = 0; i < kTermCount; ++i) {
    w.mg[i] = 1.0;
    w.eg[i] = 1.0;
  }
  phase_weight_store() = w;
}

// Helper: White-centric check queries
inline bool white_in_check(const Board& board) {
  return is_check(board, SideToMove::White);
}
inline bool black_in_check(const Board& board) {
  return is_check(board, SideToMove::Black);
}

// Material: White-centric (positive is good for White)
int evaluate_material(const Board& board) {
  return board.material_score;
}

int evaluate_pawn_center_control(const Board& board) {
  int score = 0;
  for (auto sq : {to_index(Square::D4), to_index(Square::D5),
                  to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::wP)
      score += 10;
    else if (piece == OccupancyType::bP)
      score -= 10;
  }
  return score;
}

int evaluate_center_control(const Board& board) {
  int score = 0;
  for (auto sq : {to_index(Square::D4), to_index(Square::D5),
                  to_index(Square::E4), to_index(Square::E5)}) {
    OccupancyType piece = board.pieces[sq];
    if (piece != OccupancyType::empty) {
      const bool is_white_piece = static_cast<std::size_t>(piece) <
                                  static_cast<std::size_t>(OccupancyType::bP);
      score += is_white_piece ? 5 : -5;
    }
  }
  return score;
}

// King safety: White-centric, independent of side_to_move
int evaluate_king_safety(const Board& board) {
  const auto& params = evaluation_params();
  int score = 0;

  // White pawn shield near White king
  {
    const int wking_sq = board.king_positions[to_index(PieceColor::White)];
    if (wking_sq != -1) {
      for (std::uint8_t i = 0;
           i < board.pawn_list[to_index(PieceColor::White)].count; ++i) {
        const auto pawn_sq =
            board.pawn_list[to_index(PieceColor::White)].squares[i];
        if (chebyshev_dist(to_index(pawn_sq),
                           static_cast<std::size_t>(wking_sq)) <= 2) {
          score += params.pawn_shield_bonus;
        }
      }
    }
  }

  // Black pawn shield near Black king (subtract because it's good for Black)
  {
    const int bking_sq = board.king_positions[to_index(PieceColor::Black)];
    if (bking_sq != -1) {
      for (std::uint8_t i = 0;
           i < board.pawn_list[to_index(PieceColor::Black)].count; ++i) {
        const auto pawn_sq =
            board.pawn_list[to_index(PieceColor::Black)].squares[i];
        if (chebyshev_dist(to_index(pawn_sq),
                           static_cast<std::size_t>(bking_sq)) <= 2) {
          score -= params.pawn_shield_bonus;
        }
      }
    }
  }

  // Castling status
  if (board.has_castled[to_index(PieceColor::White)])
    score += params.castling_bonus;
  if (board.has_castled[to_index(PieceColor::Black)])
    score -= params.castling_bonus;

  auto early_king_walk_penalty = [&](PieceColor color) -> int {
    if (params.king_walk_penalty <= 0) {
      return 0;
    }
    const std::size_t idx = to_index(color);
    if (board.has_castled[idx]) {
      return 0;
    }
    const int king_sq = board.king_positions[idx];
    if (king_sq < 0) {
      return 0;
    }
    const int home_sq = (color == PieceColor::White)
                            ? static_cast<int>(to_index(Square::E1))
                            : static_cast<int>(to_index(Square::E8));
    if (king_sq == home_sq) {
      return 0;
    }

    const int rights_mask = to_mask(board.castling_rights);
    const int color_mask = (color == PieceColor::White)
                               ? (to_mask(CastlingRights::WhiteKingside) |
                                  to_mask(CastlingRights::WhiteQueenside))
                               : (to_mask(CastlingRights::BlackKingside) |
                                  to_mask(CastlingRights::BlackQueenside));
    if ((rights_mask & color_mask) != 0) {
      return 0; // still technically castle-able; let opening heuristics handle
                // it
    }

    const int mg_phase = std::min(board.phase, kPstPhaseMax);
    if (params.king_walk_phase_limit > 0 &&
        mg_phase <= params.king_walk_phase_limit) {
      return 0; // already deep enough into the game
    }

    const int home_rank = (color == PieceColor::White) ? 0 : 7;
    const int rank_drift = std::abs(rank_of(king_sq) - home_rank);
    const int file_drift = std::abs(file_of(king_sq) - 4); // e-file origin
    const int distance = std::max(1, rank_drift + file_drift);
    return params.king_walk_penalty * distance;
  };

  score -= early_king_walk_penalty(PieceColor::White);
  score += early_king_walk_penalty(PieceColor::Black);

  // In-check penalties/bonuses (moderate; mate handled in search)
  if (white_in_check(board))
    score -= params.check_penalty;
  if (black_in_check(board))
    score += params.check_penalty;

  return score;
}

// King mobility: evaluate both kings, White-centric, independent of side_to_move
int evaluate_king_mobility(const Board& board) {
  const auto& params = evaluation_params();
  int score = 0;

  const int wking_sq = board.king_positions[to_index(PieceColor::White)];
  if (wking_sq != -1) {
    const int moves = popcount_bitboard(king_attack_bm(
        board, static_cast<u_int8_t>(wking_sq), SideToMove::White));
    score += params.mobility_scaling * moves;
  }

  const int bking_sq = board.king_positions[to_index(PieceColor::Black)];
  if (bking_sq != -1) {
    const int moves = popcount_bitboard(king_attack_bm(
        board, static_cast<u_int8_t>(bking_sq), SideToMove::Black));
    score -= params.mobility_scaling * moves;
  }

  // Fade in king mobility as the game simplifies so we do not reward early
  // king walks. Use the same coarse phase metric as other tapered terms.
  const int mg_phase = std::min(board.phase, kPstPhaseMax);
  const int eg_phase = kPstPhaseMax - mg_phase;
  if (eg_phase == 0 || score == 0) {
    return 0;
  }

  // Integer interpolation with rounding towards nearest.
  const int tapered =
      (score * eg_phase + (eg_phase > 0 ? eg_phase / 2 : 0)) / kPstPhaseMax;
  return tapered;
}

// Attacking pieces: if a piece is attacked by the opponent, penalize for White
// pieces, reward for Black
int evaluate_attacking_pieces(const Board& board) {
  const auto& params = evaluation_params();
  int attack_score = 0;

  for (std::size_t sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::empty)
      continue;

    const bool white_piece = static_cast<std::size_t>(piece) <
                             static_cast<std::size_t>(OccupancyType::bP);

    const int base_weight = params.threat_base[static_cast<std::size_t>(piece)];
    if (base_weight == 0)
      continue;

    // If this square is attacked by the opponent
    const SideToMove attacker =
        white_piece ? SideToMove::Black : SideToMove::White;
    if (!is_square_attacked(board, static_cast<u_int8_t>(sq), attacker))
      continue;

    // Scale by advancement (optional heuristic)
    const int rank = rank_of(static_cast<int>(sq));
    const int distance_from_home = white_piece ? rank : (7 - rank);
    const int positional_scale = 1 + distance_from_home / 2;
    const int threat_value = base_weight * positional_scale;

    attack_score += white_piece ? -threat_value : +threat_value;
  }

  return attack_score;
}

int evaluate_hanging_pieces(const Board& board) {
  const auto& params = evaluation_params();
  int score = 0;

  for (std::size_t sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[sq];
    if (piece == OccupancyType::empty)
      continue;
    if (piece == OccupancyType::wK || piece == OccupancyType::bK)
      continue;

    const bool white_piece = is_white(piece);
    const SideToMove enemy = white_piece ? SideToMove::Black : SideToMove::White;
    const SideToMove friendly =
        white_piece ? SideToMove::White : SideToMove::Black;

    if (!is_square_attacked(board, static_cast<u_int8_t>(sq), enemy))
      continue;

    if (is_square_attacked(board, static_cast<u_int8_t>(sq), friendly))
      continue;

    int penalty = piece_material_magnitude(piece) / params.hanging_divisor;
    penalty = std::max(params.hanging_min_penalty, penalty);
    score += white_piece ? -penalty : +penalty;
  }

  return score;
}

int evaluate_king_ring_pressure(const Board& board) {
  const auto& params = evaluation_params();
  auto pressure_for = [&](PieceColor color) {
    const int king_sq = board.king_positions[to_index(color)];
    if (king_sq < 0)
      return 0;

    const Bitboard ring = KING_ATTACKS[static_cast<std::size_t>(king_sq)];
    const SideToMove enemy =
        (color == PieceColor::White) ? SideToMove::Black : SideToMove::White;
    const SideToMove friendly = flip_side(enemy);

    int danger = 0;
    Bitboard mask = ring;
    while (mask) {
      const int sq = lsb_index(mask);
      mask &= mask - 1;

      if (!is_square_attacked(board, static_cast<u_int8_t>(sq), enemy))
        continue;

      int weight = params.king_ring_base;
      if (const auto attacker =
              find_smallest_attacker(board, static_cast<u_int8_t>(sq), enemy)) {
        weight +=
            params
                .king_attack_weights[static_cast<std::size_t>(attacker->piece)];
      }

      if (is_square_attacked(board, static_cast<u_int8_t>(sq), friendly)) {
        weight /= params.king_ring_defended_scale;
        weight = std::max(weight, 0);
      }

      const OccupancyType occupant = board.pieces[static_cast<std::size_t>(sq)];
      if (occupant != OccupancyType::empty) {
        const bool enemy_piece =
            is_white(occupant) != (color == PieceColor::White);
        if (enemy_piece) {
          weight += params.king_ring_enemy_occupier;
          weight += piece_material_magnitude(occupant) /
                    params.king_ring_enemy_piece_material_scale;
        }
      }

      danger += weight;
    }
    return danger;
  };

  const int white_pressure = pressure_for(PieceColor::White);
  const int black_pressure = pressure_for(PieceColor::Black);

  return black_pressure - white_pressure;
}

int compute_threat_pressure(const Board& board, SideToMove attacker) {
  const bool attacker_is_white = attacker == SideToMove::White;
  int pressure = 0;

  for (int sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece == OccupancyType::empty)
      continue;

    const bool piece_is_white = is_white(piece);
    if (piece_is_white == attacker_is_white)
      continue;

    if (!is_square_attacked(board, static_cast<u_int8_t>(sq), attacker))
      continue;

    const int magnitude = piece_material_magnitude(piece);
    pressure += std::max(1, magnitude / 100);
  }

  return pressure;
}

// let's git it some kick
int evaluate_initiative(const Board& board) {
  const auto& params = evaluation_params();
  const SideToMove us = board.side_to_move;
  const SideToMove them = flip_side(us);

  const int our_pressure = compute_threat_pressure(board, us);
  const int their_pressure = compute_threat_pressure(board, them);

  const int net_pressure = our_pressure - their_pressure;
  return params.tempo_bonus + params.threat_weight * net_pressure;
}

int evaluate_pst(const Board& board) {
  const int mg_phase = std::min(board.phase, kPstPhaseMax);
  const int eg_phase = kPstPhaseMax - mg_phase;
  const int tapered =
      (board.pst_midgame_score * mg_phase + board.pst_endgame_score * eg_phase) /
      kPstPhaseMax;
  return tapered;
}

bool is_passed_pawn(const Board& board, int sq, bool white) {
  const int file = file_of(sq);
  const int rank = rank_of(sq);
  const OccupancyType enemy_pawn = white ? OccupancyType::bP : OccupancyType::wP;
  const int dir = white ? 1 : -1;

  for (int r = rank + dir; white ? (r <= 7) : (r >= 0); r += dir) {
    for (int df = -1; df <= 1; ++df) {
      const int f = file + df;
      if (f < 0 || f > 7) {
        continue;
      }
      const int idx = r * 8 + f;
      if (board.pieces[static_cast<std::size_t>(idx)] == enemy_pawn) {
        return false;
      }
    }
  }
  return true;
}

int evaluate_passed_pawns(const Board& board) {
  const auto& params = evaluation_params();
  int score = 0;
  for (int sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece != OccupancyType::wP && piece != OccupancyType::bP) {
      continue;
    }

    const bool white = (piece == OccupancyType::wP);
    if (!is_passed_pawn(board, sq, white)) {
      continue;
    }

    const int advance = white ? rank_of(sq) : (7 - rank_of(sq));
    const int bonus =
        params.passed_pawn_base + params.passed_pawn_advance * advance;
    score += white ? bonus : -bonus;
  }
  return score;
}

int evaluate_pins(const Board& board) {
  const auto& params = evaluation_params();
  int score = 0;
  static thread_local PinnedMapByPiece pin_white_buffer;
  static thread_local PinnedMapByPiece pin_black_buffer;
  build_pinned_map_into(board, SideToMove::White, pin_white_buffer);
  build_pinned_map_into(board, SideToMove::Black, pin_black_buffer);
  const auto& pin_white = pin_white_buffer;
  const auto& pin_black = pin_black_buffer;
  auto apply_penalty = [&](const PinnedBitBoardDirections& entry,
                           int base_penalty, int mobility_penalty,
                           int side_sign) {
    if (entry.mask == 0 || entry.mask == ~Bitboard{0}) {
      return;
    }
    const int mobility = popcount_bitboard(entry.mask);
    score += side_sign * (base_penalty + mobility_penalty * mobility);
  };
  for (std::size_t sq = 0; sq < 64; ++sq) {
    const auto piece = board.pieces[sq];
    if (piece == OccupancyType::empty) {
      continue;
    }
    const bool is_white_piece = is_white(piece);
    const auto& pin_map = is_white_piece ? pin_white : pin_black;
    const int side_sign = is_white_piece ? -1 : 1;

    switch (piece) {
    case OccupancyType::wB:
    case OccupancyType::bB: {
      const auto entry = pin_map.bishop_pins[sq];
      const auto penalty = params.bishop_pin_penalty;
      apply_penalty(entry, penalty.base, penalty.mobility, side_sign);
      break;
    }
    case OccupancyType::wR:
    case OccupancyType::bR: {
      const auto entry = pin_map.rook_pins[sq];
      const auto penalty = params.rook_pin_penalty;
      apply_penalty(entry, penalty.base, penalty.mobility, side_sign);
      break;
    }
    case OccupancyType::wQ:
    case OccupancyType::bQ:
      break;
    case OccupancyType::wN:
    case OccupancyType::bN: {
      const auto entry = pin_map.knight_pins[sq];
      const auto penalty = params.knight_pin_penalty;
      apply_penalty(entry, penalty.base, penalty.mobility, side_sign);
      break;
    }
    case OccupancyType::wP:
    case OccupancyType::bP: {
      const auto entry = pin_map.pawn_pins[sq];
      const bool is_diagonal_pin = entry.direction == MoveDirection::NE ||
                                   entry.direction == MoveDirection::NW ||
                                   entry.direction == MoveDirection::SE ||
                                   entry.direction == MoveDirection::SW;
      const auto penalty = is_diagonal_pin ? params.pawn_pin_diagonal_penalty
                                           : params.pawn_pin_straight_penalty;
      apply_penalty(entry, penalty.base, penalty.mobility, side_sign);
      break;
    }
    default:
      break;
    }
  }
  return score;
}

// In-lieu of a book move, for now

inline bool is_knight_good_dev(int sq, bool white) {
  if (white) {
    return (sq == to_index(Square::C3) || sq == to_index(Square::F3));
  }
  return (sq == to_index(Square::C6) || sq == to_index(Square::F6));
}

inline bool is_bishop_good_dev(int sq, bool white) {
  if (white) {
    return (sq == to_index(Square::C4) || sq == to_index(Square::F4));
  }
  return (sq == to_index(Square::C5) || sq == to_index(Square::F5));
}

int opening_phase(const Board& board) {
  int phase = 0;
  for (int sq = 0; sq < 64; ++sq) {
    const auto piece = board.pieces[static_cast<std::size_t>(sq)];
    if (piece == OccupancyType::empty) {
      continue;
    }
    const int type_index = (static_cast<int>(piece) - 1) % 6;
    phase += kPstPhaseWeights[static_cast<std::size_t>(type_index)];
  }
  const int mg_phase = std::min(phase, kPstPhaseMax);
  const int opening = std::max(0, 64 - mg_phase);
  return opening; // [0..64], larger at start, shrinks as material/phase declines
}

int evaluate_bishop_pair(const Board& board) {
  const auto& params = evaluation_params();
  int white_bishops = 0;
  int black_bishops = 0;
  for (const auto& piece : board.pieces) {
    if (piece == OccupancyType::wB) {
      ++white_bishops;
    } else if (piece == OccupancyType::bB) {
      ++black_bishops;
    }
  }

  if (white_bishops < 2 && black_bishops < 2) {
    return 0;
  }

  const int phase = opening_phase(board);
  const int endgame_emphasis = (64 - phase) / 4;
  const int bonus = params.bishop_pair_bonus + endgame_emphasis;
  int score = 0;
  if (white_bishops >= 2) {
    score += bonus;
  }
  if (black_bishops >= 2) {
    score -= bonus;
  }
  return score;
}

int evaluate_rook_file_control(const Board& board) {
  const auto& params = evaluation_params();
  std::array<bool, 8> white_pawn_on_file{};
  std::array<bool, 8> black_pawn_on_file{};

  const auto& white_pawns = board.pawn_list[to_index(PieceColor::White)];
  for (std::uint8_t i = 0; i < white_pawns.count; ++i) {
    const int sq = static_cast<int>(white_pawns.squares[i]);
    const std::size_t file_idx = static_cast<std::size_t>(file_of(sq));
    white_pawn_on_file[file_idx] = true;
  }
  const auto& black_pawns = board.pawn_list[to_index(PieceColor::Black)];
  for (std::uint8_t i = 0; i < black_pawns.count; ++i) {
    const int sq = static_cast<int>(black_pawns.squares[i]);
    const std::size_t file_idx = static_cast<std::size_t>(file_of(sq));
    black_pawn_on_file[file_idx] = true;
  }

  const int phase = opening_phase(board);
  const int endgame = 64 - phase;
  const int open_bonus = params.rook_open_file_bonus + endgame / 2;
  const int semi_bonus = params.rook_semi_open_file_bonus + endgame / 4;

  int score = 0;
  auto accumulate = [&](PieceColor color) {
    const auto& rooks = board.rook_list[to_index(color)];
    for (std::uint8_t i = 0; i < rooks.count; ++i) {
      const int sq = static_cast<int>(rooks.squares[i]);
      const std::size_t file = static_cast<std::size_t>(file_of(sq));
      const bool friendly_pawn = (color == PieceColor::White)
                                     ? white_pawn_on_file[file]
                                     : black_pawn_on_file[file];
      if (friendly_pawn) {
        continue;
      }
      const bool enemy_pawn = (color == PieceColor::White)
                                  ? black_pawn_on_file[file]
                                  : white_pawn_on_file[file];
      const int bonus = enemy_pawn ? semi_bonus : open_bonus;
      score += (color == PieceColor::White) ? bonus : -bonus;
    }
  };

  accumulate(PieceColor::White);
  accumulate(PieceColor::Black);

  return score;
}

int evaluate_piece_mobility(const Board& board) {
  const auto& params = evaluation_params();
  if (params.knight_mobility_scale == 0 && params.bishop_mobility_scale == 0 &&
      params.rook_mobility_scale == 0 && params.queen_mobility_scale == 0) {
    return 0;
  }

  int score = 0;
  for (int sq = 0; sq < 64; ++sq) {
    const OccupancyType occ = board.pieces[static_cast<std::size_t>(sq)];
    if (occ == OccupancyType::empty) {
      continue;
    }

    const bool white_piece = is_white(occ);
    const SideToMove side = white_piece ? SideToMove::White : SideToMove::Black;
    int contribution = 0;

    switch (occ) {
    case OccupancyType::wN:
    case OccupancyType::bN: {
      if (params.knight_mobility_scale == 0) {
        break;
      }
      const Bitboard attacks =
          knight_attack_bm(board, static_cast<std::uint8_t>(sq), side);
      const int moves = popcount_bitboard(attacks);
      contribution = moves * params.knight_mobility_scale;
      break;
    }
    case OccupancyType::wB:
    case OccupancyType::bB: {
      if (params.bishop_mobility_scale == 0) {
        break;
      }
      const Bitboard attacks =
          bishop_attack_bm(board, static_cast<std::uint8_t>(sq), side);
      const int moves = popcount_bitboard(attacks);
      contribution = moves * params.bishop_mobility_scale;
      break;
    }
    case OccupancyType::wR:
    case OccupancyType::bR: {
      if (params.rook_mobility_scale == 0) {
        break;
      }
      const Bitboard attacks =
          rook_attack_bm(board, static_cast<std::uint8_t>(sq), side);
      const int moves = popcount_bitboard(attacks);
      contribution = moves * params.rook_mobility_scale;
      break;
    }
    case OccupancyType::wQ:
    case OccupancyType::bQ: {
      if (params.queen_mobility_scale == 0) {
        break;
      }
      const Bitboard attacks =
          queen_attack_bm(board, static_cast<std::uint8_t>(sq), side);
      const int moves = popcount_bitboard(attacks);
      contribution = moves * params.queen_mobility_scale;
      break;
    }
    default:
      break;
    }

    if (contribution == 0) {
      continue;
    }
    score += white_piece ? contribution : -contribution;
  }

  return score;
}

int evaluate_pawn_structure(const Board& board) {
  const auto& params = evaluation_params();
  if (params.doubled_pawn_penalty == 0 && params.isolated_pawn_penalty == 0 &&
      params.backward_pawn_penalty == 0) {
    return 0;
  }

  auto compute_penalty = [&](PieceColor color) -> int {
    const auto& list = board.pawn_list[to_index(color)];
    std::array<int, 8> file_counts{};
    for (std::uint8_t i = 0; i < list.count; ++i) {
      const int sq = static_cast<int>(list.squares[i]);
      const int file = file_of(sq);
      ++file_counts[static_cast<std::size_t>(file)];
    }

    int penalty = 0;
    for (int file = 0; file < 8; ++file) {
      const int count = file_counts[static_cast<std::size_t>(file)];
      if (count == 0) {
        continue;
      }
      if (params.doubled_pawn_penalty > 0 && count > 1) {
        penalty += (count - 1) * params.doubled_pawn_penalty;
      }
      if (params.isolated_pawn_penalty > 0) {
        const bool has_left =
            (file > 0) && (file_counts[static_cast<std::size_t>(file - 1)] > 0);
        const bool has_right =
            (file < 7) && (file_counts[static_cast<std::size_t>(file + 1)] > 0);
        if (!has_left && !has_right) {
          penalty += count * params.isolated_pawn_penalty;
        }
      }
    }

    if (params.backward_pawn_penalty > 0 && list.count > 0) {
      const bool white = (color == PieceColor::White);
      const int dir = white ? 1 : -1;
      for (std::uint8_t i = 0; i < list.count; ++i) {
        const int sq = static_cast<int>(list.squares[i]);
        const int file = file_of(sq);
        const int rank = rank_of(sq);
        const int next_rank = rank + dir;
        if (next_rank < 0 || next_rank > 7) {
          continue;
        }
        const int forward_sq = next_rank * 8 + file;
        const OccupancyType forward_occ =
            board.pieces[static_cast<std::size_t>(forward_sq)];
        const bool blocked_by_enemy = forward_occ != OccupancyType::empty &&
                                      (is_white(forward_occ) != white);
        if (!blocked_by_enemy) {
          continue;
        }

        bool supporting_pawn = false;
        for (int df : {-1, 1}) {
          const int adj_file = file + df;
          if (adj_file < 0 || adj_file > 7) {
            continue;
          }
          for (int step = rank; white ? (step <= 7) : (step >= 0); step += dir) {
            const int probe_sq = step * 8 + adj_file;
            const OccupancyType occ =
                board.pieces[static_cast<std::size_t>(probe_sq)];
            if (occ == (white ? OccupancyType::wP : OccupancyType::bP)) {
              supporting_pawn = true;
              break;
            }
          }
          if (supporting_pawn) {
            break;
          }
        }

        if (!supporting_pawn) {
          penalty += params.backward_pawn_penalty;
        }
      }
    }

    return penalty;
  };

  const int white_penalty = compute_penalty(PieceColor::White);
  const int black_penalty = compute_penalty(PieceColor::Black);
  return black_penalty - white_penalty;
}

#if SKAKS_ENABLE_HCE
EvalVector compute_eval_vector(const Board& b) {
  EvalVector v{};
  v.f[static_cast<int>(TermId::Material)] = evaluate_material(b);
  v.f[static_cast<int>(TermId::PawnCenter)] = evaluate_pawn_center_control(b);
  v.f[static_cast<int>(TermId::CenterControl)] = evaluate_center_control(b);
  v.f[static_cast<int>(TermId::Attacking)] = evaluate_attacking_pieces(b);
  v.f[static_cast<int>(TermId::KingSafety)] = evaluate_king_safety(b);
  v.f[static_cast<int>(TermId::KingMobility)] = evaluate_king_mobility(b);
  v.f[static_cast<int>(TermId::Pins)] = evaluate_pins(b);

  const int mg_phase = std::min(b.phase, kPstPhaseMax);
  const int eg_phase = kPstPhaseMax - mg_phase;
  v.f[static_cast<int>(TermId::PstMg)] = b.pst_midgame_score;
  v.f[static_cast<int>(TermId::PstEg)] = b.pst_endgame_score;

  v.f[static_cast<int>(TermId::PassedPawns)] = evaluate_passed_pawns(b);
  v.f[static_cast<int>(TermId::Initiative)] = evaluate_initiative(b);
  v.f[static_cast<int>(TermId::Hanging)] = evaluate_hanging_pieces(b);
  v.f[static_cast<int>(TermId::KingRing)] = evaluate_king_ring_pressure(b);
  v.f[static_cast<int>(TermId::BishopPair)] = evaluate_bishop_pair(b);
  v.f[static_cast<int>(TermId::RookFiles)] = evaluate_rook_file_control(b);
  v.f[static_cast<int>(TermId::MinorMobility)] = evaluate_piece_mobility(b);
  v.f[static_cast<int>(TermId::PawnStructure)] = evaluate_pawn_structure(b);

  v.mg_phase = mg_phase;
  v.eg_phase = eg_phase;
  return v;
}
#else
EvalVector compute_eval_vector(const Board& /*b*/) {
  return {};
}
#endif

int eval_linear(const EvalVector& v, const PhaseWeights& W) {
  const float mgw =
      static_cast<float>(v.mg_phase) / static_cast<float>(kPstPhaseMax);
  const float egw =
      static_cast<float>(v.eg_phase) / static_cast<float>(kPstPhaseMax);

  // Precompute interpolated weights into a small contiguous float array to
  // allow efficient vectorized dot-product implementations and to make the
  // code easier to port to other SIMD ISAs (NEON).
  alignas(32) float wbuf[kTermCount];
  for (std::size_t i = 0; i < kTermCount; ++i) {
    wbuf[i] = W.mg[i] * mgw + W.eg[i] * egw;
  }

#if SKAKS_HAVE_AVX_INTRINSICS
  // AVX path — only enabled when the compiler targets AVX-capable CPUs
  __m256 sum_vec = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 7 < kTermCount; i += 8) {
    __m256 wi_vec = _mm256_loadu_ps(&wbuf[i]);
    __m256 fi_vec = _mm256_cvtepi32_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v.f[i])));
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(wi_vec, fi_vec));
  }
  float sum = 0.0f;
  float sum_arr[8];
  _mm256_storeu_ps(sum_arr, sum_vec);
  for (int j = 0; j < 8; ++j)
    sum += sum_arr[j];
  for (; i < kTermCount; ++i) {
    sum += wbuf[i] * static_cast<float>(v.f[i]);
  }
#elif SKAKS_HAVE_NEON
  // NEON path for aarch64 — process 4 floats at a time
  float32x4_t sumv = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 3 < kTermCount; i += 4) {
    float32x4_t wv = vld1q_f32(&wbuf[i]);
    // load int32 values and convert to float32
    int32x4_t iv = vld1q_s32(reinterpret_cast<const int32_t*>(&v.f[i]));
    float32x4_t fv = vcvtq_f32_s32(iv);
    sumv = vmlaq_f32(sumv, wv, fv);
  }
  float sum = 0.0f;
  float temp[4];
  vst1q_f32(temp, sumv);
  for (int j = 0; j < 4; ++j)
    sum += temp[j];
  for (; i < kTermCount; ++i) {
    sum += wbuf[i] * static_cast<float>(v.f[i]);
  }
#else
  // Portable scalar fallback
  float sum = 0.0f;
  for (std::size_t i = 0; i < kTermCount; ++i) {
    sum += wbuf[i] * static_cast<float>(v.f[i]);
  }
#endif

  return static_cast<int>(std::lround(sum));
}

// Helper: compute per-term contributions (wi * fi) into out array. Useful
// for incremental updates where only a few terms or the phase changes.
void compute_term_contributions(const EvalVector& v, const PhaseWeights& W,
                                float out[kTermCount]) {
  const float mgw =
      static_cast<float>(v.mg_phase) / static_cast<float>(kPstPhaseMax);
  const float egw =
      static_cast<float>(v.eg_phase) / static_cast<float>(kPstPhaseMax);
  for (std::size_t i = 0; i < kTermCount; ++i) {
    const float wi = W.mg[i] * mgw + W.eg[i] * egw;
    out[i] = wi * static_cast<float>(v.f[i]);
  }
}

int evaluate_opening_principles(const Board& board) {
  const auto& params = evaluation_params();
  const int open_w = opening_phase(board);
  if (open_w == 0) {
    return 0;
  }

  int score = 0;

  bool w_back_blocked = false;
  bool b_back_blocked = false;

  for (int sq = 0; sq < 64; ++sq) {
    const auto pc = board.pieces[static_cast<std::size_t>(sq)];
    if (pc == OccupancyType::empty) {
      continue;
    }

    switch (pc) {
    case OccupancyType::wN:
      if (is_knight_good_dev(sq, true)) {
        score += (params.knight_dev_bonus * open_w) / 64;
      }
      break;
    case OccupancyType::bN:
      if (is_knight_good_dev(sq, false)) {
        score -= (params.knight_dev_bonus * open_w) / 64;
      }
      break;
    case OccupancyType::wB:
      if (is_bishop_good_dev(sq, true)) {
        score += (params.bishop_dev_bonus * open_w) / 64;
      }
      break;
    case OccupancyType::bB:
      if (is_bishop_good_dev(sq, false)) {
        score -= (params.bishop_dev_bonus * open_w) / 64;
      }
      break;
    default:
      break;
    }

    // Back-rank blockers prevent connected rooks bonus
    if (rank_of(sq) == 0) {
      if (pc == OccupancyType::wN || pc == OccupancyType::wB ||
          pc == OccupancyType::wQ) {
        w_back_blocked = true;
      }
    } else if (rank_of(sq) == 7) {
      if (pc == OccupancyType::bN || pc == OccupancyType::bB ||
          pc == OccupancyType::bQ) {
        b_back_blocked = true;
      }
    }
  }

  if (!w_back_blocked) {
    score += (params.connect_rooks_bonus * open_w) / 64;
  }
  if (!b_back_blocked) {
    score -= (params.connect_rooks_bonus * open_w) / 64;
  }

  auto add_central_pawn = [&](Square sq, OccupancyType pc) {
    if (board.pieces[static_cast<std::size_t>(sq)] == pc) {
      score +=
          (is_white(pc) ? +1 : -1) * ((params.central_pawn_bonus * open_w) / 64);
    }
  };
  add_central_pawn(Square::E4, OccupancyType::wP);
  add_central_pawn(Square::D4, OccupancyType::wP);
  add_central_pawn(Square::E5, OccupancyType::bP);
  add_central_pawn(Square::D5, OccupancyType::bP);

  if (board.has_castled[to_index(PieceColor::White)]) {
    score += (params.castle_urgency * open_w) / 64;
  } else if (open_w > 40) {
    score -= ((open_w - 40) * 2);
  }

  if (board.has_castled[to_index(PieceColor::Black)]) {
    score -= (params.castle_urgency * open_w) / 64;
  } else if (open_w > 40) {
    score += ((open_w - 40) * 2);
  }

  auto count_white_minor_developed = [&]() -> int {
    int c = 0;
    for (int sq = 0; sq < 64; ++sq) {
      const auto pc = board.pieces[static_cast<std::size_t>(sq)];
      if (pc == OccupancyType::wN || pc == OccupancyType::wB) {
        if (rank_of(sq) > 0) {
          c += 1;
        }
      }
    }
    return c;
  };
  auto count_black_minor_developed = [&]() -> int {
    int c = 0;
    for (int sq = 0; sq < 64; ++sq) {
      const auto pc = board.pieces[static_cast<std::size_t>(sq)];
      if (pc == OccupancyType::bN || pc == OccupancyType::bB) {
        if (rank_of(sq) < 7) {
          c += 1;
        }
      }
    }
    return c;
  };

  const int w_minors = count_white_minor_developed();
  const int b_minors = count_black_minor_developed();
  const bool wq_moved = board.pieces[to_index(Square::D1)] != OccupancyType::wQ;
  const bool bq_moved = board.pieces[to_index(Square::D8)] != OccupancyType::bQ;

  if (wq_moved && w_minors < 2) {
    score -= (params.early_queen_penalty * open_w) / 64;
  }
  if (bq_moved && b_minors < 2) {
    score += (params.early_queen_penalty * open_w) / 64;
  }

  auto penalize_flank_push = [&](Square start, OccupancyType pc, int dir) {
    const int s = static_cast<int>(to_index(start));
    const int t = s + dir;
    if (t < 0 || t >= 64) {
      return;
    }
    const auto from_piece = board.pieces[static_cast<std::size_t>(s)];
    const auto to_piece = board.pieces[static_cast<std::size_t>(t)];
    if (from_piece == OccupancyType::empty && to_piece == pc && open_w > 40) {
      score +=
          is_white(pc) ? -params.flank_pawn_penalty : +params.flank_pawn_penalty;
    }
  };
  penalize_flank_push(Square::A2, OccupancyType::wP, +8);
  penalize_flank_push(Square::H2, OccupancyType::wP, +8);
  penalize_flank_push(Square::A7, OccupancyType::bP, -8);
  penalize_flank_push(Square::H7, OccupancyType::bP, -8);

  return score;
}

// Final evaluation: strictly White-centric; do not flip by side_to_move
#if SKAKS_ENABLE_HCE
int evaluate_board(const Board& board) {
  if (board.king_captured == PieceColor::White) {
    return -100000; // Black wins (bad for White)
  } else if (board.king_captured == PieceColor::Black) {
    return 100000; // White wins
  }

  const auto v = compute_eval_vector(board);
  const int raw_linear = eval_linear(v, phase_weights());
  const auto& params = evaluation_params();
  const double quiet_cap = std::max(1.0, params.eval_quiet_cap);
  const double adjusted = static_cast<double>(raw_linear);
  const double compressed = quiet_cap * std::tanh(adjusted / quiet_cap);
  return static_cast<int>(std::lround(compressed));
}
#else
int evaluate_board(const Board& /*board*/) {
  return 0;
}
#endif

} // namespace chess