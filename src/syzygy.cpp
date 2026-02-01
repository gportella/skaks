#include "chess/syzygy.hpp"

#include "chess/board_arithmetic.hpp"
#include "chess/defaults.hpp"
#include "chess/types.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <tbprobe.h>

namespace chess::syzygy {

namespace {

bool g_initialized = false;
bool g_available = false;
int g_max_pieces = 0;
std::string g_path;

Bitboard piece_bb(const Board& board, Piece piece) {
  return board.pieces_bb[static_cast<std::size_t>(piece)];
}

int total_pieces(const Board& board) {
  const Bitboard occ =
      board.occupancy[static_cast<std::size_t>(PieceColor::Both)];
  return popcount_bitboard(occ);
}

bool can_probe(const Board& board) {
  if (!g_available) {
    return false;
  }
  if (board.castling_rights != CastlingRights::NoCastling) {
    return false;
  }
  if (board.fifty_move_counter != 0) {
    return false;
  }
  return total_pieces(board) <= g_max_pieces;
}

OccupancyType promo_from_tb(int promo, SideToMove stm) {
  if (promo == TB_PROMOTES_QUEEN) {
    return (stm == SideToMove::White) ? OccupancyType::wQ : OccupancyType::bQ;
  }
  if (promo == TB_PROMOTES_ROOK) {
    return (stm == SideToMove::White) ? OccupancyType::wR : OccupancyType::bR;
  }
  if (promo == TB_PROMOTES_BISHOP) {
    return (stm == SideToMove::White) ? OccupancyType::wB : OccupancyType::bB;
  }
  if (promo == TB_PROMOTES_KNIGHT) {
    return (stm == SideToMove::White) ? OccupancyType::wN : OccupancyType::bN;
  }
  return OccupancyType::empty;
}

int wdl_to_score(unsigned wdl) {
  switch (wdl) {
  case TB_WIN:
    return MATE_SCORE - 1;
  case TB_LOSS:
    return -MATE_SCORE + 1;
  case TB_DRAW:
  case TB_BLESSED_LOSS:
  case TB_CURSED_WIN:
  default:
    return 0;
  }
}

} // namespace

bool init(std::string_view path) {
  g_initialized = true;
  g_available = false;
  g_max_pieces = 0;
  g_path.assign(path.begin(), path.end());

  if (g_path.empty()) {
    return false;
  }

  if (!tb_init(g_path.c_str())) {
    g_available = false;
    g_max_pieces = 0;
    return false;
  }

  g_max_pieces = static_cast<int>(TB_LARGEST);
  g_available = g_max_pieces > 0;
  return g_available;
}

void init_from_env() {
  const char* env = std::getenv("SKAKS_SYZYGY_PATH");
  if (!env || *env == '\0') {
    const char* home = std::getenv("HOME");
    if (!home || *home == '\0') {
      return;
    }
    std::filesystem::path base =
        std::filesystem::path(home) / ".local" / "share" / "skaks";
    std::filesystem::path wdl = base / "Syzygy345WDL";
    std::filesystem::path dtz = base / "Syzygy345DTZ";
    if (std::filesystem::exists(wdl) && std::filesystem::exists(dtz)) {
      const std::string combined = wdl.string() + ":" + dtz.string();
      init(combined);
    }
    return;
  }
  init(env);
}

void free() {
  if (!g_initialized) {
    return;
  }
  tb_free();
  g_initialized = false;
  g_available = false;
  g_max_pieces = 0;
}

bool available() {
  return g_available;
}

int max_pieces() {
  return g_max_pieces;
}

std::string_view path() {
  return g_path;
}

ProbeResult probe_wdl(const Board& board) {
  ProbeResult result{};
  result.available = can_probe(board);
  if (!result.available) {
    return result;
  }

  const Bitboard white =
      board.occupancy[static_cast<std::size_t>(PieceColor::White)];
  const Bitboard black =
      board.occupancy[static_cast<std::size_t>(PieceColor::Black)];
  const Bitboard kings = piece_bb(board, Piece::wK) | piece_bb(board, Piece::bK);
  const Bitboard queens =
      piece_bb(board, Piece::wQ) | piece_bb(board, Piece::bQ);
  const Bitboard rooks = piece_bb(board, Piece::wR) | piece_bb(board, Piece::bR);
  const Bitboard bishops =
      piece_bb(board, Piece::wB) | piece_bb(board, Piece::bB);
  const Bitboard knights =
      piece_bb(board, Piece::wN) | piece_bb(board, Piece::bN);
  const Bitboard pawns = piece_bb(board, Piece::wP) | piece_bb(board, Piece::bP);
  const unsigned ep =
      (board.en_passant >= 0) ? static_cast<unsigned>(board.en_passant) : 0u;
  const bool turn = (board.side_to_move == SideToMove::White);

  const unsigned wdl =
      tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns,
                   static_cast<unsigned>(board.fifty_move_counter), 0, ep, turn);
  if (wdl == TB_RESULT_FAILED) {
    result.available = false;
    return result;
  }

  result.wdl = static_cast<int>(wdl);
  result.score = wdl_to_score(wdl);
  return result;
}

ProbeResult probe_root_wdl(const Board& board) {
  ProbeResult result{};
  result.available = can_probe(board);
  if (!result.available) {
    return result;
  }

  const Bitboard white =
      board.occupancy[static_cast<std::size_t>(PieceColor::White)];
  const Bitboard black =
      board.occupancy[static_cast<std::size_t>(PieceColor::Black)];
  const Bitboard kings = piece_bb(board, Piece::wK) | piece_bb(board, Piece::bK);
  const Bitboard queens =
      piece_bb(board, Piece::wQ) | piece_bb(board, Piece::bQ);
  const Bitboard rooks = piece_bb(board, Piece::wR) | piece_bb(board, Piece::bR);
  const Bitboard bishops =
      piece_bb(board, Piece::wB) | piece_bb(board, Piece::bB);
  const Bitboard knights =
      piece_bb(board, Piece::wN) | piece_bb(board, Piece::bN);
  const Bitboard pawns = piece_bb(board, Piece::wP) | piece_bb(board, Piece::bP);
  const unsigned ep =
      (board.en_passant >= 0) ? static_cast<unsigned>(board.en_passant) : 0u;
  const bool turn = (board.side_to_move == SideToMove::White);

  TbRootMoves moves{};
  const int ok =
      tb_probe_root_wdl(white, black, kings, queens, rooks, bishops, knights,
                        pawns, static_cast<unsigned>(board.fifty_move_counter),
                        0, ep, turn, false, &moves);
  if (!ok || moves.size == 0) {
    return result;
  }

  const TbRootMove* best = &moves.moves[0];
  for (unsigned i = 1; i < moves.size; ++i) {
    if (moves.moves[i].tbRank > best->tbRank) {
      best = &moves.moves[i];
    }
  }

  const int from = TB_MOVE_FROM(best->move);
  const int to = TB_MOVE_TO(best->move);
  const int promo = TB_MOVE_PROMOTES(best->move);

  Move move{};
  move.from = static_cast<uint16_t>(from);
  move.to = static_cast<uint16_t>(to);
  move.moving_pc = board.pieces[static_cast<std::size_t>(from)];
  move.promo_pc = promo_from_tb(promo, board.side_to_move);
  move.flags = 0;

  OccupancyType captured = board.pieces[static_cast<std::size_t>(to)];
  if (move.moving_pc == OccupancyType::wP ||
      move.moving_pc == OccupancyType::bP) {
    if (board.en_passant == to) {
      move.flags |= kFlagEnPassant;
      const int captured_sq =
          (board.side_to_move == SideToMove::White) ? (to - 8) : (to + 8);
      captured = board.pieces[static_cast<std::size_t>(captured_sq)];
    }
  }
  move.captured_pc = captured;

  result.best_move = move;
  result.wdl = static_cast<int>(best->tbScore);
  result.score = wdl_to_score(static_cast<unsigned>(best->tbScore));
  return result;
}

ProbeResult probe_root_dtz(const Board& board) {
  ProbeResult result{};
  result.available = can_probe(board);
  if (!result.available) {
    return result;
  }

  const Bitboard white =
      board.occupancy[static_cast<std::size_t>(PieceColor::White)];
  const Bitboard black =
      board.occupancy[static_cast<std::size_t>(PieceColor::Black)];
  const Bitboard kings = piece_bb(board, Piece::wK) | piece_bb(board, Piece::bK);
  const Bitboard queens =
      piece_bb(board, Piece::wQ) | piece_bb(board, Piece::bQ);
  const Bitboard rooks = piece_bb(board, Piece::wR) | piece_bb(board, Piece::bR);
  const Bitboard bishops =
      piece_bb(board, Piece::wB) | piece_bb(board, Piece::bB);
  const Bitboard knights =
      piece_bb(board, Piece::wN) | piece_bb(board, Piece::bN);
  const Bitboard pawns = piece_bb(board, Piece::wP) | piece_bb(board, Piece::bP);
  const unsigned ep =
      (board.en_passant >= 0) ? static_cast<unsigned>(board.en_passant) : 0u;
  const bool turn = (board.side_to_move == SideToMove::White);

  TbRootMoves moves{};
  const int ok =
      tb_probe_root_dtz(white, black, kings, queens, rooks, bishops, knights,
                        pawns, static_cast<unsigned>(board.fifty_move_counter),
                        0, ep, turn, false, false, &moves);
  if (!ok || moves.size == 0) {
    return result;
  }

  const TbRootMove* best = &moves.moves[0];
  for (unsigned i = 1; i < moves.size; ++i) {
    if (moves.moves[i].tbRank > best->tbRank) {
      best = &moves.moves[i];
    }
  }

  const int from = TB_MOVE_FROM(best->move);
  const int to = TB_MOVE_TO(best->move);
  const int promo = TB_MOVE_PROMOTES(best->move);

  Move move{};
  move.from = static_cast<uint16_t>(from);
  move.to = static_cast<uint16_t>(to);
  move.moving_pc = board.pieces[static_cast<std::size_t>(from)];
  move.promo_pc = promo_from_tb(promo, board.side_to_move);
  move.flags = 0;

  OccupancyType captured = board.pieces[static_cast<std::size_t>(to)];
  if (move.moving_pc == OccupancyType::wP ||
      move.moving_pc == OccupancyType::bP) {
    if (board.en_passant == to) {
      move.flags |= kFlagEnPassant;
      const int captured_sq =
          (board.side_to_move == SideToMove::White) ? (to - 8) : (to + 8);
      captured = board.pieces[static_cast<std::size_t>(captured_sq)];
    }
  }
  move.captured_pc = captured;

  result.best_move = move;
  result.wdl = static_cast<int>(best->tbScore);
  result.score = wdl_to_score(static_cast<unsigned>(best->tbScore));
  return result;
}

} // namespace chess::syzygy
