
#include "chess/board.hpp"

#include "chess/attack_masks.hpp"
#include "chess/defaults.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"
#include "chess/zobrist.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

namespace chess {

constexpr std::array<OccupancyType, 128> kPieceFromChar = [] {
  std::array<OccupancyType, 128> arr{};
  arr.fill(OccupancyType::empty);
  arr[static_cast<std::size_t>('P')] = OccupancyType::wP;
  arr[static_cast<std::size_t>('N')] = OccupancyType::wN;
  arr[static_cast<std::size_t>('B')] = OccupancyType::wB;
  arr[static_cast<std::size_t>('R')] = OccupancyType::wR;
  arr[static_cast<std::size_t>('Q')] = OccupancyType::wQ;
  arr[static_cast<std::size_t>('K')] = OccupancyType::wK;
  arr[static_cast<std::size_t>('p')] = OccupancyType::bP;
  arr[static_cast<std::size_t>('n')] = OccupancyType::bN;
  arr[static_cast<std::size_t>('b')] = OccupancyType::bB;
  arr[static_cast<std::size_t>('r')] = OccupancyType::bR;
  arr[static_cast<std::size_t>('q')] = OccupancyType::bQ;
  arr[static_cast<std::size_t>('k')] = OccupancyType::bK;
  return arr;
}();

constexpr std::array<std::string_view, 13> kPieceGlyph = {"·", "♙", "♘", "♗", "♖", "♕", "♔",
                                                          "♟", "♞", "♝", "♜", "♛", "♚"};

Board parse_fen_string(std::string_view fen) {
  Board board{};
  std::size_t pos = 0;
  std::size_t rank = 7;
  std::size_t file = 0;

  std::cout << "Parsing FEN: " << fen << "\n";
  FenFields fields = split_fen(fen);

  board.rook_list[0] = PieceList{};
  board.rook_list[1] = PieceList{};
  board.king_list[0] = PieceList{};
  board.king_list[1] = PieceList{};
  for (char c : fields.placement) {
    if (c == '/') {
      --rank;
      file = 0;
    } else if (std::isdigit(static_cast<unsigned char>(c))) {
      file += static_cast<std::size_t>(c - '0');
    } else {
      // Map character to OccupancyType and set board.pieces[rank * 8 + file]
      pos = rank * 8 + file;
      board.pieces[pos] = kPieceFromChar[static_cast<unsigned char>(c)];
      // piece list
      if (board.pieces[pos] == OccupancyType::wP) {
        board.pawn_list[0].squares[board.pawn_list[0].count++] = static_cast<Square>(pos);
      } else if (board.pieces[pos] == OccupancyType::bP) {
        board.pawn_list[1].squares[board.pawn_list[1].count++] = static_cast<Square>(pos);
      } else if (board.pieces[pos] == OccupancyType::wR) {
        board.rook_list[0].squares[board.rook_list[0].count++] = static_cast<Square>(pos);
      } else if (board.pieces[pos] == OccupancyType::bR) {
        board.rook_list[1].squares[board.rook_list[1].count++] = static_cast<Square>(pos);
      } else if (board.pieces[pos] == OccupancyType::wK) {
        board.king_list[0].squares[board.king_list[0].count++] = static_cast<Square>(pos);
        board.king_positions[0] = static_cast<int>(pos);
      } else if (board.pieces[pos] == OccupancyType::bK) {
        board.king_list[1].squares[board.king_list[1].count++] = static_cast<Square>(pos);
        board.king_positions[1] = static_cast<int>(pos);
      }

      ++file;
    }
  }

  board.castling_rights = fields.castling.empty()
                              ? CastlingRights::NoCastling
                              : static_cast<CastlingRights>(
                                    (fields.castling.find('K') != std::string_view::npos ? 1 : 0) +
                                    (fields.castling.find('Q') != std::string_view::npos ? 2 : 0) +
                                    (fields.castling.find('k') != std::string_view::npos ? 4 : 0) +
                                    (fields.castling.find('q') != std::string_view::npos ? 8 : 0));
  board.en_passant = (fields.en_passant == "-")
                         ? -1
                         : (fields.en_passant[0] - 'a') + (fields.en_passant[1] - '1') * 8;
  board.ply_count =
      fields.fullmove_number.empty() ? 1 : std::stoi(std::string(fields.fullmove_number));
  board.fifty_move_counter =
      fields.halfmove_clock.empty() ? 0 : std::stoi(std::string(fields.halfmove_clock));
  board.side_to_move = (fields.side_to_move == "w") ? SideToMove::White : SideToMove::Black;

  // Minimal FEN parsing logic can be added here
  return board;
}

Board initial_board(std::string_view fen) {
  Board board{};

  // Very minimal FEN parsing for initial position only
  if (fen != kStartFEN) {
    throw std::invalid_argument("Only starting FEN is supported in this minimal parser.");
  }
  board = parse_fen_string(fen);
  board.occupancy[static_cast<std::size_t>(PieceColor::White)] =
      calculate_occupancy(board, PieceColor::White);
  board.occupancy[static_cast<std::size_t>(PieceColor::Black)] =
      calculate_occupancy(board, PieceColor::Black);
  board.occupancy[static_cast<std::size_t>(PieceColor::Both)] =
      calculate_occupancy(board, PieceColor::Both);

  for (std::size_t sq = 0; sq < 64; ++sq) {
    const OccupancyType occ = board.pieces[static_cast<std::size_t>(sq)];
    if (occ == OccupancyType::empty)
      continue;
    const std::size_t piece_idx = static_cast<std::size_t>(occ) - 1;
    board.pieces_bb[piece_idx] |= (Bitboard(1) << sq);
  }

  init_zobrist(0xDEADBEEF); // example seed
  board.position_key = compute_position_key(board);

  std::cout << "Set up initial board position from FEN: " << fen << "\n";

  return board;
}

Bitboard calculate_occupancy(const Board& board, PieceColor color) {
  Bitboard occ = 0;
  for (std::size_t sq = 0; sq < 64; ++sq) {
    const OccupancyType piece = board.pieces[sq];
    if (piece != OccupancyType::empty) {
      bool is_W = is_white(piece);
      if ((color == PieceColor::White && is_W) || (color == PieceColor::Black && !is_W) ||
          (color == PieceColor::Both)) {
        occ |= (Bitboard(1) << sq);
      }
    }
  }
  return occ;
}

void terminal_board_print(const Board& board) {
  for (int rank = 7; rank >= 0; --rank) {
    std::cout << rank + 1 << " | ";
    for (int file = 0; file < 8; ++file) {
      const std::size_t sq = static_cast<std::size_t>(rank * 8 + file);
      const OccupancyType occ = board.pieces[sq];
      const auto glyph = kPieceGlyph[static_cast<std::size_t>(occ)];
      std::cout << glyph << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "  ------------------------\n";
  std::cout << "    a  b  c  d  e  f  g  h\n";
}

void terminal_mask_print(Bitboard mask, const Board& board) {
  for (int rank = 7; rank >= 0; --rank) {
    std::cout << rank + 1 << " | ";
    for (int file = 0; file < 8; ++file) {
      const std::size_t sq = static_cast<std::size_t>(rank * 8 + file);
      const bool is_set = (mask >> sq) & 1;
      const OccupancyType occ = board.pieces[sq];
      const auto glyph = kPieceGlyph[static_cast<std::size_t>(occ)];
      std::cout << (is_set ? "X" : glyph) << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "  ------------------------\n";
  std::cout << "    a  b  c  d  e  f  g  h\n";
}

namespace {

char occupancy_to_char(OccupancyType occ) {
  switch (occ) {
  case OccupancyType::wP:
    return 'P';
  case OccupancyType::wN:
    return 'N';
  case OccupancyType::wB:
    return 'B';
  case OccupancyType::wR:
    return 'R';
  case OccupancyType::wQ:
    return 'Q';
  case OccupancyType::wK:
    return 'K';
  case OccupancyType::bP:
    return 'p';
  case OccupancyType::bN:
    return 'n';
  case OccupancyType::bB:
    return 'b';
  case OccupancyType::bR:
    return 'r';
  case OccupancyType::bQ:
    return 'q';
  case OccupancyType::bK:
    return 'k';
  default:
    return '?';
  }
}

} // namespace

std::string board_to_fen(const Board& board) {
  std::ostringstream fen;
  for (int rank = 7; rank >= 0; --rank) {
    int empty_count = 0;
    for (int file = 0; file < 8; ++file) {
      const std::size_t sq = static_cast<std::size_t>(rank * 8 + file);
      const OccupancyType occ = board.pieces[sq];
      if (occ == OccupancyType::empty) {
        ++empty_count;
      } else {
        if (empty_count > 0) {
          fen << empty_count;
          empty_count = 0;
        }
        fen << occupancy_to_char(occ);
      }
    }
    if (empty_count > 0) {
      fen << empty_count;
    }
    if (rank > 0) {
      fen << '/';
    }
  }

  fen << ' ' << (board.side_to_move == SideToMove::White ? 'w' : 'b') << ' ';

  const int rights_mask = to_mask(board.castling_rights);
  std::string castling;
  if (rights_mask & to_mask(CastlingRights::WhiteKingside))
    castling.push_back('K');
  if (rights_mask & to_mask(CastlingRights::WhiteQueenside))
    castling.push_back('Q');
  if (rights_mask & to_mask(CastlingRights::BlackKingside))
    castling.push_back('k');
  if (rights_mask & to_mask(CastlingRights::BlackQueenside))
    castling.push_back('q');
  fen << (castling.empty() ? std::string{"-"} : castling) << ' ';

  if (board.en_passant >= 0) {
    fen << square_to_string(board.en_passant);
  } else {
    fen << '-';
  }

  fen << ' ' << board.fifty_move_counter << ' ' << board.ply_count;

  return fen.str();
}

// Set EP square after a legal double pawn push; clear otherwise.
// Call this once per make_move, after you've decided the move is legal.
inline void set_or_clear_en_passant(Board& b, int from, int to, bool is_pawn_move) {
  b.ep_square = 0;
  b.en_passant = -1;

  if (!is_pawn_move)
    return;

  // Two-rank advance (rank difference of 2)
  const int rFrom = rank_of(from);
  const int rTo = rank_of(to);
  if (std::abs(rTo - rFrom) != 2)
    return;

  // Middle square between from and to
  const int mid = (from + to) / 2;
  b.en_passant = mid;
  b.ep_square = bb_of(mid);
}

// True if the side to move has at least one pawn that could capture EP right now.
// This uses only ranks/files, so it doesn't depend on your internal square numbering direction.
bool ep_capture_available(const Board& b) {
  const int ep = b.en_passant;
  if (ep < 0)
    return false;

  const int epFile = file_of(ep);
  const int epRank = rank_of(ep);

  if (b.side_to_move == SideToMove::White) {
    // White would capture up to epRank; their pawn must be on epRank - 1 and adjacent file.
    const int reqRank = epRank - 1;
    if (reqRank < 0)
      return false;

    // Check left-adjacent pawn (file - 1)
    if (epFile > 0) {
      const int sq = (reqRank << 3) | (epFile - 1);
      if (is_white_pawn(b.pieces[to_index(sq)]))
        return true;
    }
    // Check right-adjacent pawn (file + 1)
    if (epFile < 7) {
      const int sq = (reqRank << 3) | (epFile + 1);
      if (is_white_pawn(b.pieces[to_index(sq)]))
        return true;
    }
    return false;
  } else {
    // Black would capture down to epRank; their pawn must be on epRank + 1 and adjacent file.
    const int reqRank = epRank + 1;
    if (reqRank > 7)
      return false;

    if (epFile > 0) {
      const int sq = (reqRank << 3) | (epFile - 1);
      if (is_black_pawn(b.pieces[to_index(sq)]))
        return true;
    }
    if (epFile < 7) {
      const int sq = (reqRank << 3) | (epFile + 1);
      if (is_black_pawn(b.pieces[to_index(sq)]))
        return true;
    }
    return false;
  }
}

bool is_check(const Board& b, SideToMove stm) {

  const int king_sq = b.king_positions[static_cast<std::size_t>(stm == SideToMove::White ? 0 : 1)];
  return is_square_attacked(b, static_cast<std::uint8_t>(king_sq), flip_side(stm));
}

bool has_legal_moves(Board& b, SideToMove stm) {
  uint16_t move_count = 0;
  auto _ = generate_legal_moves(b, stm, move_count);
  return move_count > 0;
}
} // namespace chess