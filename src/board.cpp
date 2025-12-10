
#include "chess/board.hpp"

#include "chess/defaults.hpp"
#include "chess/types.hpp"
#include "chess/types_io.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <span>
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
  std::cout << "Set side to move: " << board.side_to_move << "\n";
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

  std::cout << "Setting up initial board position from FEN: " << fen << "\n";
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
} // namespace chess