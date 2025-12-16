#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

namespace chess {
Board parse_fen_string(std::string_view fen);
}

namespace {

chess::Board make_board(std::string_view fen) {
  chess::Board board = chess::parse_fen_string(fen);

  board.occupancy[static_cast<std::size_t>(chess::PieceColor::White)] =
      chess::calculate_occupancy(board, chess::PieceColor::White);
  board.occupancy[static_cast<std::size_t>(chess::PieceColor::Black)] =
      chess::calculate_occupancy(board, chess::PieceColor::Black);
  board.occupancy[static_cast<std::size_t>(chess::PieceColor::Both)] =
      chess::calculate_occupancy(board, chess::PieceColor::Both);

  board.pieces_bb.fill(0);
  for (std::size_t sq = 0; sq < 64; ++sq) {
    const chess::OccupancyType occ = board.pieces[sq];
    if (occ == chess::OccupancyType::empty) {
      continue;
    }
    const std::size_t piece_idx = static_cast<std::size_t>(occ) - 1;
    board.pieces_bb[piece_idx] |= (Bitboard(1) << sq);
  }

  return board;
}

inline bool list_contains(const chess::PieceList& list, chess::Square sq) {
  for (std::uint8_t i = 0; i < list.count; ++i) {
    if (list.squares[i] == sq) {
      return true;
    }
  }
  return false;
}

std::vector<std::uint32_t> collect_moves(const chess::Board& board, chess::SideToMove stm) {
  std::uint16_t move_count = 0;
  auto buffer = chess::generate_all_moves(board, stm, move_count);
  return std::vector<std::uint32_t>(buffer.begin(), buffer.begin() + move_count);
}

std::uint32_t encode_move(chess::Square from, chess::Square to, chess::OccupancyType moving,
                          chess::OccupancyType captured, chess::OccupancyType promo,
                          std::uint8_t flags) {
  return chess::encode_move(static_cast<int>(to_index(from)), static_cast<int>(to_index(to)),
                            moving, captured, promo, flags);
}

} // namespace

TEST(MoveApplication, KingsideCastleUpdatesOnlyMovingRook) {
  auto board = make_board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  const auto original = board;

  const chess::Move move{static_cast<std::uint16_t>(to_index(chess::Square::E1)),
                         static_cast<std::uint16_t>(to_index(chess::Square::G1)),
                         chess::OccupancyType::wK,
                         chess::OccupancyType::empty,
                         chess::OccupancyType::empty,
                         chess::kFlagCastle};

  const chess::Undo undo = chess::make_move(board, move);

  EXPECT_EQ(board.pieces[to_index(chess::Square::E1)], chess::OccupancyType::empty);
  EXPECT_EQ(board.pieces[to_index(chess::Square::G1)], chess::OccupancyType::wK);
  EXPECT_EQ(board.pieces[to_index(chess::Square::F1)], chess::OccupancyType::wR);
  EXPECT_EQ(board.pieces[to_index(chess::Square::H1)], chess::OccupancyType::empty);

  const auto& white_rooks = board.rook_list[to_index(chess::PieceColor::White)];
  ASSERT_EQ(white_rooks.count, 2);
  EXPECT_TRUE(list_contains(white_rooks, chess::Square::A1));
  EXPECT_TRUE(list_contains(white_rooks, chess::Square::F1));

  const auto& black_rooks = board.rook_list[to_index(chess::PieceColor::Black)];
  ASSERT_EQ(black_rooks.count, 2);
  EXPECT_TRUE(list_contains(black_rooks, chess::Square::A8));
  EXPECT_TRUE(list_contains(black_rooks, chess::Square::H8));

  chess::undo_move(board, undo);

  EXPECT_EQ(board.pieces, original.pieces);
  EXPECT_EQ(board.occupancy[0], original.occupancy[0]);
  EXPECT_EQ(board.occupancy[1], original.occupancy[1]);
  EXPECT_EQ(board.occupancy[2], original.occupancy[2]);
  EXPECT_EQ(board.rook_list[0].count, original.rook_list[0].count);
  EXPECT_EQ(board.rook_list[1].count, original.rook_list[1].count);
  EXPECT_TRUE(list_contains(board.rook_list[0], chess::Square::A1));
  EXPECT_TRUE(list_contains(board.rook_list[0], chess::Square::H1));
  EXPECT_TRUE(list_contains(board.rook_list[1], chess::Square::A8));
  EXPECT_TRUE(list_contains(board.rook_list[1], chess::Square::H8));
}

TEST(MoveApplication, BlackKingsideCastleKeepsWhiteBackRankIntact) {
  auto board = make_board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  const auto original = board;

  const chess::Move move{static_cast<std::uint16_t>(to_index(chess::Square::E8)),
                         static_cast<std::uint16_t>(to_index(chess::Square::G8)),
                         chess::OccupancyType::bK,
                         chess::OccupancyType::empty,
                         chess::OccupancyType::empty,
                         chess::kFlagCastle};

  const chess::Undo undo = chess::make_move(board, move);

  EXPECT_EQ(board.pieces[to_index(chess::Square::E8)], chess::OccupancyType::empty);
  EXPECT_EQ(board.pieces[to_index(chess::Square::G8)], chess::OccupancyType::bK);
  EXPECT_EQ(board.pieces[to_index(chess::Square::F8)], chess::OccupancyType::bR);
  EXPECT_EQ(board.pieces[to_index(chess::Square::H8)], chess::OccupancyType::empty);

  const auto& black_rooks = board.rook_list[to_index(chess::PieceColor::Black)];
  ASSERT_EQ(black_rooks.count, 2);
  EXPECT_TRUE(list_contains(black_rooks, chess::Square::A8));
  EXPECT_TRUE(list_contains(black_rooks, chess::Square::F8));

  const auto& white_rooks = board.rook_list[to_index(chess::PieceColor::White)];
  ASSERT_EQ(white_rooks.count, 2);
  EXPECT_TRUE(list_contains(white_rooks, chess::Square::A1));
  EXPECT_TRUE(list_contains(white_rooks, chess::Square::H1));

  chess::undo_move(board, undo);

  EXPECT_EQ(board.pieces, original.pieces);
  EXPECT_EQ(board.occupancy[0], original.occupancy[0]);
  EXPECT_EQ(board.occupancy[1], original.occupancy[1]);
  EXPECT_EQ(board.occupancy[2], original.occupancy[2]);
  EXPECT_TRUE(list_contains(board.rook_list[0], chess::Square::A1));
  EXPECT_TRUE(list_contains(board.rook_list[0], chess::Square::H1));
  EXPECT_TRUE(list_contains(board.rook_list[1], chess::Square::A8));
  EXPECT_TRUE(list_contains(board.rook_list[1], chess::Square::H8));
}

TEST(MoveGeneration, WhitePawnEncodesQuietDoubleAndCapture) {
  const auto board = make_board("6k1/8/8/8/8/5n1/4P3/6K1 w - - 0 1");
  const auto moves = collect_moves(board, chess::SideToMove::White);

  const auto quiet =
      encode_move(chess::Square::E2, chess::Square::E3, chess::OccupancyType::wP,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagQuiet);
  const auto double_push =
      encode_move(chess::Square::E2, chess::Square::E4, chess::OccupancyType::wP,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagDoublePush);
  const auto capture = encode_move(chess::Square::E2, chess::Square::F3, chess::OccupancyType::wP,
                                   chess::OccupancyType::bN, chess::OccupancyType::empty, 0);

  const auto it_quiet = std::find(moves.begin(), moves.end(), quiet);
  ASSERT_NE(it_quiet, moves.end());
  const auto decoded_quiet = chess::decode_move(*it_quiet);
  EXPECT_EQ(decoded_quiet.from, static_cast<std::uint16_t>(to_index(chess::Square::E2)));
  EXPECT_EQ(decoded_quiet.to, static_cast<std::uint16_t>(to_index(chess::Square::E3)));
  EXPECT_EQ(decoded_quiet.moving_pc, chess::OccupancyType::wP);
  EXPECT_EQ(decoded_quiet.captured_pc, chess::OccupancyType::empty);
  EXPECT_EQ(decoded_quiet.flags, chess::kFlagQuiet);

  const auto it_double = std::find(moves.begin(), moves.end(), double_push);
  ASSERT_NE(it_double, moves.end());
  const auto decoded_double = chess::decode_move(*it_double);
  EXPECT_EQ(decoded_double.to, static_cast<std::uint16_t>(to_index(chess::Square::E4)));
  EXPECT_EQ(decoded_double.flags, chess::kFlagDoublePush);

  const auto it_capture = std::find(moves.begin(), moves.end(), capture);
  ASSERT_NE(it_capture, moves.end());
  const auto decoded_capture = chess::decode_move(*it_capture);
  EXPECT_EQ(decoded_capture.to, static_cast<std::uint16_t>(to_index(chess::Square::F3)));
  EXPECT_EQ(decoded_capture.captured_pc, chess::OccupancyType::bN);
  EXPECT_EQ(decoded_capture.flags, 0);
}

TEST(MoveGeneration, BlackPawnEncodesQuietDoubleAndCapture) {
  const auto board = make_board("k7/4p3/3N4/8/8/8/8/4K3 b - - 0 1");
  const auto moves = collect_moves(board, chess::SideToMove::Black);

  const auto quiet =
      encode_move(chess::Square::E7, chess::Square::E6, chess::OccupancyType::bP,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagQuiet);
  const auto double_push =
      encode_move(chess::Square::E7, chess::Square::E5, chess::OccupancyType::bP,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagDoublePush);
  const auto capture = encode_move(chess::Square::E7, chess::Square::D6, chess::OccupancyType::bP,
                                   chess::OccupancyType::wN, chess::OccupancyType::empty, 0);

  const auto it_quiet = std::find(moves.begin(), moves.end(), quiet);
  ASSERT_NE(it_quiet, moves.end());
  const auto decoded_quiet = chess::decode_move(*it_quiet);
  EXPECT_EQ(decoded_quiet.from, static_cast<std::uint16_t>(to_index(chess::Square::E7)));
  EXPECT_EQ(decoded_quiet.to, static_cast<std::uint16_t>(to_index(chess::Square::E6)));
  EXPECT_EQ(decoded_quiet.moving_pc, chess::OccupancyType::bP);
  EXPECT_EQ(decoded_quiet.flags, chess::kFlagQuiet);

  const auto it_double = std::find(moves.begin(), moves.end(), double_push);
  ASSERT_NE(it_double, moves.end());
  const auto decoded_double = chess::decode_move(*it_double);
  EXPECT_EQ(decoded_double.to, static_cast<std::uint16_t>(to_index(chess::Square::E5)));
  EXPECT_EQ(decoded_double.flags, chess::kFlagDoublePush);

  const auto it_capture = std::find(moves.begin(), moves.end(), capture);
  ASSERT_NE(it_capture, moves.end());
  const auto decoded_capture = chess::decode_move(*it_capture);
  EXPECT_EQ(decoded_capture.to, static_cast<std::uint16_t>(to_index(chess::Square::D6)));
  EXPECT_EQ(decoded_capture.captured_pc, chess::OccupancyType::wN);
  EXPECT_EQ(decoded_capture.flags, 0);
}

TEST(MoveGeneration, WhiteKnightEncodesQuietAndCapture) {
  const auto board = make_board("k7/8/4p3/8/3N4/8/8/6K1 w - - 0 1");
  const auto moves = collect_moves(board, chess::SideToMove::White);

  const auto quiet =
      encode_move(chess::Square::D4, chess::Square::F5, chess::OccupancyType::wN,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagQuiet);
  const auto capture = encode_move(chess::Square::D4, chess::Square::E6, chess::OccupancyType::wN,
                                   chess::OccupancyType::bP, chess::OccupancyType::empty, 0);

  const auto it_quiet = std::find(moves.begin(), moves.end(), quiet);
  ASSERT_NE(it_quiet, moves.end());
  const auto decoded_quiet = chess::decode_move(*it_quiet);
  EXPECT_EQ(decoded_quiet.moving_pc, chess::OccupancyType::wN);
  EXPECT_EQ(decoded_quiet.flags, chess::kFlagQuiet);

  const auto it_capture = std::find(moves.begin(), moves.end(), capture);
  ASSERT_NE(it_capture, moves.end());
  const auto decoded_capture = chess::decode_move(*it_capture);
  EXPECT_EQ(decoded_capture.captured_pc, chess::OccupancyType::bP);
  EXPECT_EQ(decoded_capture.flags, 0);
}

TEST(MoveGeneration, WhiteBishopEncodesQuietAndCapture) {
  const auto board = make_board("k7/6p1/8/8/8/2B5/8/6K1 w - - 0 1");
  const auto moves = collect_moves(board, chess::SideToMove::White);

  const auto quiet =
      encode_move(chess::Square::C3, chess::Square::F6, chess::OccupancyType::wB,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagQuiet);
  const auto capture = encode_move(chess::Square::C3, chess::Square::G7, chess::OccupancyType::wB,
                                   chess::OccupancyType::bP, chess::OccupancyType::empty, 0);

  const auto it_quiet = std::find(moves.begin(), moves.end(), quiet);
  ASSERT_NE(it_quiet, moves.end());
  const auto decoded_quiet = chess::decode_move(*it_quiet);
  EXPECT_EQ(decoded_quiet.to, static_cast<std::uint16_t>(to_index(chess::Square::F6)));
  EXPECT_EQ(decoded_quiet.flags, chess::kFlagQuiet);

  const auto it_capture = std::find(moves.begin(), moves.end(), capture);
  ASSERT_NE(it_capture, moves.end());
  const auto decoded_capture = chess::decode_move(*it_capture);
  EXPECT_EQ(decoded_capture.to, static_cast<std::uint16_t>(to_index(chess::Square::G7)));
  EXPECT_EQ(decoded_capture.captured_pc, chess::OccupancyType::bP);
}

TEST(MoveGeneration, WhiteRookEncodesQuietAndCapture) {
  const auto board = make_board("k7/3p4/8/8/3R4/8/8/6K1 w - - 0 1");
  const auto moves = collect_moves(board, chess::SideToMove::White);

  const auto quiet =
      encode_move(chess::Square::D4, chess::Square::D6, chess::OccupancyType::wR,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagQuiet);
  const auto capture = encode_move(chess::Square::D4, chess::Square::D7, chess::OccupancyType::wR,
                                   chess::OccupancyType::bP, chess::OccupancyType::empty, 0);

  const auto it_quiet = std::find(moves.begin(), moves.end(), quiet);
  ASSERT_NE(it_quiet, moves.end());
  const auto decoded_quiet = chess::decode_move(*it_quiet);
  EXPECT_EQ(decoded_quiet.to, static_cast<std::uint16_t>(to_index(chess::Square::D6)));
  EXPECT_EQ(decoded_quiet.flags, chess::kFlagQuiet);

  const auto it_capture = std::find(moves.begin(), moves.end(), capture);
  ASSERT_NE(it_capture, moves.end());
  const auto decoded_capture = chess::decode_move(*it_capture);
  EXPECT_EQ(decoded_capture.to, static_cast<std::uint16_t>(to_index(chess::Square::D7)));
  EXPECT_EQ(decoded_capture.captured_pc, chess::OccupancyType::bP);
}

TEST(MoveGeneration, WhiteQueenEncodesQuietAndCapture) {
  const auto board = make_board("k7/6n1/8/8/3Q4/8/8/6K1 w - - 0 1");
  const auto moves = collect_moves(board, chess::SideToMove::White);

  const auto quiet =
      encode_move(chess::Square::D4, chess::Square::H4, chess::OccupancyType::wQ,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagQuiet);
  const auto capture = encode_move(chess::Square::D4, chess::Square::G7, chess::OccupancyType::wQ,
                                   chess::OccupancyType::bN, chess::OccupancyType::empty, 0);

  const auto it_quiet = std::find(moves.begin(), moves.end(), quiet);
  ASSERT_NE(it_quiet, moves.end());
  const auto decoded_quiet = chess::decode_move(*it_quiet);
  EXPECT_EQ(decoded_quiet.to, static_cast<std::uint16_t>(to_index(chess::Square::H4)));
  EXPECT_EQ(decoded_quiet.flags, chess::kFlagQuiet);

  const auto it_capture = std::find(moves.begin(), moves.end(), capture);
  ASSERT_NE(it_capture, moves.end());
  const auto decoded_capture = chess::decode_move(*it_capture);
  EXPECT_EQ(decoded_capture.to, static_cast<std::uint16_t>(to_index(chess::Square::G7)));
  EXPECT_EQ(decoded_capture.captured_pc, chess::OccupancyType::bN);
}

TEST(MoveGeneration, WhiteKingEncodesCapture) {
  const auto board = make_board("k7/8/8/4p3/4K3/8/8/8 w - - 0 1");
  const auto moves = collect_moves(board, chess::SideToMove::White);

  const auto capture = encode_move(chess::Square::E4, chess::Square::E5, chess::OccupancyType::wK,
                                   chess::OccupancyType::bP, chess::OccupancyType::empty, 0);
  const auto it_capture = std::find(moves.begin(), moves.end(), capture);
  ASSERT_NE(it_capture, moves.end());
  const auto decoded_capture = chess::decode_move(*it_capture);
  EXPECT_EQ(decoded_capture.moving_pc, chess::OccupancyType::wK);
  EXPECT_EQ(decoded_capture.to, static_cast<std::uint16_t>(to_index(chess::Square::E5)));
  EXPECT_EQ(decoded_capture.captured_pc, chess::OccupancyType::bP);
}

TEST(MoveApplication, RookQuietMoveDoesNotCreateExtraQueen) {
  auto board = make_board("rnb1kbnr/1ppp1ppp/8/4p3/7q/PpPPP1P1/5P1P/RNBQKBNR w KQkq - 0 1");

  const chess::Move move{static_cast<std::uint16_t>(to_index(chess::Square::A1)),
                         static_cast<std::uint16_t>(to_index(chess::Square::A2)),
                         chess::OccupancyType::wR,
                         chess::OccupancyType::empty,
                         chess::OccupancyType::empty,
                         chess::kFlagQuiet};

  chess::make_move(board, move);

  EXPECT_EQ(board.pieces[to_index(chess::Square::A1)], chess::OccupancyType::empty);
  EXPECT_EQ(board.pieces[to_index(chess::Square::A2)], chess::OccupancyType::wR);
  EXPECT_EQ(board.pieces[to_index(chess::Square::C2)], chess::OccupancyType::empty);
  EXPECT_EQ(board.side_to_move, chess::SideToMove::Black);
  EXPECT_EQ(chess::board_to_fen(board),
            "rnb1kbnr/1ppp1ppp/8/4p3/7q/PpPPP1P1/R4P1P/1NBQKBNR b Kkq - 1 1");
}

TEST(MoveApplication, QueenCaptureUpdatesBitboard) {
  auto board = make_board("rnb1kbnr/pppp1ppp/8/4p3/7q/PP1P4/2P1PPPP/RNBQKBNR b KQkq - 0 1");

  const chess::Move qxg3{static_cast<std::uint16_t>(to_index(chess::Square::H4)),
                         static_cast<std::uint16_t>(to_index(chess::Square::G3)),
                         chess::OccupancyType::bQ,
                         chess::OccupancyType::wP,
                         chess::OccupancyType::empty,
                         0};

  chess::make_move(board, qxg3);

  EXPECT_EQ(board.pieces[to_index(chess::Square::H4)], chess::OccupancyType::empty);
  EXPECT_EQ(board.pieces[to_index(chess::Square::G3)], chess::OccupancyType::bQ);

  const auto queen_bb = board.pieces_bb[static_cast<std::size_t>(chess::OccupancyType::bQ) - 1];
  const Bitboard expected = Bitboard(1) << to_index(chess::Square::G3);
  EXPECT_EQ(queen_bb, expected);
}

TEST(MoveApplication, QueenCaptureAndUndoRestoresState) {
  auto board = make_board("rnb1kbnr/pppp1ppp/8/4p3/7q/PP1P4/2P1PPPP/RNBQKBNR b KQkq - 0 1");

  const chess::Move qxf2{static_cast<std::uint16_t>(to_index(chess::Square::H4)),
                         static_cast<std::uint16_t>(to_index(chess::Square::F2)),
                         chess::OccupancyType::bQ,
                         chess::OccupancyType::wP,
                         chess::OccupancyType::empty,
                         0};

  const auto undo = chess::make_move(board, qxf2);

  EXPECT_EQ(board.pieces[to_index(chess::Square::F2)], chess::OccupancyType::bQ);
  EXPECT_EQ(board.pieces[to_index(chess::Square::H4)], chess::OccupancyType::empty);

  chess::undo_move(board, undo);

  EXPECT_EQ(board.pieces[to_index(chess::Square::F2)], chess::OccupancyType::wP);
  EXPECT_EQ(board.pieces[to_index(chess::Square::H4)], chess::OccupancyType::bQ);

  const auto queen_bb = board.pieces_bb[static_cast<std::size_t>(chess::OccupancyType::bQ) - 1];
  const Bitboard expected = Bitboard(1) << to_index(chess::Square::H4);
  EXPECT_EQ(queen_bb, expected);
}

TEST(MoveApplication, DoublePushCreatesEnPassantTargetForOpponent) {
  auto board = make_board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1");

  const chess::Move d5{static_cast<std::uint16_t>(to_index(chess::Square::D7)),
                       static_cast<std::uint16_t>(to_index(chess::Square::D5)),
                       chess::OccupancyType::bP,
                       chess::OccupancyType::empty,
                       chess::OccupancyType::empty,
                       chess::kFlagDoublePush};

  const auto undo = chess::make_move(board, d5);

  EXPECT_EQ(board.side_to_move, chess::SideToMove::White);
  EXPECT_EQ(board.en_passant, to_index(chess::Square::D6));
  EXPECT_EQ(board.ep_square, chess::bb_of(to_index(chess::Square::D6)));

  std::uint16_t move_count = 0;
  auto legal = chess::generate_legal_moves(board, chess::SideToMove::White, move_count);
  const auto ep_move =
      encode_move(chess::Square::E5, chess::Square::D6, chess::OccupancyType::wP,
                  chess::OccupancyType::empty, chess::OccupancyType::empty, chess::kFlagEnPassant);
  const auto it = std::find(legal.begin(), legal.begin() + move_count, ep_move);
  EXPECT_NE(it, legal.begin() + move_count);

  chess::undo_move(board, undo);
  EXPECT_EQ(board.en_passant, -1);
  EXPECT_EQ(board.ep_square, 0);
}

TEST(MoveApplication, CapturingKingMarksStateAndUndoesCleanly) {
  auto board = make_board("4k3/8/8/8/4Q3/8/8/4K3 w - - 0 1");
  const auto original_black_king = board.king_positions[to_index(chess::PieceColor::Black)];

  const chess::Move qxe8{static_cast<std::uint16_t>(to_index(chess::Square::E4)),
                         static_cast<std::uint16_t>(to_index(chess::Square::E8)),
                         chess::OccupancyType::wQ,
                         chess::OccupancyType::bK,
                         chess::OccupancyType::empty,
                         0};

  const auto undo = chess::make_move(board, qxe8);

  EXPECT_EQ(board.pieces[to_index(chess::Square::E4)], chess::OccupancyType::empty);
  EXPECT_EQ(board.pieces[to_index(chess::Square::E8)], chess::OccupancyType::wQ);
  EXPECT_EQ(board.king_captured, chess::PieceColor::Black);
  EXPECT_EQ(board.king_positions[to_index(chess::PieceColor::Black)], -1);

  chess::undo_move(board, undo);

  EXPECT_EQ(board.pieces[to_index(chess::Square::E4)], chess::OccupancyType::wQ);
  EXPECT_EQ(board.pieces[to_index(chess::Square::E8)], chess::OccupancyType::bK);
  EXPECT_EQ(board.king_captured, chess::PieceColor::None);
  EXPECT_EQ(board.king_positions[to_index(chess::PieceColor::Black)], original_black_king);
}

TEST(MoveApplication, QuietPromotionUndoRestoresState) {
  auto board = make_board("k7/8/8/8/8/8/p7/1K6 b - - 0 1");
  const auto original = board;

  const chess::Move promotion{static_cast<std::uint16_t>(to_index(chess::Square::A2)),
                              static_cast<std::uint16_t>(to_index(chess::Square::A1)),
                              chess::OccupancyType::bP,
                              chess::OccupancyType::empty,
                              chess::OccupancyType::bQ,
                              0};

  const chess::Undo undo = chess::make_move(board, promotion);

  EXPECT_EQ(board.pieces[to_index(chess::Square::A2)], chess::OccupancyType::empty);
  EXPECT_EQ(board.pieces[to_index(chess::Square::A1)], chess::OccupancyType::bQ);

  const Bitboard queen_after =
      board.pieces_bb[static_cast<std::size_t>(chess::OccupancyType::bQ) - 1];
  const Bitboard expected_after = Bitboard(1) << to_index(chess::Square::A1);
  EXPECT_EQ(queen_after, expected_after);

  chess::undo_move(board, undo);

  EXPECT_EQ(board.pieces, original.pieces);
  EXPECT_EQ(board.occupancy[0], original.occupancy[0]);
  EXPECT_EQ(board.occupancy[1], original.occupancy[1]);
  EXPECT_EQ(board.occupancy[2], original.occupancy[2]);
  EXPECT_EQ(board.pieces_bb, original.pieces_bb);
}
