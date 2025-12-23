#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/casteling.hpp"
#include "chess/defaults.hpp"
#include "chess/engine.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

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

} // namespace

// TEST(EngineTests, DefaultEvaluationIsZero) {
//   chess::Engine engine;
//   auto board = make_board(chess::kStartFEN);
//   EXPECT_EQ(engine.evaluate(board), 0);
// }

TEST(CastlingTests, AllowsBothSidesWhenPathClearAndSafe) {
  const auto board = make_board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");

  const auto white_rights =
      chess::king_castle_rights(board, chess::SideToMove::White);
  const auto black_rights =
      chess::king_castle_rights(board, chess::SideToMove::Black);

  EXPECT_EQ(white_rights, chess::WK | chess::WQ);
  EXPECT_EQ(black_rights, chess::BK | chess::BQ);
}

TEST(CastlingTests, BlocksSideWhenPathOccupied) {
  const auto board = make_board("r3k2r/8/8/8/8/8/8/R3K1NR w KQkq - 0 1");

  const auto white_rights =
      chess::king_castle_rights(board, chess::SideToMove::White);
  const auto black_rights =
      chess::king_castle_rights(board, chess::SideToMove::Black);

  EXPECT_EQ(white_rights, chess::WQ);
  EXPECT_EQ(black_rights, chess::BK | chess::BQ);
}

TEST(CastlingTests, DisallowsKingsideWhenSquaresAttacked) {
  const auto board = make_board("r3k2r/8/8/8/2b5/8/8/R3K2R w KQkq - 0 1");

  const auto rights = chess::king_castle_rights(board, chess::SideToMove::White);

  EXPECT_FALSE(chess::has_rights(rights, chess::WK));
  EXPECT_TRUE(chess::has_rights(rights, chess::WQ));
}

TEST(CastlingTests, DisallowsQueensideWhenSquaresAttacked) {
  const auto board = make_board("r3k2r/8/8/3r4/8/8/8/R3K2R w KQkq - 0 1");

  const auto rights = chess::king_castle_rights(board, chess::SideToMove::White);

  EXPECT_TRUE(chess::has_rights(rights, chess::WK));
  EXPECT_FALSE(chess::has_rights(rights, chess::WQ));
}

TEST(CastlingTests, MirrorsForBlackWhenAttacked) {
  const auto board = make_board("r3k2r/8/8/3B4/8/8/8/R3K2R b KQkq - 0 1");

  const auto rights = chess::king_castle_rights(board, chess::SideToMove::Black);

  EXPECT_FALSE(chess::has_rights(rights, chess::BK));
  EXPECT_TRUE(chess::has_rights(rights, chess::BQ));
}

TEST(AttackMasksTests, KingAttackPatternsMatchReference) {
  const auto attacks = chess::build_king_attack_patterns();
  constexpr std::array<std::uint64_t, 64> kExpected{770ULL,
                                                    1797ULL,
                                                    3594ULL,
                                                    7188ULL,
                                                    14376ULL,
                                                    28752ULL,
                                                    57504ULL,
                                                    49216ULL,
                                                    197123ULL,
                                                    460039ULL,
                                                    920078ULL,
                                                    1840156ULL,
                                                    3680312ULL,
                                                    7360624ULL,
                                                    14721248ULL,
                                                    12599488ULL,
                                                    50463488ULL,
                                                    117769984ULL,
                                                    235539968ULL,
                                                    471079936ULL,
                                                    942159872ULL,
                                                    1884319744ULL,
                                                    3768639488ULL,
                                                    3225468928ULL,
                                                    12918652928ULL,
                                                    30149115904ULL,
                                                    60298231808ULL,
                                                    120596463616ULL,
                                                    241192927232ULL,
                                                    482385854464ULL,
                                                    964771708928ULL,
                                                    825720045568ULL,
                                                    3307175149568ULL,
                                                    7718173671424ULL,
                                                    15436347342848ULL,
                                                    30872694685696ULL,
                                                    61745389371392ULL,
                                                    123490778742784ULL,
                                                    246981557485568ULL,
                                                    211384331665408ULL,
                                                    846636838289408ULL,
                                                    1975852459884544ULL,
                                                    3951704919769088ULL,
                                                    7903409839538176ULL,
                                                    15806819679076352ULL,
                                                    31613639358152704ULL,
                                                    63227278716305408ULL,
                                                    54114388906344448ULL,
                                                    216739030602088448ULL,
                                                    505818229730443264ULL,
                                                    1011636459460886528ULL,
                                                    2023272918921773056ULL,
                                                    4046545837843546112ULL,
                                                    8093091675687092224ULL,
                                                    16186183351374184448ULL,
                                                    13853283560024178688ULL,
                                                    144959613005987840ULL,
                                                    362258295026614272ULL,
                                                    724516590053228544ULL,
                                                    1449033180106457088ULL,
                                                    2898066360212914176ULL,
                                                    5796132720425828352ULL,
                                                    11592265440851656704ULL,
                                                    4665729213955833856ULL};

  EXPECT_EQ(attacks, kExpected);
}
