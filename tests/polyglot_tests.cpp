#include "chess/polyglot.hpp"
#include "chess/board.hpp"

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <sstream>
#include <string_view>

namespace {

struct PolyglotCase {
  std::string_view fen;
  std::string_view hash_hex;
};

constexpr std::array<PolyglotCase, 9> kPolyglotCases{{
    PolyglotCase{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "463b96181691fc9c"},
    PolyglotCase{"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "823c9b50fd114196"},
    PolyglotCase{"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "0756b94461c50fb0"},
    PolyglotCase{"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2", "662fafb965db29d4"},
    PolyglotCase{"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "22a48b5a8e47ff78"},
    PolyglotCase{"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR b kq - 0 3", "652a607ca3f242c1"},
    PolyglotCase{"rnbq1bnr/ppp1pkpp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR w - - 0 4", "00fdd303c946bdd9"},
    PolyglotCase{"rnbqkbnr/p1pppppp/8/8/PpP4P/8/1P1PPPP1/RNBQKBNR b KQkq c3 0 3", "3c8123ea7b067637"},
    PolyglotCase{"rnbqkbnr/p1pppppp/8/8/P6P/R1p5/1P1PPPP1/1NBQKBNR b Kkq - 0 4", "5c3f9b829b279560"},
}};

uint64_t parse_hex(std::string_view hex) {
  std::stringstream ss;
  ss << std::hex << std::string(hex);
  uint64_t value = 0;
  ss >> value;
  return value;
}

} // namespace

TEST(Polyglot, MatchesReferenceHashes) {
  for (const auto& test_case : kPolyglotCases) {
    const auto board = chess::initial_board(test_case.fen);
    const auto computed = chess::polyglot::compute_key(board);
    EXPECT_EQ(computed, parse_hex(test_case.hash_hex)) << "FEN: " << test_case.fen;
  }
}
