#include "sf_eval.hpp"

#include "chess/board.hpp"
#include "chess/pst_tables.hpp"

#include <array>

namespace sf_eval {

namespace {

constexpr std::array<int, 7> kSunfishMaterial = {0,   100, 280,  320,
                                                 479, 929, 60000};

} // namespace

int8_t encode_piece(chess::OccupancyType occ) noexcept {
  switch (occ) {
  case chess::OccupancyType::wP:
    return 1;
  case chess::OccupancyType::wN:
    return 2;
  case chess::OccupancyType::wB:
    return 3;
  case chess::OccupancyType::wR:
    return 4;
  case chess::OccupancyType::wQ:
    return 5;
  case chess::OccupancyType::wK:
    return 6;
  case chess::OccupancyType::bP:
    return -1;
  case chess::OccupancyType::bN:
    return -2;
  case chess::OccupancyType::bB:
    return -3;
  case chess::OccupancyType::bR:
    return -4;
  case chess::OccupancyType::bQ:
    return -5;
  case chess::OccupancyType::bK:
    return -6;
  default:
    return 0;
  }
}

bool parse_fen(const std::string& fen, Board& out_board) noexcept {
  try {
    out_board.fill(0);
    chess::Board cb = chess::initial_board(fen);
    for (std::size_t sq = 0; sq < 64; ++sq) {
      out_board[sq] = encode_piece(cb.pieces[sq]);
    }
    return true;
  } catch (...) {
    return false;
  }
}

// Inline-friendly evaluator using material + midgame PST only (white
// perspective)
EvalResult evaluate(const Board& board) noexcept {
  int32_t score = 0;
  for (std::size_t sq = 0; sq < 64; ++sq) {
    int8_t p = board[sq];
    if (p == 0)
      continue;
    int idx = p > 0 ? p : -p;
    if (idx < 1 || idx > 6) {
      continue;
    }
    bool white = p > 0;
    int mat = kSunfishMaterial[static_cast<std::size_t>(idx)];
    score += white ? mat : -mat;
    int type_index = idx - 1;
    const int oriented_sq =
        white ? static_cast<int>(sq) : chess::mirror_rank(static_cast<int>(sq));
    int mg_entry =
        chess::kMidgamePstSunfish[static_cast<std::size_t>(type_index)]
                                 [static_cast<std::size_t>(oriented_sq)];
    score += white ? mg_entry : -mg_entry;
  }
  return EvalResult{score};
}

} // namespace sf_eval
