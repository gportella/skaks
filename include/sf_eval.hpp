// Extremely lightweight Sunfish-style evaluation implementation.
// C++23, header-only declarations for fast inlining.
#pragma once

#include "chess/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sf_eval {

struct EvalResult {
  int32_t score; // centipawns
};

// Minimal board representation: 64 squares, 0=empty, piece codes small ints
using Board = std::array<int8_t, 64>;

// Parse a simple FEN into Board. Returns true on success.
bool parse_fen(const std::string& fen, Board& out_board) noexcept;

// Evaluate a board position in centipawns using a Sunfish-like linear eval.
// This function is intentionally simple and fast: it uses piece-square tables
// and material weights compatible with classic Sunfish behavior.
EvalResult evaluate(const Board& board) noexcept;

// Encode an engine occupancy value as a Sunfish-style signed piece code.
int8_t encode_piece(chess::OccupancyType occ) noexcept;

} // namespace sf_eval
