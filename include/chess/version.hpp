#pragma once

#include <array>
#include <string_view>

namespace chess {

inline constexpr std::string_view kEngineName = "skaks";
inline constexpr std::string_view kEngineVersion = "0.1";

inline constexpr std::array<std::string_view, 5> kOptimizationFeatures{
    "Bitboard move generation with precomputed attack masks",
    "Alpha-beta search with transposition table caching",
    "Zobrist hashing for incremental board state keys",
    "Repetition detection through historical position tracking",
    "Move ordering seeded by cached transposition moves"};

} // namespace chess
