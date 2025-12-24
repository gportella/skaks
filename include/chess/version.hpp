#pragma once

#include <array>
#include <string_view>

namespace chess {

inline constexpr std::string_view kEngineName = "skaks";
inline constexpr std::string_view kEngineVersion = "0.9";

inline constexpr std::array<std::string_view, 13> kOptimizationFeatures{
    "Bitboard move generation with precomputed attack masks",
    "Alpha-beta search with transposition table caching",
    "Zobrist hashing for incremental board state keys",
    "Repetition detection through historical position tracking",
    "Move ordering seeded by cached transposition moves",
    "PV search with some reductions",
    "Root move exclusion support for iterative deepening",
    "Quiescence search to reduce horizon effect",
    "Killer move heuristic for quiet move ordering",
    "Support for polyglot book of moves",
    "Null move pruning, historical heuristic and SEE sorting",
    "Time management for search limits",
    "Incremental evaluation with piece-square tables"};

} // namespace chess
