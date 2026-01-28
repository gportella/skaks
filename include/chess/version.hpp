#pragma once

#include <array>
#include <string_view>

namespace chess {

inline constexpr std::string_view kEngineName = "skaks";
inline constexpr std::string_view kEngineVersion = "0.17.5";

inline constexpr std::array<std::string_view, 19> kOptimizationFeatures{
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
    "Null move pruning, 2-ply historical heuristic and SEE sorting",
    "Time management for search limits",
    "Added MVV-LVA and SEE for capture move ordering",
    "Threaded UCI search support, with pondering, go time, nodes and infinite",
    "Parameter loading from external file",
    "SIMD optimizations for bitboard operations and threads",
    "Incremental eval NNUE from Stockfish",
    "Extensive CLI modes for self-play, benchmarking, and profiling",
    "Futility, reverse futility and SEE pruning in search"};
} // namespace chess

// Detect whether NEON headers are available at compile time (for aarch64).
#if defined(__aarch64__)
#if defined(__has_include)
#if __has_include(<arm_neon.h>)
namespace chess {
inline constexpr bool kCompiledWithNeon = true;
} // namespace chess
#else
namespace chess {
inline constexpr bool kCompiledWithNeon = false;
} // namespace chess
#endif
#else
namespace chess {
inline constexpr bool kCompiledWithNeon = false;
} // namespace chess
#endif
#else
namespace chess {
inline constexpr bool kCompiledWithNeon = false;
} // namespace chess
#endif
