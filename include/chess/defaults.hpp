#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace chess {
inline constexpr int INF = 1000000;
inline constexpr int MATE_SCORE = 90000;
inline constexpr int MATE_BOUND = MATE_SCORE + 1024;
inline constexpr int ASPIRATION_WINDOW_INITIAL = 800;
inline constexpr int ASPIRATION_WINDOW_MAX = 6400;
inline constexpr int MOBILITY_SCALING = 15;
inline constexpr int REPETITION_PENALTY = 250;
inline constexpr uint8_t MAX_BB_UPDATE_PIECES = 3; // max 3 pieces change per move
                                                   // (castle+capture || capture promotion)
inline constexpr std::array<int, 8> kRanks{0, 1, 2, 3, 4, 5, 6, 7};
inline constexpr std::string_view kFiles = "abcdefgh";
inline constexpr std::uint8_t kBoardSize = 64;
inline constexpr std::uint16_t kMaxMovementCount = 256;
inline constexpr std::uint16_t MAX_PLY = 1024;
inline constexpr std::string_view kStartFEN{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
// "rnbqkbnr/ppp1ppp1/7p/3p4/1P2P4/8/2P1PPPP/RNBQKBNR w KQkq - 0 1"};
} // namespace chess