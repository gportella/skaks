#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace chess {

inline constexpr std::uint16_t MAX_PLY = 1024;
inline constexpr int REPETITION_PENALTY = 250;
inline constexpr int INF = 1000000;
inline constexpr int MATE_SCORE = 90000;
inline constexpr int CHECK_EXTENSION = 1;
inline constexpr int MATE_BOUND = MATE_SCORE + 1024;
inline constexpr int MATE_VALUE = INF;
inline constexpr int MATE_THRESHOLD = MATE_VALUE - static_cast<int>(MAX_PLY);
inline constexpr int MOBILITY_SCALING = 15;

inline constexpr int ASPIRATION_WINDOW_INITIAL = 100;
inline constexpr int ASPIRATION_WINDOW_MAX = 800;
inline constexpr int QUIESCENCE_DELTA_MARGIN = 250;
inline constexpr int QUIESCENCE_MAX_PLY = 8;
inline constexpr int QUIESCENCE_MAX_NOISY_MOVES = 10;
inline constexpr int QUIESCENCE_ZERO_GAIN_SKIP_INDEX = 1;
inline constexpr int QUIESCENCE_MAX_QUIET_CHECKS = 0;
inline constexpr int NULL_MOVE_REDUCTION = 3;
inline constexpr int NULL_MOVE_REDUCTION_DIVISOR = 8;
inline constexpr int NULL_MOVE_MIN_DEPTH = 4;
inline constexpr double LMR_INTERCEPT = 0.0;
inline constexpr double LMR_DIVISOR = 1.5;
inline constexpr double LMR_HISTORY_DIVISOR = 8000.0;
inline constexpr double LMR_PV_OFFSET = 1.0;
inline constexpr std::array<int, 4> FUTILITY_MARGINS{0, 120, 240, 400};
inline constexpr int RAZOR_MARGIN_BONUS = 80;
inline constexpr int PROBCUT_MARGIN = 250;
inline constexpr int PROBCUT_REDUCTION = 4;
inline constexpr int PROBCUT_MAX_CAPTURES = 3;
inline constexpr int SEE_CAPTURE_THRESHOLD = 10;
inline constexpr int SEE_CAPTURE_MAX_VALUE = 330;
inline constexpr std::size_t DEFAULT_HASH_MB = 512;

inline constexpr uint8_t MAX_BB_UPDATE_PIECES =
    3; // max 3 pieces change per move
       // (castle+capture || capture promotion)
inline constexpr std::array<int, 8> kRanks{0, 1, 2, 3, 4, 5, 6, 7};
inline constexpr std::string_view kFiles = "abcdefgh";
inline constexpr std::uint8_t kBoardSize = 64;
inline constexpr std::uint16_t kMaxMovementCount = 256;
inline constexpr std::string_view kStartFEN{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
} // namespace chess
// namespace chess