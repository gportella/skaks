#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace chess {
inline constexpr std::array<int, 8> kRanks{0, 1, 2, 3, 4, 5, 6, 7};
inline constexpr std::string_view kFiles = "abcdefgh";
inline constexpr std::uint8_t kBoardSize = 64;
inline constexpr std::uint16_t kMaxMovementCount = 256;
inline constexpr std::string_view kStartFEN{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
} // namespace chess