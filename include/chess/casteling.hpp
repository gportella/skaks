#pragma once

#include "chess/types.hpp"

#include <array>

namespace chess {

inline bool has_rights(chess::CastlingRights cr, chess::CastlingRights flag) {
  return (static_cast<int>(cr) & static_cast<int>(flag)) != 0;
}

struct CastlingSideConfig {
  Square king_start;
  Square king_kingside_target;
  Square king_queenside_target;
  Square rook_kingside_start;
  Square rook_kingside_target;
  Square rook_queenside_start;
  Square rook_queenside_target;
  std::array<Square, 2> king_path;
  std::array<Square, 3> queen_path;
  std::array<Square, 2> king_safe;
  std::array<Square, 2> queen_safe;
  CastlingRights king_flag;
  CastlingRights queen_flag;
};
// clang-format off
inline constexpr std::array<CastlingSideConfig, 2> kCastlingSideConfigs = {
    CastlingSideConfig{Square::E1, Square::G1, Square::C1, 
                       Square::H1, Square::F1, Square::A1, Square::D1,
                       {Square::F1, Square::G1},
                       {Square::D1, Square::C1, Square::B1},
                       {Square::F1, Square::G1},
                       {Square::D1, Square::C1},
                       WK,
                       WQ},
    CastlingSideConfig{Square::E8, Square::G8, Square::C8,
                       Square::H8, Square::F8, Square::A8, Square::D8,
                       {Square::F8, Square::G8},
                       {Square::D8, Square::C8, Square::B8},
                       {Square::F8, Square::G8},
                       {Square::D8, Square::C8},
                       BK,
                       BQ}};
// clang-format on

} // namespace chess