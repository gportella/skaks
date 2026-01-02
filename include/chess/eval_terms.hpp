#pragma once

#include <cstddef>

namespace chess {

enum class TermId : int {
  Material = 0,
  PawnCenter,
  CenterControl,
  Attacking,
  KingSafety,
  KingMobility,
  Pins,
  PstMg,
  PstEg,
  PassedPawns,
  Initiative,
  Hanging,
  KingRing,
  BishopPair,
  RookFiles,
  MinorMobility,
  PawnStructure,
  Count
};

} // namespace chess
