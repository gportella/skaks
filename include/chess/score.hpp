#pragma once

#include "chess/defaults.hpp"

#include <cstdlib>

namespace chess {

inline constexpr bool is_mate_score(int score) {
  return score >= MATE_THRESHOLD || score <= -MATE_THRESHOLD;
}

inline int normalize_mate_score(int score, int ply) {
  if (score >= MATE_THRESHOLD) {
    return MATE_VALUE - ply;
  }
  if (score <= -MATE_THRESHOLD) {
    return -MATE_VALUE + ply;
  }
  return score;
}

inline int encode_mate_score(int score, int ply) {
  if (score >= MATE_THRESHOLD) {
    return score + ply;
  }
  if (score <= -MATE_THRESHOLD) {
    return score - ply;
  }
  return score;
}

inline int decode_mate_score(int stored_score, int ply) {
  if (stored_score >= MATE_THRESHOLD) {
    return stored_score - ply;
  }
  if (stored_score <= -MATE_THRESHOLD) {
    return stored_score + ply;
  }
  return stored_score;
}

inline int mate_moves_from_score(int score) {
  const int mate_ply = MATE_VALUE - std::abs(score);
  return (mate_ply + 1) / 2;
}

} // namespace chess
