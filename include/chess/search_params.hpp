#pragma once

#include "chess/defaults.hpp"

#include <array>

namespace chess {

struct SearchParams {
  int aspiration_window_initial = ASPIRATION_WINDOW_INITIAL;
  int aspiration_window_max = ASPIRATION_WINDOW_MAX;
  int quiescence_delta_margin = QUIESCENCE_DELTA_MARGIN;
  int quiescence_max_ply = QUIESCENCE_MAX_PLY;
  int quiescence_max_noisy_moves = QUIESCENCE_MAX_NOISY_MOVES;
  int quiescence_zero_gain_skip_index = QUIESCENCE_ZERO_GAIN_SKIP_INDEX;
  int quiescence_max_quiet_checks = QUIESCENCE_MAX_QUIET_CHECKS;
  int null_move_reduction = NULL_MOVE_REDUCTION;
  int null_move_reduction_divisor = NULL_MOVE_REDUCTION_DIVISOR;
  int null_move_min_depth = NULL_MOVE_MIN_DEPTH;
  double lmr_intercept = LMR_INTERCEPT;
  double lmr_divisor = LMR_DIVISOR;
  double lmr_history_divisor = LMR_HISTORY_DIVISOR;
  double lmr_pv_offset = LMR_PV_OFFSET;
  std::array<int, 4> futility_margins = FUTILITY_MARGINS;
  int razor_margin_bonus = RAZOR_MARGIN_BONUS;
  int probcut_margin = PROBCUT_MARGIN;
  int probcut_reduction = PROBCUT_REDUCTION;
  int probcut_max_captures = PROBCUT_MAX_CAPTURES;
  int see_capture_threshold = SEE_CAPTURE_THRESHOLD;
  int see_capture_max_value = SEE_CAPTURE_MAX_VALUE;
};

inline SearchParams default_search_params() {
  return SearchParams{};
}

inline SearchParams& mutable_search_params() {
  static SearchParams params = default_search_params();
  return params;
}

inline const SearchParams& search_params() {
  return mutable_search_params();
}

inline void set_search_params(const SearchParams& params) {
  mutable_search_params() = params;
}

inline void reset_search_params() {
  mutable_search_params() = default_search_params();
}

} // namespace chess
