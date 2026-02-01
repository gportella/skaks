#pragma once

#include "chess/types.hpp"

namespace chess::search_detail {

bool should_retry_pvs(SideToMove stm, int child_score, int narrow_alpha,
                      int narrow_beta);

inline int singular_beta(int tt_score, int depth, int margin_per_depth) {
  return tt_score - depth * margin_per_depth;
}

} // namespace chess::search_detail
