#pragma once

#include "chess/types.hpp"

namespace chess::search_detail {

bool should_retry_pvs(SideToMove stm, int child_score, int narrow_alpha,
                      int narrow_beta);

} // namespace chess::search_detail
