#pragma once

#include "chess/types.hpp"

namespace chess {

inline bool has_rights(chess::CastlingRights cr, chess::CastlingRights flag) {
  return (static_cast<int>(cr) & static_cast<int>(flag)) != 0;
}

} // namespace chess