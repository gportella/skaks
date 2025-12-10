#pragma once

#include "chess/attack_masks.hpp"
#include "chess/defaults.hpp"
#include "chess/types_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string_view>
#include <utility>

namespace chess::debug {

template <typename Emitter>
void dump_moves(std::string_view label, Emitter&& emitter, std::ostream& os = std::cout,
                std::uint16_t max_to_show = 10) {
  std::array<std::uint32_t, kMaxMovementCount> moves{};
  std::uint16_t count = 0;
  std::forward<Emitter>(emitter)(moves, count);

  os << label << " (" << count << ")\n";

  const std::uint16_t limit = std::min<std::uint16_t>(count, max_to_show);
  for (std::uint16_t i = 0; i < limit; ++i) {
    const auto move = moves[i];
    const auto from = move_from(move);
    const auto to = move_to(move);
    const auto captured = move_captured(move);
    const auto promo = move_promo(move);

    os << i << ": " << square_to_string(from) << " -> " << square_to_string(to);

    if (captured != static_cast<int>(OccupancyType::empty)) {
      os << " x" << to_string(static_cast<OccupancyType>(captured));
    }

    if (promo != static_cast<int>(OccupancyType::empty)) {
      os << " =" << to_string(static_cast<OccupancyType>(promo));
    }

    os << '\n';
  }

  if (count > max_to_show) {
    os << "..." << '\n';
  }

  os << '\n';
}

} // namespace chess::debug
