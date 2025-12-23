
#include "chess/exchange.hpp"

#include "chess/attack_masks.hpp"
#include "chess/piece_values.hpp"

#include <algorithm>

namespace {
constexpr std::size_t kSEECacheCapacity = 128;
}

namespace chess {
void reset_see_cache(SEECache& cache, std::uint64_t position_key) {
  if (cache.key != position_key) {
    cache.key = position_key;
    cache.size = 0;
    cache.next = 0;
  }
}

bool see_cache_lookup(const SEECache& cache, uint32_t code, int& value_out) {
  if (cache.size == 0) {
    return false;
  }
  const std::size_t capacity =
      std::min<std::size_t>(cache.entries.size(), kSEECacheCapacity);
  const std::size_t limit = std::min(cache.size, capacity);
  for (std::size_t idx = 0; idx < limit; ++idx) {
    if (cache.entries[idx].code == code) {
      value_out = cache.entries[idx].value;
      return true;
    }
  }
  return false;
}

void see_cache_store(SEECache& cache, uint32_t code, int value) {
  const std::size_t capacity =
      std::min<std::size_t>(cache.entries.size(), kSEECacheCapacity);
  if (capacity == 0) {
    return;
  }
  const std::size_t idx = cache.next % capacity;
  cache.entries[idx].code = code;
  cache.entries[idx].value = value;
  if (cache.size < capacity) {
    ++cache.size;
  }
  cache.next = (idx + 1) % capacity;
}

inline int SEE_sq(Board& b, int sq) {
  int value = 0;
  auto attacker =
      find_smallest_attacker(b, static_cast<u_int8_t>(sq), b.side_to_move);
  if (attacker.has_value()) {
    UndoSEE undo = make_see_move(
        b,
        Move{static_cast<uint16_t>(attacker->square), static_cast<uint16_t>(sq),
             b.pieces[static_cast<std::size_t>(attacker->square)],
             b.pieces[static_cast<std::size_t>(sq)], OccupancyType::empty, 0});
    b.side_to_move = flip_side(b.side_to_move);
    value =
        std::max(0, piece_material_magnitude(undo.captured_pc) - SEE_sq(b, sq));
    undo_see_move(b, undo);
    b.side_to_move = flip_side(b.side_to_move);
  }
  return value;
}

int static_exchange_eval(Board b, const Move& move) {
  // we are going to skip unless we capture or promote
  if (move.captured_pc == OccupancyType::empty ||
      move.promo_pc == OccupancyType::empty) {
    auto target_sq = move.to;
    return SEE_sq(b, target_sq);
  }
  return 0;
}

int static_exchange_eval_cached(const Board& board, const Move& move,
                                SEECache* cache) {
  if (cache == nullptr) {
    return static_exchange_eval(board, move);
  }

  reset_see_cache(*cache, board.position_key);

  const uint32_t code = encode_move(move.from, move.to, move.moving_pc,
                                    move.captured_pc, move.promo_pc,
                                    move.flags);
  int cached_value = 0;
  if (see_cache_lookup(*cache, code, cached_value)) {
    return cached_value;
  }

  const int value = static_exchange_eval(board, move);
  see_cache_store(*cache, code, value);
  return value;
}
} // namespace chess