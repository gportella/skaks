#pragma once

#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"

#include <array>
#include <optional>

namespace chess {

struct SEECacheEntry {
  uint32_t code{0};
  int value{0};
};

struct SEECache {
  std::uint64_t key{0};
  std::size_t size{0};
  std::size_t next{0};
  std::array<SEECacheEntry, 16> entries{};
};

void reset_see_cache(SEECache& cache, std::uint64_t position_key);
bool see_cache_lookup(const SEECache& cache, uint32_t code, int& value_out);
void see_cache_store(SEECache& cache, uint32_t code, int value);

int static_exchange_eval(Board b, const Move& move);
int static_exchange_eval_cached(const Board& board, const Move& move,
                                SEECache* cache);

} // namespace chess