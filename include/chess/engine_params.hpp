#pragma once

#include "chess/search_params.hpp"

#include <cstddef>

namespace chess {

struct EngineParams {
  SearchParams search;
  SearchParams search_nnue;
  bool use_search_nnue = false;
};

inline EngineParams default_engine_params() {
  EngineParams params{};
  params.search = default_search_params();
  params.search_nnue = params.search;
  params.use_search_nnue = true;
  return params;
}

inline void set_engine_params(const EngineParams& params) {
  const bool use_nnue_search = params.use_search_nnue;
  set_search_params(use_nnue_search ? params.search_nnue : params.search);
}

inline void reset_engine_params() {
  reset_search_params();
}

} // namespace chess
