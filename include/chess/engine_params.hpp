#pragma once

#include "chess/evaluation_params.hpp"
#include "chess/search_params.hpp"

namespace chess {

struct EngineParams {
  EvaluationParams evaluation;
  SearchParams search;
};

inline EngineParams default_engine_params() {
  EngineParams params{};
  params.evaluation = default_evaluation_params();
  params.search = default_search_params();
  return params;
}

inline void set_engine_params(const EngineParams& params) {
  set_evaluation_params(params.evaluation);
  set_search_params(params.search);
}

inline void reset_engine_params() {
  reset_evaluation_params();
  reset_search_params();
}

} // namespace chess
