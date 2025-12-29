#pragma once

#include "chess/evaluation_params.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/search_params.hpp"

namespace chess {

struct EngineParams {
  EvaluationParams evaluation;
  SearchParams search;
  PhaseWeights phase_weights;
};

inline EngineParams default_engine_params() {
  EngineParams params{};
  params.evaluation = default_evaluation_params();
  params.search = default_search_params();
  for (std::size_t i = 0; i < static_cast<std::size_t>(TermId::Count); ++i) {
    params.phase_weights.mg[i] = 1.0;
    params.phase_weights.eg[i] = 1.0;
  }
  return params;
}

inline void set_engine_params(const EngineParams& params) {
  set_evaluation_params(params.evaluation);
  set_search_params(params.search);
  set_phase_weights(params.phase_weights);
}

inline void reset_engine_params() {
  reset_evaluation_params();
  reset_search_params();
  reset_phase_weights();
}

} // namespace chess
