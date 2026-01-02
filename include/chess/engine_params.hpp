#pragma once

#include "chess/evaluation_params.hpp"
<<<<<<< HEAD
=======
#include "chess/scoring_rules.hpp"
>>>>>>> nnue_version
#include "chess/search_params.hpp"

namespace chess {

struct EngineParams {
  EvaluationParams evaluation;
  SearchParams search;
<<<<<<< HEAD
=======
  PhaseWeights phase_weights;
>>>>>>> nnue_version
};

inline EngineParams default_engine_params() {
  EngineParams params{};
  params.evaluation = default_evaluation_params();
  params.search = default_search_params();
<<<<<<< HEAD
=======
  for (std::size_t i = 0; i < static_cast<std::size_t>(TermId::Count); ++i) {
    params.phase_weights.mg[i] = 1.0;
    params.phase_weights.eg[i] = 1.0;
  }
>>>>>>> nnue_version
  return params;
}

inline void set_engine_params(const EngineParams& params) {
  set_evaluation_params(params.evaluation);
  set_search_params(params.search);
<<<<<<< HEAD
=======
  set_phase_weights(params.phase_weights);
>>>>>>> nnue_version
}

inline void reset_engine_params() {
  reset_evaluation_params();
  reset_search_params();
<<<<<<< HEAD
=======
  reset_phase_weights();
>>>>>>> nnue_version
}

} // namespace chess
