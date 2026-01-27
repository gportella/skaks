#pragma once

#include "chess/eval_mode.hpp"
#include "chess/evaluation_params.hpp"
#include "chess/scoring_rules.hpp"
#include "chess/search_params.hpp"

#include <cstddef>

namespace chess {

struct EngineParams {
  EvaluationParams evaluation;
  SearchParams search;
  SearchParams search_nnue;
  bool use_search_nnue = false;
  PhaseWeights phase_weights;
};

inline EngineParams default_engine_params() {
  EngineParams params{};
  params.evaluation = default_evaluation_params();
  params.search = default_search_params();
  params.search_nnue = params.search;
  params.use_search_nnue = true;
  for (std::size_t i = 0; i < static_cast<std::size_t>(TermId::Count); ++i) {
    params.phase_weights.mg[i] = 1.0F;
    params.phase_weights.eg[i] = 1.0F;
  }
  return params;
}

inline void set_engine_params(const EngineParams& params) {
  set_evaluation_params(params.evaluation);
  set_search_params(params.search);
  set_phase_weights(params.phase_weights);
}

inline void set_engine_params_for_mode(const EngineParams& params,
                                       EvaluationMode mode) {
  set_evaluation_params(params.evaluation);
  const bool use_nnue_search =
      (mode == EvaluationMode::Stockfish) && params.use_search_nnue;
  set_search_params(use_nnue_search ? params.search_nnue : params.search);
  set_phase_weights(params.phase_weights);
}

inline void reset_engine_params() {
  reset_evaluation_params();
  reset_search_params();
  reset_phase_weights();
}

} // namespace chess
