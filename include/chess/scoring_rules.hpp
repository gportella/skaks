#pragma once

#include "chess/board.hpp"
#include "chess/eval_terms.hpp"
#include "chess/evaluation_params.hpp"

#include <array>

namespace chess {

struct PhaseWeights {
  std::array<float, static_cast<std::size_t>(TermId::Count)> mg{};
  std::array<float, static_cast<std::size_t>(TermId::Count)> eg{};
};

struct EvalVector {
  std::array<int, static_cast<int>(TermId::Count)> f{};
  int mg_phase = 0; // [0..kPstPhaseMax]
  int eg_phase = 0; // [0..kPstPhaseMax]
};

// Feature collection helpers
EvalVector compute_eval_vector(const Board& b);
int eval_linear(const EvalVector& v, const PhaseWeights& w);

// Weight accessors (defaults to 1.0 for all terms)
PhaseWeights& mutable_phase_weights();
const PhaseWeights& phase_weights();
void set_phase_weights(const PhaseWeights& w);
void reset_phase_weights();

int evaluate_board(const Board& board);
int evaluate_attacking_pieces(const Board& board);

} // namespace chess