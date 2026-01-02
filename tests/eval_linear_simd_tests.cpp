#include "chess/evaluation_params.hpp"
#include "chess/scoring_rules.hpp"

#include <iostream>
#include <random>

using namespace chess;

int main() {
  // Seed deterministic RNG for test repeatability
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> intval(-200, 200);
  std::uniform_int_distribution<int> phase_dist(0, kPstPhaseMax);

  PhaseWeights W;
  for (std::size_t i = 0; i < static_cast<std::size_t>(TermId::Count); ++i) {
    W.mg[i] = 1.0f + static_cast<float>((i % 5) - 2) * 0.1f;
    W.eg[i] = 1.0f + static_cast<float>(((i + 3) % 7) - 3) * 0.05f;
  }

  for (int t = 0; t < 200; ++t) {
    EvalVector v{};
    for (int i = 0; i < static_cast<int>(TermId::Count); ++i)
      v.f[i] = intval(rng);
    v.mg_phase = phase_dist(rng);
    v.eg_phase = kPstPhaseMax - v.mg_phase;

    // Compute contributions via helper and sum
    float contributions[kTermCount];
    compute_term_contributions(v, W, contributions);
    float sum = 0.0f;
    for (std::size_t i = 0; i < kTermCount; ++i)
      sum += contributions[i];

    // Compute via eval_linear
    int lin = eval_linear(v, W);

    int approx = static_cast<int>(std::lround(sum));
    if (lin != approx) {
      std::cerr << "Mismatch: eval_linear=" << lin
                << " vs contrib_sum=" << approx << "\n";
      return 2;
    }
  }

  std::cout << "eval_linear SIMD parity test passed\n";
  return 0;
}
