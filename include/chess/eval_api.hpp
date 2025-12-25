#pragma once

#include "chess/board.hpp"
#include "chess/engine.hpp"
#include "chess/engine_params.hpp"

#include <string>

namespace chess {

struct EvalResponse {
  int centipawns = 0;
  bool ok = false;
  std::string error;
};

inline EvalResponse evaluate_fen_with_params(const std::string& fen,
                                             const EngineParams& params) {
  EvalResponse resp{};
  try {
    Board b = initial_board(fen);
    set_engine_params(params);
    Engine engine;
    resp.centipawns = engine.evaluate(b);
    resp.ok = true;
    return resp;
  } catch (const std::exception& ex) {
    resp.ok = false;
    resp.error = ex.what();
    return resp;
  }
}

inline EvalResponse evaluate_fen_default_params(const std::string& fen) {
  return evaluate_fen_with_params(fen, default_engine_params());
}

} // namespace chess
