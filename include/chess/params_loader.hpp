#pragma once

#include "chess/engine_params.hpp"

#include <string>

namespace chess {

// Loads YAML engine parameters from file. On success, updates params and
// returns true. On failure, returns false and populates error.
bool load_engine_params_from_file(const std::string& path, EngineParams& params,
                                  std::string& error);

} // namespace chess
