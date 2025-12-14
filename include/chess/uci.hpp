#pragma once

#include "chess/engine.hpp"

namespace chess {

void run_uci_loop(Engine& engine, int default_depth);

} // namespace chess
