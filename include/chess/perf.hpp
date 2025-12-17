#pragma once
#include "chess/cli.hpp"
#include "chess/engine.hpp"

namespace chess {
int run_perf_mode(Engine& engine, const chess::CliOptions& options);

}