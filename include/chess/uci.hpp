#pragma once

#include "chess/engine.hpp"
#include "chess/polyglot.hpp"

#include <filesystem>
#include <optional>

namespace chess {

struct UciPolyglotContext {
  const polyglot::Book* book = nullptr;
  std::filesystem::path book_path;
  bool use_weighted = true;
};

void run_uci_loop(
    Engine& engine, int default_depth,
    const std::optional<UciPolyglotContext>& polyglot_ctx = std::nullopt);

} // namespace chess
