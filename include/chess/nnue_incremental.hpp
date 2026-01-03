#pragma once

#include "chess/defaults.hpp"
#include "nnue.h"

#include <array>
#include <cstddef>

namespace chess {

struct Board;
struct Move;
struct Undo;

class SfNnueStack {
public:
  SfNnueStack();

  void reset();
  void push_move(const Board& board, const Move& move, const Undo& undo);
  void push_null();
  void pop();

  [[nodiscard]] int depth() const;
  [[nodiscard]] NNUEdata& current_data();
  [[nodiscard]] const NNUEdata& current_data() const;
  [[nodiscard]] std::array<NNUEdata*, 3> pointer_triplet();

private:
  static constexpr std::size_t kMaxDepth = static_cast<std::size_t>(MAX_PLY) + 8;
  std::array<NNUEdata, kMaxDepth> stack_{};
  std::size_t top_ = 0;
  std::array<bool, kMaxDepth> force_refresh_{};
};

class ScopedNnueThreadContext {
public:
  explicit ScopedNnueThreadContext(SfNnueStack* stack);
  ~ScopedNnueThreadContext();

  ScopedNnueThreadContext(const ScopedNnueThreadContext&) = delete;
  ScopedNnueThreadContext& operator=(const ScopedNnueThreadContext&) = delete;

private:
  SfNnueStack* previous_ = nullptr;
};

SfNnueStack* current_thread_nnue_stack();
void set_thread_nnue_stack(SfNnueStack* stack);

} // namespace chess
