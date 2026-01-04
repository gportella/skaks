#include "chess/nnue_incremental.hpp"

#include <cstddef>

namespace chess {

SfNnueStack::SfNnueStack() {
  for (std::size_t i = 0; i < ptrs_.size(); ++i) {
    ptrs_[i] = &buffers_[i];
  }
}

NNUEdata** SfNnueStack::pointer_triplet() {
  return ptrs_.data();
}

SfNnueStack* current_thread_nnue_stack() {
  thread_local SfNnueStack stack;
  return &stack;
}

} // namespace chess
