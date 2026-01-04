#pragma once

#include "sf_nnue/nnue.h"

#include <array>
#include <cstddef>

namespace chess {

class SfNnueStack {
public:
  SfNnueStack();

  NNUEdata** pointer_triplet();

private:
  std::array<NNUEdata, 3> buffers_{};
  std::array<NNUEdata*, 3> ptrs_{};
};

SfNnueStack* current_thread_nnue_stack();

} // namespace chess
