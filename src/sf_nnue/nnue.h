#pragma once

#include <cstddef>
#include <cstdint>

struct alignas(64) NNUEdata {
  std::uint8_t storage[64];
};

#ifdef __cplusplus
extern "C" {
#endif

void nnue_init(const char* evalFile);
int nnue_evaluate_fen(const char* fen);
int nnue_evaluate(int player, int* pieces, int* squares);
int nnue_evaluate_incremental(int player, int* pieces, int* squares,
                              NNUEdata** nnue_data);

#ifdef __cplusplus
}
#endif
