// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "chess/board.hpp"
#include "chess/moves.hpp"

#include <memory>

namespace chess {

class NnueAdapter {
public:
  NnueAdapter();
  explicit NnueAdapter(const Board& board);
  ~NnueAdapter();

  NnueAdapter(const NnueAdapter&) = delete;
  NnueAdapter& operator=(const NnueAdapter&) = delete;
  NnueAdapter(NnueAdapter&&) noexcept;
  NnueAdapter& operator=(NnueAdapter&&) noexcept;

  void reset(const Board& board);
  void push_move(const Move& move);
  void push_null();
  void pop_move();
  void pop_null();

private:
  friend int evaluate_nnue_stockfish_incremental(const NnueAdapter& adapter,
                                                 bool adjusted, int* complexity);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

int evaluate_nnue_stockfish(const Board& board);
int evaluate_nnue_stockfish_incremental(const NnueAdapter& adapter,
                                        bool adjusted = true,
                                        int* complexity = nullptr);

} // namespace chess