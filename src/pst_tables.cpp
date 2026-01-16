#include "chess/pst_tables.hpp"

namespace chess {
namespace {
std::array<Pst, 6> g_midgame_pst = kMidgamePst;
std::array<Pst, 6> g_endgame_pst = kEndgamePst;
} // namespace

const std::array<Pst, 6>& midgame_pst() {
  return g_midgame_pst;
}

const std::array<Pst, 6>& endgame_pst() {
  return g_endgame_pst;
}

void set_pst_tables(const std::array<Pst, 6>& midgame,
                    const std::array<Pst, 6>& endgame) {
  g_midgame_pst = midgame;
  g_endgame_pst = endgame;
}

void reset_pst_tables() {
  g_midgame_pst = kMidgamePst;
  g_endgame_pst = kEndgamePst;
}

} // namespace chess
