#include "chess/attack_masks.hpp"
#include "chess/board.hpp"
#include "chess/types.hpp"

#pragma once
namespace chess {
int test_masks();
void dump_attacks(Board& b, SideToMove stm);
}