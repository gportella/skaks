#pragma once

#include "chess/types.hpp"

#include <cstdint>

namespace chess {

/**
 * @brief Initialize rook/bishop magic bitboard tables.
 *
 * Safe to call multiple times; initialization is performed once.
 */
void init_magic_bitboards();

/**
 * @brief Returns true if magic tables are initialized.
 */
bool magic_bitboards_ready();

/**
 * @brief Get rook attacks using magic bitboards.
 */
Bitboard rook_attacks_magic(int sq, Bitboard occ);

/**
 * @brief Get bishop attacks using magic bitboards.
 */
Bitboard bishop_attacks_magic(int sq, Bitboard occ);

} // namespace chess
