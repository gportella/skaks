#pragma once

#include "chess/types.hpp"

#include <cstdint>
#include <string_view>

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

enum class MagicBitboardsMode { Auto, Slow };

MagicBitboardsMode magic_bitboards_mode();
void set_magic_bitboards_mode(MagicBitboardsMode mode);

/**
 * @brief Get rook attacks using magic bitboards.
 */
Bitboard rook_attacks_magic(int sq, Bitboard occ);

/**
 * @brief Get bishop attacks using magic bitboards.
 */
Bitboard bishop_attacks_magic(int sq, Bitboard occ);

/**
 * @brief Returns the current magic bitboard source label.
 * Values: "disabled", "cache", "embedded", "generated".
 */
std::string_view magic_source();

} // namespace chess
