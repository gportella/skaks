#include "chess/magic_bitboards.hpp"

#include "chess/board_arithmetic.hpp"
#include "chess/ray_tables.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <random>
#include <vector>

namespace chess {

namespace {

constexpr std::array<int, 64> kRookMagicShift = {
    12, 11, 11, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 12, 11, 11, 11, 11, 11, 11, 12};

constexpr std::array<int, 64> kBishopMagicShift = {
    6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7,
    5, 5, 5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 7,
    7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6};

struct MagicEntry {
  Bitboard mask = 0;
  Bitboard magic = 0;
  int shift = 0;
  const Bitboard* attacks = nullptr;
  const int* bits = nullptr;
  int bit_count = 0;
};

std::array<MagicEntry, 64> g_rook_magics{};
std::array<MagicEntry, 64> g_bishop_magics{};
std::vector<Bitboard> g_rook_attacks{};
std::vector<Bitboard> g_bishop_attacks{};
std::array<std::array<int, 14>, 64> g_rook_bits{};
std::array<std::array<int, 14>, 64> g_bishop_bits{};
std::array<int, 64> g_rook_bit_counts{};
std::array<int, 64> g_bishop_bit_counts{};
std::once_flag g_magic_once;
bool g_magic_ready = false;

Bitboard rook_mask(int sq) {
  const int rank = sq / 8;
  const int file = sq % 8;
  Bitboard mask = 0;
  for (int r = rank + 1; r <= 6; ++r) {
    mask |= Bitboard(1) << (r * 8 + file);
  }
  for (int r = rank - 1; r >= 1; --r) {
    mask |= Bitboard(1) << (r * 8 + file);
  }
  for (int f = file + 1; f <= 6; ++f) {
    mask |= Bitboard(1) << (rank * 8 + f);
  }
  for (int f = file - 1; f >= 1; --f) {
    mask |= Bitboard(1) << (rank * 8 + f);
  }
  return mask;
}

Bitboard bishop_mask(int sq) {
  const int rank = sq / 8;
  const int file = sq % 8;
  Bitboard mask = 0;
  for (int r = rank + 1, f = file + 1; r <= 6 && f <= 6; ++r, ++f) {
    mask |= Bitboard(1) << (r * 8 + f);
  }
  for (int r = rank + 1, f = file - 1; r <= 6 && f >= 1; ++r, --f) {
    mask |= Bitboard(1) << (r * 8 + f);
  }
  for (int r = rank - 1, f = file + 1; r >= 1 && f <= 6; --r, ++f) {
    mask |= Bitboard(1) << (r * 8 + f);
  }
  for (int r = rank - 1, f = file - 1; r >= 1 && f >= 1; --r, --f) {
    mask |= Bitboard(1) << (r * 8 + f);
  }
  return mask;
}

Bitboard rook_attacks_slow(int sq, Bitboard occ) {
  const auto& rays = ROOK_RAYS[static_cast<std::size_t>(sq)];

  Bitboard north = rays.north;
  Bitboard blocker = north & occ;
  if (blocker) {
    int b = lsb_index(blocker);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    north &= mask_inclusive;
  }

  Bitboard east = rays.east;
  Bitboard east_blockers = east & occ;
  if (east_blockers) {
    int b = lsb_index(east_blockers);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    east &= mask_inclusive;
  }

  Bitboard south = rays.south;
  Bitboard south_blockers = south & occ;
  if (south_blockers) {
    int b = msb_index(south_blockers);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    south &= mask_inclusive;
  }

  Bitboard west = rays.west;
  Bitboard west_blockers = west & occ;
  if (west_blockers) {
    int b = msb_index(west_blockers);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    west &= mask_inclusive;
  }

  return north | south | east | west;
}

Bitboard bishop_attacks_slow(int sq, Bitboard occ) {
  const auto& rays = BISHOP_RAYS[static_cast<std::size_t>(sq)];

  Bitboard northeast = rays.northeast;
  Bitboard ne_blocker = northeast & occ;
  if (ne_blocker) {
    int b = lsb_index(ne_blocker);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    northeast &= mask_inclusive;
  }

  Bitboard southeast = rays.southeast;
  Bitboard se_blocker = southeast & occ;
  if (se_blocker) {
    int b = msb_index(se_blocker);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    southeast &= mask_inclusive;
  }

  Bitboard northwest = rays.northwest;
  Bitboard nw_blocker = northwest & occ;
  if (nw_blocker) {
    int b = lsb_index(nw_blocker);
    const Bitboard mask_inclusive =
        (b == 63) ? ~Bitboard(0) : ((Bitboard(1) << (b + 1)) - 1);
    northwest &= mask_inclusive;
  }

  Bitboard southwest = rays.southwest;
  Bitboard sw_blocker = southwest & occ;
  if (sw_blocker) {
    int b = msb_index(sw_blocker);
    Bitboard mask_inclusive = ~((Bitboard(1) << b) - 1);
    southwest &= mask_inclusive;
  }

  return northeast | southeast | northwest | southwest;
}

std::vector<int> mask_bits(Bitboard mask) {
  std::vector<int> bits;
  bits.reserve(static_cast<std::size_t>(popcount_bitboard(mask)));
  while (mask) {
    const int sq = lsb_index(mask);
    bits.push_back(sq);
    mask &= (mask - 1);
  }
  return bits;
}

Bitboard occupancy_from_index(std::uint64_t index,
                              const std::vector<int>& bits) {
  Bitboard occ = 0;
  for (std::size_t i = 0; i < bits.size(); ++i) {
    if (index & (Bitboard(1) << i)) {
      occ |= Bitboard(1) << bits[i];
    }
  }
  return occ;
}

[[maybe_unused]] std::size_t index_from_occupancy(Bitboard occ, const int* bits,
                                                  int bit_count) {
  std::size_t index = 0;
  for (int i = 0; i < bit_count; ++i) {
    if (occ & (Bitboard(1) << bits[i])) {
      index |= (std::size_t(1) << i);
    }
  }
  return index;
}

Bitboard random_magic(std::mt19937_64& rng) {
  std::uniform_int_distribution<std::uint64_t> dist;
  return dist(rng) & dist(rng) & dist(rng);
}

bool good_magic_candidate(Bitboard mask, Bitboard magic) {
  return popcount_bitboard((mask * magic) & 0xFF00000000000000ULL) >= 6;
}

/**
 * @brief Search for a collision-free magic number for the given occupancy map.
 */
Bitboard find_magic(Bitboard mask, int index_bits,
                    const std::vector<Bitboard>& occupancies,
                    const std::vector<Bitboard>& attacks, std::mt19937_64& rng) {
  const std::size_t table_size = Bitboard(1) << index_bits;
  std::vector<Bitboard> used(table_size);
  std::vector<bool> filled(table_size);

  for (;;) {
    const Bitboard magic = random_magic(rng);
    if (!good_magic_candidate(mask, magic)) {
      continue;
    }
    std::fill(used.begin(), used.end(), 0);
    std::fill(filled.begin(), filled.end(), false);

    bool failed = false;
    for (std::size_t i = 0; i < occupancies.size(); ++i) {
      const Bitboard occ = occupancies[i];
      const std::size_t idx =
          static_cast<std::size_t>((occ * magic) >> (64 - index_bits));
      if (!filled[idx]) {
        filled[idx] = true;
        used[idx] = attacks[i];
      } else if (used[idx] != attacks[i]) {
        failed = true;
        break;
      }
    }
    if (!failed) {
      return magic;
    }
  }
}

void init_magic_table(std::array<MagicEntry, 64>& table,
                      std::vector<Bitboard>& attacks,
                      std::array<std::array<int, 14>, 64>& bit_slots,
                      std::array<int, 64>& bit_counts,
                      const std::array<int, 64>& shift_table,
                      const std::array<Bitboard, 64>* magic_override,
                      Bitboard (*mask_fn)(int),
                      Bitboard (*attack_fn)(int, Bitboard), bool use_magic) {
  std::array<std::size_t, 64> offsets{};
  std::size_t total_size = 0;
  std::array<int, 64> local_bit_counts{};
  std::array<Bitboard, 64> masks{};

  for (int sq = 0; sq < 64; ++sq) {
    const Bitboard mask = mask_fn(sq);
    masks[static_cast<std::size_t>(sq)] = mask;
    const int bits = popcount_bitboard(mask);
    local_bit_counts[static_cast<std::size_t>(sq)] = bits;
    const int shift_bits =
        use_magic ? shift_table[static_cast<std::size_t>(sq)] : bits;
    const std::size_t table_size = Bitboard(1) << shift_bits;
    offsets[static_cast<std::size_t>(sq)] = total_size;
    total_size += table_size;
  }

  attacks.assign(total_size, 0);

  std::mt19937_64 rng(0xC0FFEE1234ULL);

  for (int sq = 0; sq < 64; ++sq) {
    const Bitboard mask = masks[static_cast<std::size_t>(sq)];
    const int relevant_bits = local_bit_counts[static_cast<std::size_t>(sq)];
    const int shift_bits =
        use_magic ? shift_table[static_cast<std::size_t>(sq)] : relevant_bits;
    const std::size_t table_size = Bitboard(1) << shift_bits;
    const std::size_t occupancy_count = Bitboard(1) << relevant_bits;
    const std::size_t offset = offsets[static_cast<std::size_t>(sq)];
    auto bits = mask_bits(mask);

    std::vector<Bitboard> occupancies(occupancy_count);
    std::vector<Bitboard> reference_attacks(occupancy_count);

    for (std::size_t idx = 0; idx < occupancy_count; ++idx) {
      const Bitboard occ = occupancy_from_index(idx, bits);
      occupancies[idx] = occ;
      reference_attacks[idx] = attack_fn(sq, occ);
    }

    Bitboard magic = 0;
    if (magic_override) {
      magic = (*magic_override)[static_cast<std::size_t>(sq)];
    } else if (use_magic) {
      magic = find_magic(mask, shift_bits, occupancies, reference_attacks, rng);
    }

    Bitboard* table_ptr = attacks.data() + offset;
    std::fill(table_ptr, table_ptr + table_size, 0);

    for (std::size_t idx = 0; idx < occupancy_count; ++idx) {
      const Bitboard occ = occupancies[idx];
      const std::size_t store_idx =
          (magic != 0)
              ? static_cast<std::size_t>((occ * magic) >> (64 - shift_bits))
              : idx;
      table_ptr[store_idx] = reference_attacks[idx];
    }

    MagicEntry entry{};
    entry.mask = mask;
    entry.magic = magic;
    entry.shift = (magic != 0) ? (64 - shift_bits) : 0;
    entry.attacks = table_ptr;
    bit_counts[static_cast<std::size_t>(sq)] = relevant_bits;
    for (int i = 0; i < relevant_bits; ++i) {
      bit_slots[static_cast<std::size_t>(sq)][static_cast<std::size_t>(i)] =
          bits[static_cast<std::size_t>(i)];
    }
    entry.bits = bit_slots[static_cast<std::size_t>(sq)].data();
    entry.bit_count = relevant_bits;
    table[static_cast<std::size_t>(sq)] = entry;
  }
}

void init_magic_tables() {
  const bool use_magic =
#if defined(SKAKS_TESTS)
      false;
#else
      (std::getenv("SKAKS_MAGIC_DISABLE") == nullptr);
#endif
  const char* cache_path = std::getenv("SKAKS_MAGIC_CACHE");
  std::array<Bitboard, 64> rook_cached{};
  std::array<Bitboard, 64> bishop_cached{};
  const std::array<Bitboard, 64>* rook_override = nullptr;
  const std::array<Bitboard, 64>* bishop_override = nullptr;

  if (cache_path && *cache_path) {
    std::ifstream in(cache_path, std::ios::binary);
    if (in.good()) {
      in.read(
          reinterpret_cast<char*>(rook_cached.data()),
          static_cast<std::streamsize>(rook_cached.size() * sizeof(Bitboard)));
      in.read(
          reinterpret_cast<char*>(bishop_cached.data()),
          static_cast<std::streamsize>(bishop_cached.size() * sizeof(Bitboard)));
      if (in.good()) {
        rook_override = &rook_cached;
        bishop_override = &bishop_cached;
      }
    }
  }

  init_magic_table(g_rook_magics, g_rook_attacks, g_rook_bits, g_rook_bit_counts,
                   kRookMagicShift, rook_override, rook_mask, rook_attacks_slow,
                   use_magic && rook_override == nullptr);
  init_magic_table(g_bishop_magics, g_bishop_attacks, g_bishop_bits,
                   g_bishop_bit_counts, kBishopMagicShift, bishop_override,
                   bishop_mask, bishop_attacks_slow,
                   use_magic && bishop_override == nullptr);

  if (cache_path && *cache_path && (!rook_override || !bishop_override)) {
    std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
    if (out.good()) {
      for (std::size_t i = 0; i < 64; ++i) {
        rook_cached[i] = g_rook_magics[i].magic;
        bishop_cached[i] = g_bishop_magics[i].magic;
      }
      out.write(
          reinterpret_cast<const char*>(rook_cached.data()),
          static_cast<std::streamsize>(rook_cached.size() * sizeof(Bitboard)));
      out.write(
          reinterpret_cast<const char*>(bishop_cached.data()),
          static_cast<std::streamsize>(bishop_cached.size() * sizeof(Bitboard)));
    }
  }
}

} // namespace

void init_magic_bitboards() {
  std::call_once(g_magic_once, []() {
    init_magic_tables();
    g_magic_ready = true;
  });
}

bool magic_bitboards_ready() {
  return g_magic_ready;
}

Bitboard rook_attacks_magic(int sq, Bitboard occ) {
#if defined(SKAKS_TESTS)
  return rook_attacks_slow(sq, occ);
#else
  if (!g_magic_ready) {
    init_magic_bitboards();
  }
  const auto& entry = g_rook_magics[static_cast<std::size_t>(sq)];
  if (entry.magic == 0) {
    return rook_attacks_slow(sq, occ);
  }
  const Bitboard occ_masked = occ & entry.mask;
  if (entry.magic != 0) {
    const std::size_t idx =
        static_cast<std::size_t>((occ_masked * entry.magic) >> entry.shift);
    return entry.attacks[idx];
  }
  const std::size_t idx =
      index_from_occupancy(occ_masked, entry.bits, entry.bit_count);
  return entry.attacks[idx];
#endif
}

Bitboard bishop_attacks_magic(int sq, Bitboard occ) {
#if defined(SKAKS_TESTS)
  return bishop_attacks_slow(sq, occ);
#else
  if (!g_magic_ready) {
    init_magic_bitboards();
  }
  const auto& entry = g_bishop_magics[static_cast<std::size_t>(sq)];
  if (entry.magic == 0) {
    return bishop_attacks_slow(sq, occ);
  }
  const Bitboard occ_masked = occ & entry.mask;
  if (entry.magic != 0) {
    const std::size_t idx =
        static_cast<std::size_t>((occ_masked * entry.magic) >> entry.shift);
    return entry.attacks[idx];
  }
  const std::size_t idx =
      index_from_occupancy(occ_masked, entry.bits, entry.bit_count);
  return entry.attacks[idx];
#endif
}

} // namespace chess
