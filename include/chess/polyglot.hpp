#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <optional>
#include <string>
#include <span>

#include "chess/board.hpp"
#include "chess/moves.hpp"
#include "chess/types.hpp"

namespace chess::polyglot {

// Entry as stored in memory
struct Entry {
  uint16_t move;   // Polyglot 16-bit move encoding
  uint16_t weight; // frequency/weight
  uint32_t learn;  // unused (from file), kept for completeness
};

// Book container: map from 64-bit polyglot key to entries
using Book = std::unordered_map<uint64_t, std::vector<Entry>>;

// Compute Polyglot hash for a Board
uint64_t compute_key(const Board& b);

// Load a .bin file into memory. Returns true on success.
bool load_book(const std::string& path, Book& out);

// Decode Polyglot move to engine Move (as encoded uint32_t), or std::nullopt if illegal/unmappable.
// Promo mapping uses side-to-move color from Board b.
std::optional<uint32_t> decode_move(uint16_t poly_move, const Board& b);

// Lookup candidates for a position. Returns span to internal vector for convenience.
std::span<const Entry> lookup(const Book& book, const Board& b);

// Select a move from entries:
// - if use_weighted = true: weighted random by Entry.weight (seed provided)
// - else: highest weight
// Returns encoded engine move or nullopt if none legal.
std::optional<uint32_t> select(const Board& b, std::span<const Entry> entries,
                               bool use_weighted, uint64_t seed);

// Convenience: try book at root, return chosen move or nullopt.
std::optional<uint32_t> choose_move(const Book& book, const Board& b,
                                    bool use_weighted, uint64_t seed);

} // namespace chess::polyglot