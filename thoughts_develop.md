
# Next Steps

## Explore `<bit>`

I read that the new `<bit>` is better for bitwise opps and bitcouting. Will need to make sure to check it out.

> TODO: incremental Zobrist key, got lazy
> Tapering position tables as game progresses, got tangled up in search.


## Suggested Build Plan


- [x]  **Represent the board:**
	- Implement a `Board` struct holding piece bitboards, side to move, castling, en-passant, and Zobrist hash.
	- Add FEN parsing and perft-style smoke tests to verify bitboard correctness.
- [x] **Move generation:**
	- Start with simple pseudo-legal generators for each piece; reuse precomputed masks (files, ranks, knight moves).
	- Upgrade to sliding attacks (rook/bishop/queen) using lookup tables or a magic-bitboard helper when the basics feel solid.
	- Gate complex optimizations behind unit tests that lock in move counts for known positions.
- [x]  **Make/unmake mechanics:**
	- Implement a stack-based make/unmake (or do/undo) layer that updates bitboards, occupancy, and incremental hash.
	- Add assertions for illegal states and fuzz tests that play random move sequences to ensure consistency.
- [x]  **Evaluation scaffolding:**
	- Begin with material count plus simple piece-square tables, keeping calculations branch-light.
	- Design the API so evaluation can read only from `Board` and reusable tables; avoid dynamic allocations.
-  [x] **Search framework:**
	- Implement iterative deepening with alpha-beta, quiescence, and basic move ordering (captures first, killers later).
	- Introduce a transposition table (fixed-size array of aligned buckets). Start single-threaded to keep reasoning simple.
    - Introduce `perf` and `puzzles` to see how well each change does.
- [ ] **Parallel search (optional later):**
	- When ready, layer a thread pool on top of the single-threaded search.
	- Prefer work-stealing or per-thread stacks to minimize locking. Guard shared structures (TT, endgame tablebases) with atomics or fine-grained locks only where measurements prove contention.
	- Use Google Benchmark or integration tests to detect regressions introduced by synchronization bugs.

## Gotchas 

- **Threading:** stick to single-threaded search until all core logic is fuzz- and perft-tested. When adding threads, keep shared state minimal, and run ThreadSanitizer builds (`asan-debug` preset with `-fsanitize=thread` swapped in) to expose data races.
- **Undefined behavior:** prefer unsigned bitboards and explicit masks; sanitize shifts (`1ULL << square`) by ensuring `square < 64`.
- **Debug vs release:** run tests under both `dev-debug` and `dev-release`; performance-only issues (like uninitialized data) often appear only at `-O3`.
- **Benchmark drift:** regression-test move generation with `perft` counts at each milestone so later optimizations do not silently break correctness.

## Design Hints for Future Parallelism

- Favor value semantics: pass `Board` and `SearchLimits` by value or `const&`, return lightweight structs (principal variation, score) so callers are not coupled to shared pointers.
- Build modules around pure functions (move generation, evaluation, hashing) and keep orchestration separate; it keeps threading a thin wrapper over existing logic.
- Introduce a `SearchContext` (per-thread scratch buffers, killer/history tables) even while single-threaded; later you can hand one to each worker without rewiring APIs.
- Wrap shared structures such as the transposition table behind small accessors so you can swap in atomics or fine-grained locks without touching call sites.
- Prefer Plain Old Data (POD) types for hot structs: contiguous, trivially copyable layouts keep caches happy and make per-thread copies cheap.
- Keep memory predictable: use `std::array` for fixed data, `std::vector::reserve` for growing lists, and central allocators (arena or object pool) for transient nodes. Avoid sprinkling `new`/`delete` inside the search loop.
- Document invariants (e.g., "move generator returns only pseudo-legal moves") right where they are enforced so future parallel work has clear contracts.
- Maintain regression suites (perft tables, fuzz tests) and run them before and after structural changes; catching race-induced bugs is easier with a strong baseline.



## Make sure to follow the rules 

I think I cover most of the repetition rules and so on, but not guaranteed...