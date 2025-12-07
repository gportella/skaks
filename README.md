# chess_engine_c++

Simple toy engine, for learning

## Prerequisites

- CMake 3.24+
- Ninja (recommended generator)
- clang/clang++ (Apple Clang 15+ or LLVM clang)
- Python 3.11+ if you plan to use additional tooling scripts (not included yet)

Optional:

- `llvm` package (Homebrew) for updated clang-tidy/clang-format tools
- Cross-compilation sysroots for Linux targets (see `cmake/toolchains/`)

## Building

```sh
cmake --preset dev-debug
cmake --build --preset dev-debug
```

Release build with interprocedural optimization:

```sh
cmake --preset dev-release
cmake --build --preset dev-release
```

## Testing

```sh
ctest --preset dev-debug
```

Or run directly:

```sh
cmake --build --preset dev-debug --target chess_engine_tests
ctest --test-dir build/debug --output-on-failure
```

## Cross-compiling

Two sample toolchain presets are provided for Linux x86_64 and arm64 targets (`cmake/toolchains/clang-linux-*.cmake`). Update the sysroot paths to match your environment, then configure:

```sh
cmake --preset linux-cross-x86_64
cmake --build --preset linux-cross-x86_64
```

The presets assume a clang-based cross toolchain (`clang --target=<triple>`). Adjust the compiler paths and `CMAKE_SYSROOT` as needed.

## Tooling

- `.clang-format` and `.clang-tidy` are configured for LLVM-style formatting and broad static analysis.
- `CMakePresets.json` centralizes configuration for development, release, sanitizers, and cross targets.
- Compiler warnings are elevated to errors by default; disable by setting `-DCHESS_ENABLE_WARNINGS_AS_ERRORS=OFF`.
- Sanitizers can be toggled with `-DCHESS_ENABLE_SANITIZERS=ON` (enabled automatically in `dev-debug`).

## Next Steps

- Flesh out the chess engine logic in `include/chess/` and `src/`.
- Add more unit and integration tests under `tests/`.
- Customize toolchain files for your deployment targets.
- Integrate continuous integration (GitHub Actions, etc.) if desired.

## Suggested Build Plan



- [x]  **Represent the board:**
	- Implement a `Board` struct holding piece bitboards, side to move, castling, en-passant, and Zobrist hash.
	- Add FEN parsing and perft-style smoke tests to verify bitboard correctness.
- [ ] **Move generation:**
	- Start with simple pseudo-legal generators for each piece; reuse precomputed masks (files, ranks, knight moves).
	- Upgrade to sliding attacks (rook/bishop/queen) using lookup tables or a magic-bitboard helper when the basics feel solid.
	- Gate complex optimizations behind unit tests that lock in move counts for known positions.
- [ ]  **Make/unmake mechanics:**
	- Implement a stack-based make/unmake (or do/undo) layer that updates bitboards, occupancy, and incremental hash.
	- Add assertions for illegal states and fuzz tests that play random move sequences to ensure consistency.
- [ ]  **Evaluation scaffolding:**
	- Begin with material count plus simple piece-square tables, keeping calculations branch-light.
	- Design the API so evaluation can read only from `Board` and reusable tables; avoid dynamic allocations.
- [ ] **Search framework:**
	- Implement iterative deepening with alpha-beta, quiescence, and basic move ordering (captures first, killers later).
	- Introduce a transposition table (fixed-size array of aligned buckets). Start single-threaded to keep reasoning simple.
- [ ] **Parallel search (optional later):**
	- When ready, layer a thread pool on top of the single-threaded search.
	- Prefer work-stealing or per-thread stacks to minimize locking. Guard shared structures (TT, endgame tablebases) with atomics or fine-grained locks only where measurements prove contention.
	- Use Google Benchmark or integration tests to detect regressions introduced by synchronization bugs.

## Gotchas & Safety Nets

- **Threading:** stick to single-threaded search until all core logic is fuzz- and perft-tested. When adding threads, keep shared state minimal, and run ThreadSanitizer builds (`asan-debug` preset with `-fsanitize=thread` swapped in) to expose data races.
- **Undefined behavior:** prefer unsigned bitboards and explicit masks; sanitize shifts (`1ULL << square`) by ensuring `square < 64`.
- **Debug vs release:** run tests under both `dev-debug` and `dev-release`; performance-only issues (like uninitialized data) often appear only at `-O3`.
- **Benchmark drift:** regression-test move generation with `perft` counts at each milestone so later optimizations do not silently break correctness.
- **Tooling discipline:** keep `.clang-tidy` warnings actionable—disable checks only with justification to avoid masking future mistakes.

## Design Hints for Future Parallelism

- Favor value semantics: pass `Board` and `SearchLimits` by value or `const&`, return lightweight structs (principal variation, score) so callers are not coupled to shared pointers.
- Build modules around pure functions (move generation, evaluation, hashing) and keep orchestration separate; it keeps threading a thin wrapper over existing logic.
- Introduce a `SearchContext` (per-thread scratch buffers, killer/history tables) even while single-threaded; later you can hand one to each worker without rewiring APIs.
- Wrap shared structures such as the transposition table behind small accessors so you can swap in atomics or fine-grained locks without touching call sites.
- Prefer Plain Old Data (POD) types for hot structs: contiguous, trivially copyable layouts keep caches happy and make per-thread copies cheap.
- Keep memory predictable: use `std::array` for fixed data, `std::vector::reserve` for growing lists, and central allocators (arena or object pool) for transient nodes. Avoid sprinkling `new`/`delete` inside the search loop.
- Document invariants (e.g., "move generator returns only pseudo-legal moves") right where they are enforced so future parallel work has clear contracts.
- Maintain regression suites (perft tables, fuzz tests) and run them before and after structural changes; catching race-induced bugs is easier with a strong baseline.
