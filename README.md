# Skaks -- learning by doing 

> Uses Stockfish NNUE as I was tired of tuning my eval, no incremental eval for NNUE yet.
> This whole thing then becomes GLPv3, will slowly add the following in the source code:t -C /Users/guillem/work/repos/skaks status -s
t -C /Users/guillem/work/repos/skaks status -s | cat

> // SPDX-License-Identifier: GPL-3.0-or-later

Simple toy chess engine, for learning and fun. 

When executed, it does `UCI` by default (just run `skaks`, or `skaks --uci`). It can self-play `skaks -s` from canonical start or from `--fen` position. 

There's a `--perf` option, useful for regressions. I leave here a couple of scripts to track the performance, both in therms of nodes/ms and 
and "quality" in puzzle solving -- not awesome, about 12% if lucky for *tricky* (?) ones.

> Careful, there are still some corner cases that might not be great, need to check the 50 moves rule

## Current version 

```test
> skaks -vv

skaks version 0.16.4
Optimizations:
 - Bitboard move generation with precomputed attack masks
 - Alpha-beta search with transposition table caching
 - Zobrist hashing for incremental board state keys
 - Repetition detection through historical position tracking
 - Move ordering seeded by cached transposition moves
 - PV search with some reductions
 - Root move exclusion support for iterative deepening
 - Quiescence search to reduce horizon effect
 - Killer move heuristic for quiet move ordering
 - Support for polyglot book of moves
 - Null move pruning, 2-ply historical heuristic and SEE sorting
 - Time management for search limits
 - MVV-LVA and SEE for capture move ordering
 - Threaded UCI search support, with pondering
 - Parameter loading from external file
 - SIMD optimizations for bitboard operations and threads
 - Non-incremental NNUE from Stockfish
 - Extensive CLI modes for self-play, benchmarking, and profiling
 - Futility and SEE pruning in search
 - Compiled with NEON eval_linear path

```


## Prerequisites

- CMake 3.24+
- Ninja (recommended generator)
- clang/clang++ (Apple Clang 15+ or LLVM clang)
- Python 3.11+ if you plan to use additional tooling for perf benchamarking and puzzles.
- `stockfish` or any other engine to challange.

Optional:

- `llvm` package (Homebrew) for updated clang-tidy/clang-format tools
- Cross-compilation sysroots for Linux targets (see `cmake/toolchains/`)

## Building

```sh
cmake --preset dev-debug
cmake --build --preset dev-debug
```

The binary is called `skaks`, there's some help in `skaks -h`.

Release build with interprocedural optimization:

```sh
cmake --preset dev-release
cmake --build --preset dev-release
```

## Installing

After building, install the `skaks` executable to your chosen prefix:

```sh
cmake --install build/debug
```

To place the binary directly in `~/bin`, configure with a custom install directory and override the prefix during install:

```sh
cmake --preset dev-debug -DSKAKS_INSTALL_BINDIR=.
cmake --build --preset dev-debug
cmake --install build/debug --prefix "$HOME/bin"
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

