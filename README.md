# Skaks chess engine

Simple toy chess engine, for learning and fun. 

> Uses Stockfish NNUE as I was tired of tuning my eval. There's at least 400 ELO point difference between the classical heuristic and the NNUE version. The incremental eval for NNUE is ok, though done via an adaptor. Even with the Stockfish NNUE and all my efforts to improve search, it's not a very strong engine, 2000 ELO tops.


When executed, it does `UCI` by default (just run `skaks`, or `skaks --uci`). Other options available via cli, `skaks -h`.

By default, the NNUE is active. To use the old heuristics (*aka* HCE), do `skaks --no-nnue --uci`, do use `--uci` or `-u` to enable `UCI` mode or else it starts self-play.

By default it uses as many threads as logical CPU cores, but you can change this via `--threads`. 

Supports time controls (though could be better), fixed-depth or fixed-node search (fixed node is single-threaded for now), both via cli or `UCI`. The most significant (??) search params can be set via cli or `UCI`, besides the python bindings (see further down).


## Current version 

```test
> skaks -vv

skaks version 0.17.0
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
 - Added MVV-LVA and SEE for capture move ordering
 - Threaded UCI search support, with pondering
 - Parameter loading from external file
 - SIMD optimizations for bitboard operations and threads
 - Incremental eval NNUE from Stockfish
 - Extensive CLI modes for self-play, benchmarking, and profiling
 - Futility and SEE pruning in search
 - Compiled with NEON eval_linear path
```


#### Aditional helpers 

There's a `--perf` option, useful for regressions. I leave here a couple of scripts to track the performance in terms of number of nodes and nodes/ms. It's not very significant, as the real performance can be best seen via play. 

There's a small python binding in `bindings/python`, a library called `skaks_eval`, which makes it easier to play around with the evals, do a large number of self-play with different sets of params, etc. I use these in `skaks-opt`, which is a *cli* for parameter tuning, etc. No documentation for now, too many changes and needs clean-up.

First install the `skaks_eval` library, e.g. 

```bash
cd bindings/python
pip install -e . 
```

and then you can install `skaks-opt`, from this directory, also using `pip install -e .` or similar. 

`skaks-opt` has a subcommand, `skaks-opt perf-pgn` that allows to profile the execution a bit better by playing some games, it's good to check the miliseconds per ply.


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

