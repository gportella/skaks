# Skaks chess engine

Simple toy chess engine, for learning and fun. 

> Uses Stockfish NNUE as I was tired of tuning my eval. There's at least 200 ELO point difference between the classical heuristic and the NNUE version. The incremental eval for NNUE is ok, though done via an adaptor. Even with the Stockfish NNUE and all my efforts to improve search, it's not a very strong engine, 2000 ELO tops.

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

### HCE vs NNUE

I did not manage to improve the classical heuristics very much, I don't have the computational resources to do so. Borrowed some PST from various engines and tried with `Texel`, but a good fit against the `win-lose-draw` curve from quiet positions obtained from millions of high-profile games did not really translate into great play for me. `SPSA` + SPRT is cool, but don't have the time and resources, and the evaluation parameters were improving very slowly. Did not want to add more terms to the mix. Therefore I took the approach to *borrow* Stockfish's NNUE, and spend the time coding search. Not awesome, but much better now, quick and dirty comparison with `fastchess`:

```
fastchess -rounds 1200 \
-engine cmd=skaks name=Skaks-NNUE \
-engine cmd=skaks_hce name=Skaks-HCE \
-openings file=lichess_elite_2015-09.pgn format=pgn order=random \ 
-repeat \
-concurrency 4 \
-sprt elo0=0 elo1=2 alpha=0.05 beta=0.05  
-each tc=5+0.1  

[...]
--------------------------------------------------
Results of Skaks-NNUE vs Skaks-HCE (5+0.1, 14t, NULL, lichess_elite_2015-09.pgn):
Elo: 232.74 +/- 23.37, nElo: 292.26 +/- 22.05
LOS: 100.00 %, DrawRatio: 40.04 %, PairsRatio: 285.00
Games: 954, Wins: 739, Losses: 181, Draws: 34, Points: 756.0 (79.25 %)
Ptnml(0-2): [1, 0, 191, 10, 275], WL/DD Ratio: 14.92
LLR: 2.95 (100.0%) (-2.94, 2.94) [0.00, 2.00]
--------------------------------------------------
SPRT ([0.00, 2.00]) completed - H1 was accepted
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

