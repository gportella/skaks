# Skaks -- learning by doing 

> Uses Stockfish NNUE as I was tired of tuning my eval, no incremental eval.

Simple toy chess engine, for learning and fun. 

When executed, it does `UCI` by default (just run `skaks`, or `skaks --uci`). It can self-play `skaks -s` from canonical start or from `--fen` position. 
Use `--eval stockfish` together with `--nnue /path/to/net.nnue` to run the bundled Stockfish NNUE evaluator (the legacy `sunfish` spelling remains as an alias).
Default depth I think it's 4 right now, as I improve it we can allow ourselves better depth. No time limits yet. 

There's a `--perf` option, useful for regressions. I leave here a couple of scripts to track the performance, both in therms of nodes/ms and 
and "quality" in puzzle solving -- not awesome, about 12% if lucky for *tricky* (?) ones.

> Careful, there are still some corner cases that might not be great, need to check the 50 moves rule, and perhas some legal move generation issues.

## Current version 

```test
> skaks -vv

skaks version 0.9.5
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
 - Null move pruning, historical heuristic and SEE sorting
 - Time management for search limits
 - Incremental evaluation with piece-square tables
 - MVV-LVA and SEE for capture move ordering
 - Threaded UCI search support, with pondering
 - Parammeter loading from external file
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

## Evaluating and tuning

- `validation_moves/eval_from_pgn.py` takes a PGN file (one or many games) and prints a CSV with both Stockfish and skaks scores for each position. Example:

	```sh
	python validation_moves/eval_from_pgn.py --pgn my_games.pgn --stockfish /usr/local/bin/stockfish > evals.csv
	```

- The `tuning/` scripts consume those score pairs to train new eval parameters and write YAML configs you can pass back to the engine. After collecting more data, rerun tuning to update the parameters.

- Phase split helper for Texel/analysis datasets:

	```sh
	# Split a CSV (fen + score/stockfish_cp) into opening/middlegame/endgame buckets
	skaks-opt fen-phase-split \
	  --input tuning/texel_fit_quiet.csv \
	  --output-dir tuning/phased

	# Or write a single CSV with a phase column
	skaks-opt fen-phase-split --input tuning/texel_fit_quiet.csv --phase-column
	```


### Phase-weight-only tuning

Use these when you only want to optimize the phase weights (`phase_weights_mg` / `phase_weights_eg`) already present in `tuning/best_params.yaml`.

```bash
# CP regression on eval_pairs CSV (Optuna)
python -m skaks_opt fit \
	--data eval_pairs_pvs_with_results.csv \
	--phase-weights-only \
	--trials 80 \
	--threads 0 \
	--batch-size 512 \
	--pov side \
	--best-out tuning/best_params_phase.yaml

# Self-play optimizer
skaks-opt param-optimize \
	--start-params tuning/best_params.yaml \
	--phase-weights-only \
	--output tuning/best_params_phase.yaml

# Quick internal arena (baseline vs tuned params)
./build/debug/bin/skaks --arena --params tuning/best_params_phase.yaml --arena-games 200
```

## Cross-compiling

Two sample toolchain presets are provided for Linux x86_64 and arm64 targets (`cmake/toolchains/clang-linux-*.cmake`). Update the sysroot paths to match your environment, then configure:

```sh
cmake --preset linux-cross-x86_64
cmake --build --preset linux-cross-x86_64
```

The presets assume a clang-based cross toolchain (`clang --target=<triple>`). Adjust the compiler paths and `CMAKE_SYSROOT` as needed.

## Tooling

There is a `tunning` directory with Python scripts to help with performance benchmarking and puzzle solving. These scripts require Python 3.11+ and can be run as follows:

```bashpython3 tunning/perf_benchmark.py --help
```
```bash


```

- `.clang-format` and `.clang-tidy` are configured for LLVM-style formatting and broad static analysis.
- `CMakePresets.json` centralizes configuration for development, release, sanitizers, and cross targets.
- Compiler warnings are elevated to errors by default; disable by setting `-DCHESS_ENABLE_WARNINGS_AS_ERRORS=OFF`.
- Sanitizers can be toggled with `-DCHESS_ENABLE_SANITIZERS=ON` (enabled automatically in `dev-debug`).
