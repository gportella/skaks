#!/bin/bash

# Define the path to the actual engine and the NNUE file
ENGINE_PATH="/usr/local/bin/skaks"
NNUE_FILE="/Users/ktxc111/work/repos/skaks/nn-vdv.nnue"
PARAM_FILE="/Users/ktxc111/work/repos/skaks/besty_optimized.yaml"

# Execute the engine with the required command-line arguments
#exec "$ENGINE_PATH" --nnue "$NNUE_FILE" --uci
exec "$ENGINE_PATH" --params "$PARAM_FILE" --uci
