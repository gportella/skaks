#!/usr/bin/env python
import sys
import chess

def position_to_fen(cmd: str) -> str:
    tokens = cmd.strip().split()
    if len(tokens) < 2 or tokens[0] != "position" or tokens[1] != "startpos":
        raise ValueError("Expected: position startpos moves ...")
    board = chess.Board()
    if "moves" in tokens:
        idx = tokens.index("moves") + 1
        for uci in tokens[idx:]:
            board.push_uci(uci)
    return board.fen()

if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read().strip()
    print(position_to_fen(cmd))
