#! /usr/bin/env python
import chess
import chess.pgn

input_path = "game_what.txt"
output_path = "game.pgn"

with open(input_path, "r", encoding="utf-8") as f:
    tokens = f.read().strip().split()

board = chess.Board()
game = chess.pgn.Game()
game.headers["Event"] = "Converted from UCI list"
game.headers["Result"] = "*"

node = game
for uci in tokens:
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {uci} at ply {board.fullmove_number}")
    board.push(move)
    node = node.add_variation(move)

game.headers["Result"] = board.result(claim_draw=True)

with open(output_path, "w", encoding="utf-8") as f:
    print(game, file=f)

print(f"Wrote {output_path}")
