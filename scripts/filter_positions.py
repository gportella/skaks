#! /usr/bin/env python
import pandas as pd
from pathlib import Path

src = Path("eval_pairs_pvs_with_results.csv")
out = Path("eval_pairs_pvs_ply8_80_cp2k.csv")

df = pd.read_csv(src)
mask = df["ply"].between(8, 80)
df = df[mask]
# Drop obvious mates/huge outliers; adjust cap if you like
df = df[df["stockfish_cp"].between(-2000, 2000)]

# Optional: dedup on FEN to reduce repetition
df = df.drop_duplicates(subset=["fen"])

df.to_csv(out, index=False)
print("wrote", out, "rows", len(df))
print("ply min/max", df["ply"].min(), df["ply"].max())
print(
    "cp quantiles",
    df["stockfish_cp"].quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).to_dict(),
)
