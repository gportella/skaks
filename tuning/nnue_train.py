import argparse
import json
import math
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import skaks_eval as sk


# Quantization helpers matching engine side
Q8_MIN, Q8_MAX = -127, 127
INT32_MIN, INT32_MAX = -2147483648, 2147483647


def q8(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x), Q8_MIN, Q8_MAX).to(torch.int8)


def q32(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x), INT32_MIN, INT32_MAX).to(torch.int32)


class FenDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[str, float]],
        cache_features: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.cache_features = cache_features
        if cache_features:
            self._feat_cache: Optional[List[Optional[torch.Tensor]]] = [
                None for _ in range(len(self.rows))
            ]
        else:
            self._feat_cache = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        fen, label = self.rows[idx]
        if self.cache_features:
            assert self._feat_cache is not None
            if self._feat_cache[idx] is None:
                feats = torch.from_numpy(sk.features_from_fen(fen)).to(torch.float32)
                self._feat_cache[idx] = feats
            feats = self._feat_cache[idx]
        else:
            feats = torch.from_numpy(sk.features_from_fen(fen)).to(torch.float32)
        return feats, torch.tensor(label, dtype=torch.float32)


INPUT_SIZE = 1537  # kNnueInputs; bindings don't expose the constant


class TinyNnue(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


def load_rows(
    path: str,
    fen_col: str,
    outcome_col: Optional[str],
    eval_col: Optional[str],
    cp_scale: float,
    pov: str,
    prefer_eval: bool,
) -> List[Tuple[str, float]]:
    df = pd.read_csv(path)
    if fen_col not in df.columns:
        raise ValueError(f"Missing fen column '{fen_col}' in {path}")
    if outcome_col and outcome_col in df.columns and not prefer_eval:
        labels = df[outcome_col].astype(float)
    else:
        candidate_eval_cols = []
        if eval_col:
            candidate_eval_cols.append(eval_col)
        candidate_eval_cols.extend(["eval_cp", "stockfish_cp", "skaks_cp"])
        chosen = None
        for c in candidate_eval_cols:
            if c in df.columns:
                chosen = c
                break
        if not chosen:
            raise ValueError(
                "Provide outcome_col or an eval column (eval_cp/stockfish_cp/skaks_cp) present in CSV"
            )

        cp = df[chosen].astype(float)
        if pov == "side" and "side_to_move" in df.columns:
            stm = df["side_to_move"].apply(
                lambda s: 1 if str(s).lower().startswith("w") else -1
            )
            cp = cp * stm
        labels = 1.0 / (1.0 + (-cp / cp_scale).apply(math.exp))
    return list(zip(df[fen_col].tolist(), labels.tolist()))


def export_yaml(model: TinyNnue, path: str, scale: float = 1.0) -> None:
    state = model.state_dict()
    w1 = state["fc1.weight"].cpu()
    b1 = state["fc1.bias"].cpu()
    w2 = state["fc2.weight"].cpu()
    b2 = state["fc2.bias"].cpu()

    w1_q = q8(w1).reshape(-1).tolist()
    b1_q = q32(b1).tolist()
    w2_q = q8(w2).reshape(-1).tolist()
    b2_q = int(q32(b2)[0].item())

    payload = {
        "nnue": {
            "hidden": w2.shape[1] if w2.dim() == 2 else len(w2_q),
            "w1": w1_q,
            "b1": b1_q,
            "w2": w2_q,
            "b2": b2_q,
            "scale": float(scale),
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))


def train(
    rows: Sequence[Tuple[str, float]],
    hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    cache_features: bool,
    device: torch.device,
    label_smoothing: float,
    lr_scheduler: str,
    lr_step: int,
    lr_gamma: float,
    early_stop_patience: int,
    min_epochs: int,
    pos_weight: Optional[float],
) -> TinyNnue:
    ds = FenDataset(rows, cache_features=cache_features)
    if val_split > 0.0:
        val_size = int(len(ds) * val_split)
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = ds, None

    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinyNnue(hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pw = None
    if pos_weight is not None and pos_weight > 0.0:
        pw = torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    scheduler = None
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_step, gamma=lr_gamma
        )

    best_state = None
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if label_smoothing > 0.0:
                yb = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)
        if val_dl is not None:
            model.eval()
            val_loss = 0.0
            assert val_ds is not None
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    if label_smoothing > 0.0:
                        yb = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_ds)
            print(
                f"epoch {epoch + 1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}"
            )

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (
                early_stop_patience > 0
                and epoch + 1 >= min_epochs
                and no_improve >= early_stop_patience
            ):
                print("Early stopping triggered")
                break
        else:
            print(f"epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        if (
            early_stop_patience > 0
            and val_dl is not None
            and epoch + 1 >= min_epochs
            and no_improve >= early_stop_patience
        ):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny NNUE on FEN data")
    parser.add_argument(
        "--data", required=True, help="Path to CSV with FENs and labels"
    )
    parser.add_argument("--fen-col", default="fen", help="Column with FEN strings")
    parser.add_argument(
        "--outcome-col", default="outcome", help="Column with targets in [0,1]"
    )
    parser.add_argument(
        "--eval-col",
        default=None,
        help="Alternative column with centipawn evals to convert",
    )
    parser.add_argument(
        "--prefer-eval",
        action="store_true",
        help="Use eval_col even if outcome_col exists",
    )
    parser.add_argument(
        "--cp-scale", type=float, default=600.0, help="Sigmoid scale for eval->prob"
    )
    parser.add_argument(
        "--pov",
        choices=["white", "side"],
        default="side",
        help="How to interpret eval_col (side uses side_to_move column if present)",
    )
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout after first layer"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "step"],
        default="none",
        help="Optional LR scheduler",
    )
    parser.add_argument("--lr-step", type=int, default=10, help="Step size for step LR")
    parser.add_argument(
        "--lr-gamma", type=float, default=0.5, help="Decay factor for step LR"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Apply label smoothing to targets",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive class weight for BCE (win side)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if no val improvement for N epochs (requires val split)",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Do not early-stop before this many epochs",
    )
    parser.add_argument(
        "--cache-features", action="store_true", help="Cache features in RAM"
    )
    parser.add_argument(
        "--out", default="nnue_weights.json", help="Output YAML/JSON file"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Output scaling factor"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    rows = load_rows(
        path=args.data,
        fen_col=args.fen_col,
        outcome_col=args.outcome_col,
        eval_col=args.eval_col,
        cp_scale=args.cp_scale,
        pov=args.pov,
        prefer_eval=args.prefer_eval,
    )
    print(f"Loaded {len(rows)} rows")

    model = train(
        rows=rows,
        hidden=args.hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        cache_features=args.cache_features,
        device=device,
        label_smoothing=args.label_smoothing,
        lr_scheduler=args.lr_scheduler,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        pos_weight=args.pos_weight,
    )

    export_yaml(model, args.out, scale=args.scale)
    print(f"Wrote quantized weights to {args.out}")


if __name__ == "__main__":
    main()
