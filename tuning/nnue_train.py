import argparse
import json
import math
import random
from pathlib import Path
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


class TensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        assert features.shape[0] == labels.shape[0]
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


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
    eval_clip: Optional[float],
    regression: bool,
    target_scale: float,
    pov: str,
    prefer_eval: bool,
    min_ply: Optional[int],
    max_ply: Optional[int],
    sample_fraction: float,
    max_rows: Optional[int],
    seed: int,
) -> List[Tuple[str, float]]:
    rng = random.Random(seed)
    rows: List[Tuple[str, float]] = []
    chunk = 200_000
    for df in pd.read_csv(path, chunksize=chunk):
        if fen_col not in df.columns:
            raise ValueError(f"Missing fen column '{fen_col}' in {path}")

        if min_ply is not None or max_ply is not None:
            if "ply" in df.columns:
                if min_ply is not None:
                    df = df[df["ply"] >= min_ply]
                if max_ply is not None:
                    df = df[df["ply"] <= max_ply]
            else:
                # no ply column; keep all
                pass

        if df.empty:
            continue

        if sample_fraction < 1.0:
            mask = [rng.random() < sample_fraction for _ in range(len(df))]
            df = df.loc[mask]
            if df.empty:
                continue

        if (
            outcome_col
            and outcome_col in df.columns
            and not prefer_eval
            and not regression
        ):
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
            if eval_clip is not None and eval_clip > 0:
                cp = cp.clip(-eval_clip, eval_clip)
            if pov == "side" and "side_to_move" in df.columns:
                stm = df["side_to_move"].apply(
                    lambda s: 1 if str(s).lower().startswith("w") else -1
                )
                cp = cp * stm
            if regression:
                labels = cp.astype(float) / float(target_scale)
            else:
                labels = 1.0 / (1.0 + (-cp / cp_scale).apply(math.exp))

        for fen_val, label_val in zip(df[fen_col].tolist(), labels.tolist()):
            rows.append((fen_val, label_val))
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def precompute_features(
    rows: Sequence[Tuple[str, float]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[torch.Tensor] = []
    labels: List[float] = []
    for fen, label in rows:
        feats.append(torch.from_numpy(sk.features_from_fen(fen)).to(torch.float32))
        labels.append(float(label))
    X = torch.stack(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.float32)
    if device.type == "cuda":
        X = X.pin_memory()
    return X, y


def load_init_weights(path: Path) -> Tuple[Optional[dict], Optional[int]]:
    if not path.exists():
        raise FileNotFoundError(f"init weights not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "nnue" not in data:
            raise ValueError("JSON does not contain 'nnue' section")
        nnue = data["nnue"]
        hidden = int(nnue.get("hidden", 256))
        scale = float(nnue.get("scale", 1.0))
        w1 = torch.tensor(nnue["w1"], dtype=torch.float32).reshape(hidden, INPUT_SIZE)
        b1 = torch.tensor(nnue["b1"], dtype=torch.float32)
        w2 = torch.tensor(nnue["w2"], dtype=torch.float32).reshape(1, hidden) * scale
        b2 = torch.tensor([nnue["b2"]], dtype=torch.float32) * scale
        state = {
            "fc1.weight": w1,
            "fc1.bias": b1,
            "fc2.weight": w2,
            "fc2.bias": b2,
        }
        return state, hidden
    else:
        blob = torch.load(path, map_location="cpu")
        if isinstance(blob, dict) and all(k.startswith("fc") for k in blob.keys()):
            hidden = int(blob["fc1.weight"].shape[0]) if "fc1.weight" in blob else None
            return blob, hidden
        raise ValueError(
            "Unsupported init weights format; expect state_dict .pt or nnue JSON"
        )


def export_yaml(
    model: TinyNnue, path: str, scale: float = 1.0, quant_gain: float = 1.0
) -> None:
    state = model.state_dict()
    w1 = state["fc1.weight"].cpu() * quant_gain
    b1 = state["fc1.bias"].cpu() * quant_gain
    w2 = state["fc2.weight"].cpu() * quant_gain
    b2 = state["fc2.bias"].cpu() * quant_gain

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
    precomputed: Optional[Tuple[torch.Tensor, torch.Tensor]],
    num_workers: int,
    weight_decay: float,
    stratified_split: bool,
    init_state: Optional[dict],
    device: torch.device,
    label_smoothing: float,
    lr_scheduler: str,
    lr_step: int,
    lr_gamma: float,
    early_stop_patience: int,
    min_epochs: int,
    pos_weight: Optional[float],
    regression: bool,
    regression_loss: str,
) -> TinyNnue:
    if precomputed is not None:
        feats, labels = precomputed
        ds = TensorDataset(feats, labels)
    else:
        ds = FenDataset(rows, cache_features=cache_features)
    if val_split > 0.0:
        total_len = len(ds)
        val_size = int(total_len * val_split)
        if stratified_split and val_size > 0:
            if isinstance(ds, TensorDataset):
                labels_seq = ds.labels.tolist()
            else:
                labels_seq = [float(lbl) for _, lbl in rows[:total_len]]
            bins = [1 if label_val >= 0.5 else 0 for label_val in labels_seq]
            idx0 = [i for i, bin_val in enumerate(bins) if bin_val == 0]
            idx1 = [i for i, bin_val in enumerate(bins) if bin_val == 1]
            random.shuffle(idx0)
            random.shuffle(idx1)
            val0 = int(len(idx0) * val_split)
            val1 = val_size - val0
            val_indices = (idx0[:val0] + idx1[:val1])[:val_size]
            train_indices = idx0[val0:] + idx1[val1:]
            train_ds = torch.utils.data.Subset(ds, train_indices)
            val_ds = torch.utils.data.Subset(ds, val_indices)
        else:
            train_size = total_len - val_size
            train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = ds, None

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": max(0, num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    dl = DataLoader(train_ds, **loader_kwargs)
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(val_ds, **val_loader_kwargs)

    model = TinyNnue(hidden=hidden, dropout=dropout).to(device)
    if init_state:
        try:
            model.load_state_dict(init_state, strict=False)
        except Exception as exc:
            print(f"Warning: failed to load init weights: {exc}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if regression:
        if regression_loss == "smoothl1":
            loss_fn = nn.SmoothL1Loss()
        else:
            loss_fn = nn.MSELoss()
    else:
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
            if not regression and label_smoothing > 0.0:
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
                    if not regression and label_smoothing > 0.0:
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
        "--cp-scale",
        type=float,
        default=600.0,
        help="Sigmoid scale for eval->prob (classification mode)",
    )
    parser.add_argument(
        "--eval-clip",
        type=float,
        default=None,
        help="Clamp eval_cp to +/- this many centipawns before scaling",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Train on evals directly (regression); labels become scaled evals",
    )
    parser.add_argument(
        "--eval-target-scale",
        type=float,
        default=600.0,
        help="Divide evals by this before regression (set to expected eval range)",
    )
    parser.add_argument(
        "--regression-loss",
        choices=["smoothl1", "mse"],
        default="smoothl1",
        help="Loss to use in regression mode",
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
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay",
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
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for feature computation",
    )
    parser.add_argument(
        "--precompute-features",
        type=str,
        help="Path to save/load precomputed features (.pt). If exists, load; otherwise compute and save before training.",
    )
    parser.add_argument(
        "--save-fp32",
        type=str,
        default=None,
        help="Optional path to save fp32 state_dict before quantization",
    )
    parser.add_argument(
        "--init-weights",
        type=str,
        help="Initialize model from weights file (torch state_dict .pt or exported nnue JSON)",
    )
    parser.add_argument(
        "--stratified-split",
        action="store_true",
        help="Use stratified train/val split on labels",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=None,
        help="Minimum ply to include from CSV",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=None,
        help="Maximum ply to include from CSV",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of rows (0-1]",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap number of rows to load after filtering",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="RNG seed for sampling",
    )
    parser.add_argument(
        "--out", default="nnue_weights.json", help="Output YAML/JSON file"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Output scaling factor"
    )
    parser.add_argument(
        "--quant-gain",
        type=float,
        default=1.0,
        help="Multiply weights/biases by this before int quantization to avoid zeros",
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
        eval_clip=args.eval_clip,
        regression=args.regression,
        target_scale=args.eval_target_scale,
        pov=args.pov,
        prefer_eval=args.prefer_eval,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        sample_fraction=args.sample_fraction,
        max_rows=args.max_rows,
        seed=args.sample_seed,
    )
    print(f"Loaded {len(rows)} rows")

    precomputed: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if args.precompute_features:
        pre_path = Path(args.precompute_features)
        if pre_path.exists():
            blob = torch.load(pre_path, map_location="cpu")
            precomputed = (blob["features"], blob["labels"])
            print(f"Loaded precomputed features from {pre_path}")
        else:
            print(f"Precomputing features to {pre_path} ...")
            X, y = precompute_features(rows, device=torch.device("cpu"))
            torch.save({"features": X, "labels": y}, pre_path)
            precomputed = (X, y)
            print(f"Saved precomputed features to {pre_path}")

    init_state: Optional[dict] = None
    if args.init_weights:
        state, hidden_override = load_init_weights(Path(args.init_weights))
        init_state = state
        if hidden_override:
            args.hidden = hidden_override
            print(f"Initializing from {args.init_weights} (hidden={hidden_override})")

    model = train(
        rows=rows,
        hidden=args.hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        cache_features=args.cache_features,
        precomputed=precomputed,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        stratified_split=args.stratified_split,
        init_state=init_state,
        device=device,
        label_smoothing=args.label_smoothing,
        lr_scheduler=args.lr_scheduler,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        pos_weight=args.pos_weight,
        regression=args.regression,
        regression_loss=args.regression_loss,
    )

    if args.save_fp32:
        torch.save(model.state_dict(), args.save_fp32)
        print(f"Saved fp32 state_dict to {args.save_fp32}")

    export_yaml(model, args.out, scale=args.scale, quant_gain=args.quant_gain)
    print(f"Wrote quantized weights to {args.out}")


if __name__ == "__main__":
    main()
