import argparse
import contextlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import chess
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import amp as torch_amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

# HalfKP constants
NUM_KING_SQ = 64
NUM_PIECE_SQ = 64
NUM_PIECE_TYPES = 10  # W: P,N,B,R,Q=0..4, B: P,N,B,R,Q=5..9
NUM_FEATURES = NUM_KING_SQ * NUM_PIECE_SQ * NUM_PIECE_TYPES  # 40960
PT_MAP = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.BLACK, chess.PAWN): 5,
    (chess.BLACK, chess.KNIGHT): 6,
    (chess.BLACK, chess.BISHOP): 7,
    (chess.BLACK, chess.ROOK): 8,
    (chess.BLACK, chess.QUEEN): 9,
}


def halfkp_id(king_sq: int, piece_sq: int, pt: int) -> int:
    return king_sq + NUM_KING_SQ * piece_sq + NUM_KING_SQ * NUM_PIECE_SQ * pt


def extract_halfkp(board: chess.Board) -> Tuple[List[int], List[int]]:
    wking = board.king(chess.WHITE)
    bking = board.king(chess.BLACK)
    if wking is None or bking is None:
        return [], []
    w_k = int(wking)
    b_k = int(bking)
    w_ids: List[int] = []
    b_ids: List[int] = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        pt = PT_MAP.get((piece.color, piece.piece_type))
        if pt is None:
            continue
        psq = int(sq)
        w_ids.append(halfkp_id(w_k, psq, pt))
        b_ids.append(halfkp_id(b_k, psq, pt))
    return w_ids, b_ids


def extract_halfkp_perspective(
    board: chess.Board, stm: int
) -> Tuple[List[int], List[int]]:
    """HalfKP features in side-to-move perspective.

    When stm == 1 (white), uses the board as-is.
    When stm == -1 (black), mirror squares and swap colors so that the
    side to move becomes "white" in feature space.
    """
    wking = board.king(chess.WHITE)
    bking = board.king(chess.BLACK)
    if wking is None or bking is None:
        return [], []

    if stm == 1:
        w_k = int(wking)
        b_k = int(bking)
    else:
        w_k = chess.square_mirror(int(bking))
        b_k = chess.square_mirror(int(wking))

    w_ids: List[int] = []
    b_ids: List[int] = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        if stm == 1:
            color = piece.color
            psq = int(sq)
        else:
            color = not piece.color
            psq = chess.square_mirror(int(sq))
        pt = PT_MAP.get((color, piece.piece_type))
        if pt is None:
            continue
        if color == chess.WHITE:
            w_ids.append(halfkp_id(w_k, psq, pt))
        else:
            b_ids.append(halfkp_id(b_k, psq, pt))
    return w_ids, b_ids


@dataclass
class FenCpRow:
    w_ids: List[int]
    b_ids: List[int]
    cp: float
    stm: int  # +1 for white to move, -1 for black


class FenCpDataset(Dataset):
    def __init__(self, rows: Sequence[FenCpRow]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> FenCpRow:
        return self.rows[idx]


def collate(batch: List[FenCpRow]):
    w_all: List[int] = []
    b_all: List[int] = []
    w_offsets = [0]
    b_offsets = [0]
    cps: List[float] = []
    stms: List[int] = []
    for row in batch:
        w_all.extend(row.w_ids)
        b_all.extend(row.b_ids)
        w_offsets.append(w_offsets[-1] + len(row.w_ids))
        b_offsets.append(b_offsets[-1] + len(row.b_ids))
        cps.append(row.cp)
        stms.append(row.stm)
    w_ids = torch.tensor(w_all, dtype=torch.long)
    b_ids = torch.tensor(b_all, dtype=torch.long)
    w_offsets = torch.tensor(w_offsets[:-1], dtype=torch.long)
    b_offsets = torch.tensor(b_offsets[:-1], dtype=torch.long)
    targets = torch.tensor(cps, dtype=torch.float32)
    stm_tensor = torch.tensor(stms, dtype=torch.float32)
    return w_ids, w_offsets, b_ids, b_offsets, stm_tensor, targets


def pearson_corr(preds: torch.Tensor, target: torch.Tensor) -> float:
    x = preds - preds.mean()
    y = target - target.mean()
    denom = torch.sqrt((x * x).mean()) * torch.sqrt((y * y).mean())
    denom_val = float(denom)
    if denom_val == 0.0:
        return float("nan")
    return float((x * y).mean() / denom_val)


def sign_agreement(preds: torch.Tensor, target: torch.Tensor) -> float:
    if preds.numel() == 0:
        return float("nan")
    return float((torch.sign(preds) == torch.sign(target)).float().mean())


def spearman_corr(preds: torch.Tensor, target: torch.Tensor) -> float:
    x_rank = preds.argsort().argsort().float()
    y_rank = target.argsort().argsort().float()
    return pearson_corr(x_rank, y_rank)


def gather_tensors_across_ranks(t: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return t
    gather_list: List[Optional[torch.Tensor]] = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, t.cpu())
    tensors: List[torch.Tensor] = [x for x in gather_list if x is not None]
    return torch.cat(tensors, dim=0) if tensors else torch.empty(0)


def setup_distributed() -> Tuple[bool, int, int, int]:
    if not dist.is_available():
        return False, 0, 1, 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    return True, rank, world_size, local_rank


class ClippedReLU(nn.Module):
    def __init__(self, cap: float = 32.0):
        super().__init__()
        self.cap = cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.relu(x), max=self.cap)


class HalfKPNNUE(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        hidden: int = 64,
        cap: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb = nn.Embedding(NUM_FEATURES, embed_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.act = ClippedReLU(cap)
        self.fc1 = nn.Linear(embed_dim * 2 + 1, hidden)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        w_ids: torch.Tensor,
        w_offsets: torch.Tensor,
        b_ids: torch.Tensor,
        b_offsets: torch.Tensor,
        stm: torch.Tensor,
    ) -> torch.Tensor:
        # Per-bag sums via embedding_bag (sum)
        w_sum = F.embedding_bag(
            w_ids, self.emb.weight, w_offsets, mode="sum", include_last_offset=False
        )
        b_sum = F.embedding_bag(
            b_ids, self.emb.weight, b_offsets, mode="sum", include_last_offset=False
        )
        w_act = self.act(w_sum)
        b_act = self.act(b_sum)
        stm_feat = stm.unsqueeze(1)
        x = torch.cat([w_act, b_act, stm_feat], dim=1)
        x = self.act(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        out = self.fc2(x)
        return out.squeeze(1)


def load_rows_csv(
    path: Path,
    fen_col: str,
    eval_col: str,
    clamp_cp: int,
    pov: str,
    min_ply: int,
    max_ply: int,
    sample_fraction: float,
    seed: int,
    max_rows: int,
) -> List[FenCpRow]:
    rng = torch.Generator().manual_seed(seed)
    rows: List[FenCpRow] = []
    for df in pd.read_csv(path, chunksize=200_000):
        if fen_col not in df.columns or eval_col not in df.columns:
            raise ValueError("Missing fen or eval column")
        if "ply" in df.columns:
            if min_ply:
                df = df[df["ply"] >= min_ply]
            if max_ply:
                df = df[df["ply"] <= max_ply]
        if sample_fraction < 1.0:
            mask = torch.rand(len(df), generator=rng) < sample_fraction
            df = df.loc[mask.numpy()]
        if "side_to_move" in df.columns:
            stm_series = (
                df["side_to_move"]
                .astype(str)
                .str.lower()
                .str.startswith("w")
                .map({True: 1, False: -1})
            )
        else:
            stm_series = pd.Series(1, index=df.index)

        if pov == "side":
            evals = (df[eval_col].astype(float) * stm_series).tolist()
        else:
            evals = df[eval_col].astype(float).tolist()
        stms = stm_series.tolist()
        fens = df[fen_col].tolist()
        for fen, cp, stm_val in zip(fens, evals, stms):
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            cp = float(max(-clamp_cp, min(clamp_cp, cp)))
            w_ids, b_ids = extract_halfkp_perspective(board, stm=int(stm_val))
            if not w_ids and not b_ids:
                continue
            rows.append(FenCpRow(w_ids=w_ids, b_ids=b_ids, cp=cp, stm=int(stm_val)))
            if max_rows > 0 and len(rows) >= max_rows:
                return rows
    return rows


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    delta: float,
    is_main: bool,
    world_size: int,
    train_sampler: Optional[torch.utils.data.Sampler] = None,
    val_sampler: Optional[torch.utils.data.Sampler] = None,
) -> Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
    use_cuda = device.type == "cuda"
    use_distributed = train_sampler is not None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=delta)

    grad_scaler_ctor = getattr(torch_amp, "GradScaler", torch.cuda.amp.GradScaler)
    try:
        scaler = grad_scaler_ctor("cuda", enabled=use_cuda)  # type: ignore[call-arg]
    except TypeError:
        scaler = grad_scaler_ctor(enabled=use_cuda)
    autocast = getattr(torch_amp, "autocast", torch.cuda.amp.autocast)

    def run_epoch(
        loader: DataLoader, train: bool
    ) -> Tuple[
        float,
        float,
        Optional[float],
        Optional[float],
        Optional[Tuple[float, float, float, float]],
        Optional[float],
    ]:
        model.train(train)
        total_loss = 0.0
        total_mae = 0.0
        n = 0
        preds_all: List[torch.Tensor] = []
        targets_all: List[torch.Tensor] = []
        for w_ids, w_offs, b_ids, b_offs, stm, y in loader:
            w_ids = w_ids.to(device)
            w_offs = w_offs.to(device)
            b_ids = b_ids.to(device)
            b_offs = b_offs.to(device)
            stm = stm.to(device)
            y = y.to(device)
            ctx = autocast("cuda") if use_cuda else contextlib.nullcontext()  # type: ignore[arg-type]
            with ctx:
                pred = model(w_ids, w_offs, b_ids, b_offs, stm)
                loss = loss_fn(pred, y)
            if train:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total_loss += loss.item() * y.size(0)
            total_mae += torch.mean(torch.abs(pred - y)).item() * y.size(0)
            n += y.size(0)
            if not train:
                preds_all.append(pred.detach().cpu())
                targets_all.append(y.detach().cpu())

        pearson: Optional[float] = None
        spearman: Optional[float] = None
        stats: Optional[Tuple[float, float, float, float]] = None
        sign_match: Optional[float] = None
        if not train:
            local_preds = torch.cat(preds_all) if preds_all else torch.empty(0)
            local_targets = torch.cat(targets_all) if targets_all else torch.empty(0)
            preds = (
                gather_tensors_across_ranks(local_preds, world_size)
                if use_distributed
                else local_preds
            )
            targets = (
                gather_tensors_across_ranks(local_targets, world_size)
                if use_distributed
                else local_targets
            )
            pearson = pearson_corr(preds, targets)
            spearman = spearman_corr(preds, targets)
            stats = (
                float(preds.mean()),
                float(preds.std()),
                float(targets.mean()),
                float(targets.std()),
            )
            sign_match = sign_agreement(preds, targets)

        return total_loss / n, total_mae / n, pearson, spearman, stats, sign_match

    last_val: Tuple[float, float, Optional[float], Optional[float], Optional[float]] = (
        float("nan"),
        float("nan"),
        None,
        None,
        None,
    )

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # type: ignore[call-arg]
        tr_loss, tr_mae, _, _, _, _ = run_epoch(train_loader, train=True)
        if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)  # type: ignore[call-arg]
        val_loss, val_mae, val_r, val_s, stats, sign_match = run_epoch(
            val_loader, train=False
        )
        last_val = (val_loss, val_mae, val_r, val_s, sign_match)
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / epochs))
        corr_tail = ""
        if val_r is not None:
            corr_tail += (
                f" val_r={val_r:.3f}" if not math.isnan(val_r) else " val_r=nan"
            )
        if val_s is not None:
            corr_tail += (
                f" val_s={val_s:.3f}" if not math.isnan(val_s) else " val_s=nan"
            )
        if sign_match is not None:
            corr_tail += (
                f" val_sign={sign_match:.3f}"
                if not math.isnan(sign_match)
                else " val_sign=nan"
            )
        if is_main:
            print(
                f"epoch {epoch + 1}: train_huber={tr_loss:.3f} train_mae={tr_mae:.1f} "
                f"val_huber={val_loss:.3f} val_mae={val_mae:.1f}{corr_tail}"
            )
            if stats is not None:
                p_mean, p_std, t_mean, t_std = stats
                # Minimal stats line to debug correlation/sign issues
                print(
                    f"           val_stats pred_mean={p_mean:.1f} pred_std={p_std:.1f} "
                    f"tgt_mean={t_mean:.1f} tgt_std={t_std:.1f}"
                )

    if use_distributed:
        dist.barrier()
    return last_val


def main() -> None:
    ap = argparse.ArgumentParser(description="HalfKP NNUE trainer (offline)")
    ap.add_argument("--data", required=True, help="CSV with fen/eval columns")
    ap.add_argument("--fen-col", default="fen")
    ap.add_argument("--eval-col", default="stockfish_cp")
    ap.add_argument("--pov", choices=["white", "side"], default="side")
    ap.add_argument("--min-ply", type=int, default=0)
    ap.add_argument("--max-ply", type=int, default=0)
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--clamp-cp", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=64.0, help="Huber delta")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save", type=str, default=None, help="Path to save state_dict")
    ap.add_argument("--optuna-trials", type=int, default=0)
    ap.add_argument("--optuna-epochs", type=int, default=8)
    ap.add_argument("--optuna-sample-fraction", type=float, default=0.5)
    args = ap.parse_args()
    use_dist, rank, world_size, local_rank = setup_distributed()
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    is_main = not use_dist or rank == 0
    torch.manual_seed(args.seed)
    if is_main:
        print(
            f"Using device: {device} (distributed={use_dist}, world_size={world_size})"
        )

    rows = load_rows_csv(
        path=Path(args.data),
        fen_col=args.fen_col,
        eval_col=args.eval_col,
        clamp_cp=args.clamp_cp,
        pov=args.pov,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    if is_main:
        print(f"Loaded {len(rows)} rows")
    if len(rows) < 1000:
        raise SystemExit("Not enough rows after filtering")

    if args.optuna_trials > 0:
        if use_dist and world_size > 1:
            raise SystemExit("Optuna search is only supported in single-process mode")
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - optuna optional
            raise SystemExit("Install optuna to use --optuna-trials") from exc

        def objective(trial: "optuna.Trial") -> float:
            trial_seed = args.seed + trial.number
            embed = trial.suggest_categorical("embed", [192, 256, 320, 384])
            hidden = trial.suggest_categorical("hidden", [128, 160, 192, 224, 256])
            dropout = trial.suggest_float("dropout", 0.0, 0.1)
            lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            take_n = max(1000, int(len(rows) * args.optuna_sample_fraction))
            perm = torch.randperm(
                len(rows), generator=torch.Generator().manual_seed(trial_seed)
            )
            subset_idx = perm[:take_n].tolist()
            subset_rows = [rows[i] for i in subset_idx]

            ds = FenCpDataset(subset_rows)
            val_len = max(1, int(len(ds) * args.val_split))
            train_len = len(ds) - val_len
            tgen = torch.Generator().manual_seed(trial_seed)
            tds, vds = random_split(ds, [train_len, val_len], generator=tgen)

            tr_loader = DataLoader(
                tds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate,
                pin_memory=device.type == "cuda",
            )
            va_loader = DataLoader(
                vds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate,
                pin_memory=device.type == "cuda",
            )

            model = HalfKPNNUE(embed_dim=embed, hidden=hidden, dropout=dropout)
            model.to(device)
            _, val_mae, _, _, _ = train_one(
                model=model,
                train_loader=tr_loader,
                val_loader=va_loader,
                device=device,
                epochs=args.optuna_epochs,
                lr=lr,
                weight_decay=weight_decay,
                delta=args.delta,
                is_main=True,
                world_size=1,
            )
            return val_mae

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.optuna_trials)
        if is_main:
            print(
                "Best trial:",
                study.best_trial.params,
                "val_mae=",
                study.best_trial.value,
            )
        return

    base_ds = FenCpDataset(rows)
    val_len = max(1, int(len(base_ds) * args.val_split))
    train_len = len(base_ds) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(base_ds, [train_len, val_len], generator=generator)

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if use_dist:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = HalfKPNNUE(embed_dim=args.embed, hidden=args.hidden, dropout=args.dropout)
    model.to(device)
    if use_dist:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    try:
        train_one(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            delta=args.delta,
            is_main=is_main,
            world_size=world_size,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
        )
    except KeyboardInterrupt:
        print("Interrupted by user, exiting cleanly.")
        return

    if args.save:
        if use_dist:
            dist.barrier()
        if is_main:
            state = (
                model.module.state_dict()
                if isinstance(model, DDP)
                else model.state_dict()
            )
            torch.save(state, args.save)
            print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
