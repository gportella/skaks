from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .data import Dataset
from .params import DEFAULT_PARAMS, apply_param_updates

__all__ = ["EvalResult", "evaluate_params"]


@dataclass(frozen=True)
class EvalResult:
    loss: float
    mae: float
    mse: float
    rmse: float
    error_count: int
    evaluated: int


def evaluate_params(
    param_updates: Dict[str, int],
    dataset: Dataset,
    batch_size: int = 256,
    threads: int = 0,
    error_penalty: float = 2000.0,
    pov: str = "side",
    cp_cap: float | None = None,
) -> EvalResult:
    """Compute weighted MAE against target scores.

    error_penalty is applied (once) for any FEN that fails to evaluate.
    """

    try:
        import skaks_eval as sk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "skaks_eval is not installed; install bindings first"
        ) from exc

    params = apply_param_updates(DEFAULT_PARAMS, param_updates)

    total_weight = 0.0
    total_abs_err = 0.0
    total_sq_err = 0.0
    error_count = 0

    for fens, targets, weights, side in dataset.iter_batches(batch_size):
        result = sk.eval_fens(fens, params=params, threads=threads)
        cp_raw = np.asarray(result["cp"], dtype=np.float64)
        if cp_cap is not None:
            cp_raw = np.clip(cp_raw, -cp_cap, cp_cap)
            targets = np.clip(targets, -cp_cap, cp_cap)
        if pov == "side":
            cp = cp_raw * side  # Stockfish scores are side-to-move; align ours
        else:
            cp = cp_raw
        errs = result["errors"]

        # accumulate errors
        for i, err in enumerate(errs):
            w = float(weights[i])
            if err is not None:
                total_abs_err += error_penalty * w
                total_sq_err += (error_penalty**2) * w
                total_weight += w
                error_count += 1
                continue
            diff = cp[i] - float(targets[i])
            total_abs_err += abs(diff) * w
            total_sq_err += (diff * diff) * w
            total_weight += w

    if total_weight > 0:
        mae = total_abs_err / total_weight
        mse = total_sq_err / total_weight
        rmse = float(np.sqrt(mse))
    else:
        mae = mse = rmse = float("inf")

    # loss is MAE for compatibility with existing studies
    loss = mae
    return EvalResult(
        loss=loss,
        mae=mae,
        mse=mse,
        rmse=rmse,
        error_count=error_count,
        evaluated=len(dataset),
    )
