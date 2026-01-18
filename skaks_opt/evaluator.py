from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np

from .data import Dataset
from .params import DEFAULT_PARAMS, apply_param_updates
from .pst import apply_pst_symmetry

__all__ = ["EvalResult", "evaluate_params"]


def _coerce_numeric_types(template: Any, payload: Any) -> Any:
    """Best-effort type alignment so ints in the template stay ints."""
    if isinstance(payload, dict):
        template_dict = template if isinstance(template, dict) else {}
        return {
            key: _coerce_numeric_types(template_dict.get(key), value)
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        template_list = template if isinstance(template, list) else []
        coerced: List[Any] = []
        for idx, value in enumerate(payload):
            tmpl = template_list[idx] if idx < len(template_list) else None
            coerced.append(_coerce_numeric_types(tmpl, value))
        return coerced

    if template is None:
        if isinstance(payload, float) and math.isfinite(payload):
            rounded = float(round(payload))
            if abs(payload - rounded) < 1e-9:
                return int(rounded)
        return payload
    if isinstance(template, bool):
        return bool(payload)
    if isinstance(template, int):
        if isinstance(payload, (int, bool)):
            return int(payload)
        if isinstance(payload, float):
            return int(round(payload))
        return payload
    if isinstance(template, float):
        if isinstance(payload, (int, float)):
            return float(payload)
        return payload
    return payload


@dataclass(frozen=True)
class EvalResult:
    loss: float
    mae: float
    mse: float
    rmse: float
    error_count: int
    evaluated: int


def evaluate_params(
    param_updates: Mapping[str, int | float],
    dataset: Dataset,
    batch_size: int = 256,
    threads: int = 0,
    error_penalty: float = 2000.0,
    pov: str = "side",
    cp_cap: float | None = None,
    base_params: Mapping[str, Mapping] | None = None,
    skip_pst: bool = False,
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

    params = apply_param_updates(base_params or DEFAULT_PARAMS, param_updates)
    params = _coerce_numeric_types(DEFAULT_PARAMS, params)
    if not skip_pst:
        apply_pst_symmetry(params)

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
