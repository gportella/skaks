from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping

__all__ = [
    "ParamSpec",
    "DEFAULT_PARAMS",
    "default_param_space",
    "param_space_for_mode",
    "param_set_prefixes",
    "apply_param_updates",
]


@dataclass(frozen=True)
class ParamSpec:
    """Definition of a tunable parameter.

    name uses dotted segments; array entries use e.g. "search.futility_margins[1]".
    """

    name: str
    low: float
    high: float
    step: float | None = 1
    is_float: bool = False


DEFAULT_PARAMS: Dict[str, Dict] = {
    "search": {
        "aspiration_window_initial": 800,
        "aspiration_window_max": 6400,
        "quiescence_delta_margin": 150,
        "quiescence_max_ply": 12,
        "quiescence_max_noisy_moves": 16,
        "quiescence_zero_gain_skip_index": 6,
        "quiescence_max_quiet_checks": 6,
        "null_move_reduction": 3,
        "null_move_reduction_divisor": 6,
        "null_move_min_depth": 4,
        "lmr_intercept": 0.0,
        "lmr_divisor": 1.1764705882352942,
        "lmr_history_divisor": 8000.0,
        "lmr_pv_offset": 1.0,
        "futility_margins": [0, 120, 240, 400],
    },
    "search_nnue": {
        "aspiration_window_initial": 100,
        "aspiration_window_max": 1200,
        "quiescence_delta_margin": 150,
        "quiescence_max_ply": 12,
        "quiescence_max_noisy_moves": 16,
        "quiescence_zero_gain_skip_index": 6,
        "quiescence_max_quiet_checks": 6,
        "null_move_reduction": 3,
        "null_move_reduction_divisor": 6,
        "null_move_min_depth": 4,
        "lmr_intercept": 0.0,
        "lmr_divisor": 1.1764705882352942,
        "lmr_history_divisor": 8000.0,
        "lmr_pv_offset": 1.0,
        "futility_margins": [0, 120, 240, 400],
    },
}


def default_param_space(include_arrays: bool = False) -> List[ParamSpec]:
    """Reasonable starting space around defaults.

    Arrays are optional because they greatly increase dimensionality.
    """
    specs: List[ParamSpec] = [
        ParamSpec("search.aspiration_window_initial", 50, 200, step=10),
        ParamSpec("search.aspiration_window_max", 600, 2400, step=20),
        ParamSpec("search.quiescence_delta_margin", 200, 420, step=5),
        ParamSpec("search.quiescence_max_ply", 8, 16),
        ParamSpec("search.quiescence_max_noisy_moves", 12, 30),
        ParamSpec("search.quiescence_zero_gain_skip_index", 0, 6),
        ParamSpec("search.quiescence_max_quiet_checks", 0, 12),
        ParamSpec("search.null_move_reduction", 2, 5),
        ParamSpec("search.null_move_reduction_divisor", 4, 10),
        ParamSpec("search.null_move_min_depth", 4, 9),
        ParamSpec("search.lmr_intercept", -0.5, 1.5, step=0.05, is_float=True),
        ParamSpec("search.lmr_divisor", 0.6, 2.5, step=0.05, is_float=True),
        ParamSpec(
            "search.lmr_history_divisor",
            2000.0,
            20000.0,
            step=200.0,
            is_float=True,
        ),
        ParamSpec("search.lmr_pv_offset", 0.0, 2.5, step=0.05, is_float=True),
        ParamSpec("search.futility_margins[0]", 0, 0, step=1),
        ParamSpec("search.futility_margins[1]", 0, 600, step=10),
        ParamSpec("search.futility_margins[2]", 0, 800, step=10),
        ParamSpec("search.futility_margins[3]", 0, 1000, step=10),
        ParamSpec("search_nnue.aspiration_window_initial", 50, 200, step=10),
        ParamSpec("search_nnue.aspiration_window_max", 600, 2400, step=20),
        ParamSpec("search_nnue.quiescence_delta_margin", 200, 420, step=5),
        ParamSpec("search_nnue.quiescence_max_ply", 8, 16),
        ParamSpec("search_nnue.quiescence_max_noisy_moves", 12, 30),
        ParamSpec("search_nnue.quiescence_zero_gain_skip_index", 0, 6),
        ParamSpec("search_nnue.quiescence_max_quiet_checks", 0, 12),
        ParamSpec("search_nnue.null_move_reduction", 2, 5),
        ParamSpec("search_nnue.null_move_reduction_divisor", 4, 10),
        ParamSpec("search_nnue.null_move_min_depth", 4, 9),
        ParamSpec("search_nnue.lmr_intercept", -0.5, 1.5, step=0.05, is_float=True),
        ParamSpec("search_nnue.lmr_divisor", 0.6, 2.5, step=0.05, is_float=True),
        ParamSpec(
            "search_nnue.lmr_history_divisor",
            2000.0,
            20000.0,
            step=200.0,
            is_float=True,
        ),
        ParamSpec("search_nnue.lmr_pv_offset", 0.0, 2.5, step=0.05, is_float=True),
        ParamSpec("search_nnue.futility_margins[0]", 0, 0, step=1),
        ParamSpec("search_nnue.futility_margins[1]", 0, 600, step=10),
        ParamSpec("search_nnue.futility_margins[2]", 0, 800, step=10),
        ParamSpec("search_nnue.futility_margins[3]", 0, 1000, step=10),
    ]

    return specs


def param_space_for_mode(
    mode: str = "both", *, include_arrays: bool = False
) -> List[ParamSpec]:
    """Return a filtered parameter space.

    mode: both | search | search_nnue
    """

    specs = default_param_space(include_arrays=include_arrays)
    if mode == "search":
        return [spec for spec in specs if spec.name.startswith("search.")]
    if mode == "search_nnue":
        return [spec for spec in specs if spec.name.startswith("search_nnue.")]
    return specs


def param_set_prefixes(mode: str = "both") -> List[str]:
    """Return dotted-prefix filters for param-optimize param sets."""

    if mode == "search":
        return ["search."]
    if mode == "search_nnue":
        return ["search_nnue."]
    return ["search.", "search_nnue."]


_SEGMENT_RE = re.compile(r"^(?P<name>[a-zA-Z0-9_]+)(?:\[(?P<idx>\d+)\])?$")
_LEAF_RE = re.compile(r"^(?P<name>[a-zA-Z0-9_]+)(?P<indices>(\[\d+\])*)$")


def _parse_leaf_token(token: str) -> tuple[str, List[int]]:
    m = _LEAF_RE.match(token)
    if not m:
        raise ValueError(f"bad key segment: {token}")
    indices_str = m.group("indices") or ""
    indices = [int(idx) for idx in re.findall(r"\[(\d+)\]", indices_str)]
    return m.group("name"), indices


def _assign_nested_list(container: Any, indices: List[int], value: int | float, name: str) -> None:
    ref = container
    for depth, idx in enumerate(indices):
        if not isinstance(ref, list):
            raise TypeError(f"array '{name}' is not indexable")
        if idx >= len(ref):
            raise IndexError(f"index {idx} out of bounds for '{name}'")
        if depth == len(indices) - 1:
            ref[idx] = value
            return
        if isinstance(ref[idx], tuple):
            ref[idx] = list(ref[idx])
        ref = ref[idx]


def _set_nested(target: MutableMapping, dotted: str, value: int | float) -> None:
    parts = dotted.split(".")
    section = parts[0]
    cur: MutableMapping = target
    for part in parts[:-1]:
        m = _SEGMENT_RE.match(part)
        if not m:
            raise ValueError(f"bad key segment: {part}")
        name = m.group("name")
        idx = m.group("idx")
        if idx is not None:
            raise ValueError("array indices only allowed at leaf")
        if name not in cur:
            cur[name] = {}
        cur = cur[name]

    leaf = parts[-1]
    name, indices = _parse_leaf_token(leaf)

    if not indices:
        cur[name] = value
        return

    if name not in cur:
        raise ValueError(f"array {name} missing in base params")
    if isinstance(cur[name], tuple):
        cur[name] = list(cur[name])
    _assign_nested_list(cur[name], indices, value, name)


def apply_param_updates(base: Dict, updates: MutableMapping[str, int | float]) -> Dict:
    """Return a deep-ish copy of base with flat-key updates applied."""
    base_search = deepcopy(DEFAULT_PARAMS["search"])
    for key, val in base.get("search", {}).items():
        if isinstance(val, dict) and isinstance(base_search.get(key), dict):
            base_search[key] = {**base_search[key], **val}
        else:
            base_search[key] = val

    base_search_nnue = deepcopy(DEFAULT_PARAMS.get("search_nnue", base_search))
    for key, val in base.get("search_nnue", {}).items():
        if isinstance(val, dict) and isinstance(base_search_nnue.get(key), dict):
            base_search_nnue[key] = {**base_search_nnue[key], **val}
        else:
            base_search_nnue[key] = val

    merged = {
        "search": base_search,
        "search_nnue": base_search_nnue,
    }
    for key, val in updates.items():
        _set_nested(merged, key, val)
    return merged
