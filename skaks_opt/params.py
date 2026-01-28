from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping

from skaks_opt.pst import (PIECE_ORDER, default_endgame_tables,
                           default_midgame_tables)

__all__ = [
    "ParamSpec",
    "DEFAULT_PARAMS",
    "default_param_space",
    "phase_weight_param_space",
    "pst_param_space",
    "param_space_for_mode",
    "param_set_prefixes",
    "apply_param_updates",
]


@dataclass(frozen=True)
class ParamSpec:
    """Definition of a tunable parameter.

    name uses dotted segments; array entries use e.g. "evaluation.king_attack_weights[0]".
    """

    name: str
    low: float
    high: float
    step: float | None = 1
    is_float: bool = False


DEFAULT_PARAMS: Dict[str, Dict] = {
    "evaluation": {
        "check_penalty": 100,
        "pawn_shield_bonus": 25,
        "castling_bonus": 50,
        "tempo_bonus": 14,
        "threat_weight": 4,
        "passed_pawn_base": 20,
        "passed_pawn_advance": 8,
        "hanging_divisor": 4,
        "hanging_min_penalty": 18,
        "king_ring_base": 6,
        "king_ring_defended_scale": 2,
        "king_ring_enemy_occupier": 14,
        "king_ring_enemy_piece_material_scale": 14,
        "bishop_pair_bonus": 28,
        "rook_open_file_bonus": 34,
        "rook_semi_open_file_bonus": 18,
        "mobility_scaling": 15,
        "knight_dev_bonus": 12,
        "bishop_dev_bonus": 10,
        "connect_rooks_bonus": 10,
        "central_pawn_bonus": 12,
        "castle_urgency": 20,
        "early_queen_penalty": 16,
        "flank_pawn_penalty": 8,
        "knight_mobility_scale": 4,
        "bishop_mobility_scale": 5,
        "rook_mobility_scale": 3,
        "queen_mobility_scale": 1,
        "doubled_pawn_penalty": 12,
        "isolated_pawn_penalty": 16,
        "backward_pawn_penalty": 10,
        "king_attack_weights": [14, 32, 30, 44, 74, 20, 14, 32, 30, 44, 74, 20],
        "threat_base": [0, 12, 30, 30, 45, 180, 540, 12, 30, 30, 45, 180, 540],
        "eval_quiet_cap": 2200,
        "bishop_pin_penalty": {"base": 12, "mobility": 2},
        "rook_pin_penalty": {"base": 6, "mobility": 1},
        "knight_pin_penalty": {"base": 15, "mobility": 0},
        "pawn_pin_straight_penalty": {"base": 6, "mobility": 2},
        "pawn_pin_diagonal_penalty": {"base": 10, "mobility": 2},
        "phase_weights_mg": [1.0] * 17,
        "phase_weights_eg": [1.0] * 17,
        "pst_midgame": default_midgame_tables(),
        "pst_endgame": default_endgame_tables(),
    },
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
        # Eval scalars narrowed around current best
        ParamSpec("evaluation.check_penalty", 12, 40),
        ParamSpec("evaluation.pawn_shield_bonus", 0, 24),
        ParamSpec("evaluation.castling_bonus", 10, 70),
        ParamSpec("evaluation.tempo_bonus", 0, 20),
        ParamSpec("evaluation.threat_weight", 4, 14),
        ParamSpec("evaluation.passed_pawn_base", 0, 24),
        ParamSpec("evaluation.passed_pawn_advance", 0, 24),
        ParamSpec("evaluation.hanging_divisor", 2, 8),
        ParamSpec("evaluation.hanging_min_penalty", 16, 48),
        ParamSpec("evaluation.king_ring_base", 0, 14),
        ParamSpec("evaluation.king_ring_defended_scale", 2, 10),
        ParamSpec("evaluation.king_ring_enemy_occupier", 4, 30),
        ParamSpec("evaluation.king_ring_enemy_piece_material_scale", 16, 48),
        ParamSpec("evaluation.bishop_pair_bonus", 40, 90),
        ParamSpec("evaluation.rook_open_file_bonus", 30, 90),
        ParamSpec("evaluation.rook_semi_open_file_bonus", 20, 80),
        ParamSpec("evaluation.mobility_scaling", 12, 34),
        ParamSpec("evaluation.knight_dev_bonus", 20, 44),
        ParamSpec("evaluation.bishop_dev_bonus", 18, 44),
        ParamSpec("evaluation.connect_rooks_bonus", 20, 48),
        ParamSpec("evaluation.central_pawn_bonus", 8, 28),
        ParamSpec("evaluation.castle_urgency", 40, 70),
        ParamSpec("evaluation.early_queen_penalty", 0, 24),
        ParamSpec("evaluation.flank_pawn_penalty", 16, 32),
        ParamSpec("evaluation.knight_mobility_scale", 0, 16),
        ParamSpec("evaluation.bishop_mobility_scale", 0, 20),
        ParamSpec("evaluation.rook_mobility_scale", 0, 16),
        ParamSpec("evaluation.queen_mobility_scale", 0, 12),
        ParamSpec("evaluation.doubled_pawn_penalty", 0, 32),
        ParamSpec("evaluation.isolated_pawn_penalty", 0, 40),
        ParamSpec("evaluation.backward_pawn_penalty", 0, 32),
        ParamSpec("evaluation.eval_quiet_cap", 2200, 2500, step=5),
        # Pin penalties narrowed
        ParamSpec("evaluation.bishop_pin_penalty.base", 10, 28),
        ParamSpec("evaluation.bishop_pin_penalty.mobility", 0, 6),
        ParamSpec("evaluation.rook_pin_penalty.base", 10, 30),
        ParamSpec("evaluation.rook_pin_penalty.mobility", 0, 5),
        ParamSpec("evaluation.knight_pin_penalty.base", 8, 26),
        ParamSpec("evaluation.knight_pin_penalty.mobility", 0, 6),
        ParamSpec("evaluation.pawn_pin_straight_penalty.base", 12, 28),
        ParamSpec("evaluation.pawn_pin_straight_penalty.mobility", 4, 10),
        ParamSpec("evaluation.pawn_pin_diagonal_penalty.base", 14, 30),
        ParamSpec("evaluation.pawn_pin_diagonal_penalty.mobility", 4, 10),
        # Search narrowed
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
        ParamSpec("search.lmr_history_divisor", 2000.0, 20000.0, step=200.0, is_float=True),
        ParamSpec("search.lmr_pv_offset", 0.0, 2.5, step=0.05, is_float=True),
        ParamSpec("search.futility_margins[0]", 0, 0, step=1),
        ParamSpec("search.futility_margins[1]", 0, 600, step=10),
        ParamSpec("search.futility_margins[2]", 0, 800, step=10),
        ParamSpec("search.futility_margins[3]", 0, 1000, step=10),
        # Search NNUE (mirrors search ranges)
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
        ParamSpec("search_nnue.lmr_history_divisor", 2000.0, 20000.0, step=200.0, is_float=True),
        ParamSpec("search_nnue.lmr_pv_offset", 0.0, 2.5, step=0.05, is_float=True),
        ParamSpec("search_nnue.futility_margins[0]", 0, 0, step=1),
        ParamSpec("search_nnue.futility_margins[1]", 0, 600, step=10),
        ParamSpec("search_nnue.futility_margins[2]", 0, 800, step=10),
        ParamSpec("search_nnue.futility_margins[3]", 0, 1000, step=10),
    ]

    if include_arrays:
        for idx in range(len(DEFAULT_PARAMS["evaluation"]["king_attack_weights"])):
            specs.append(ParamSpec(f"evaluation.king_attack_weights[{idx}]", 0, 120))
        specs.extend(_phase_weight_specs())

    return specs


def _phase_weight_specs() -> List[ParamSpec]:
    specs: List[ParamSpec] = []
    for idx in range(len(DEFAULT_PARAMS["evaluation"]["phase_weights_mg"])):
        specs.append(
            ParamSpec(
                f"evaluation.phase_weights_mg[{idx}]",
                -1.0,
                1.0,
                step=0.05,
                is_float=True,
            )
        )
    for idx in range(len(DEFAULT_PARAMS["evaluation"]["phase_weights_eg"])):
        specs.append(
            ParamSpec(
                f"evaluation.phase_weights_eg[{idx}]",
                -1.0,
                1.0,
                step=0.05,
                is_float=True,
            )
        )
    return specs


def _clamp_phase_weights(eval_section: MutableMapping, low: float = -1.0, high: float = 1.0) -> None:
    for key in ("phase_weights_mg", "phase_weights_eg"):
        arr = eval_section.get(key)
        if isinstance(arr, tuple):
            arr = list(arr)
        if isinstance(arr, list):
            eval_section[key] = [
                float(max(low, min(high, float(val))))
                for val in arr
            ]


def phase_weight_param_space() -> List[ParamSpec]:
    """Only tune midgame/endgame phase weights."""

    return _phase_weight_specs()


def _pst_param_specs(low: int = -200, high: int = 200) -> List[ParamSpec]:
    specs: List[ParamSpec] = []
    for block in ("pst_midgame", "pst_endgame"):
        for piece in PIECE_ORDER:
            for rank in range(8):
                for file in range(8):
                    specs.append(
                        ParamSpec(
                            f"evaluation.{block}.{piece}[{rank}][{file}]",
                            low,
                            high,
                        )
                    )
    return specs


def pst_param_space(low: int = -200, high: int = 200) -> List[ParamSpec]:
    """Tune every entry of the PST tables."""

    return _pst_param_specs(low=low, high=high)


# Parameter subsets for narrower Texel searches
_OFFENSE_KEYS = {
    "evaluation.check_penalty",
    "evaluation.tempo_bonus",
    "evaluation.threat_weight",
    "evaluation.king_ring_base",
    "evaluation.king_ring_defended_scale",
    "evaluation.king_ring_enemy_occupier",
    "evaluation.king_ring_enemy_piece_material_scale",
    "evaluation.bishop_pair_bonus",
    "evaluation.rook_open_file_bonus",
    "evaluation.rook_semi_open_file_bonus",
    "evaluation.mobility_scaling",
    "evaluation.knight_mobility_scale",
    "evaluation.bishop_mobility_scale",
    "evaluation.rook_mobility_scale",
    "evaluation.queen_mobility_scale",
    "evaluation.passed_pawn_base",
    "evaluation.passed_pawn_advance",
    "evaluation.hanging_divisor",
    "evaluation.hanging_min_penalty",
    "evaluation.castle_urgency",
    "evaluation.central_pawn_bonus",
    "evaluation.connect_rooks_bonus",
    "evaluation.king_attack_weights",
    "evaluation.threat_base",
    "evaluation.doubled_pawn_penalty",
    "evaluation.isolated_pawn_penalty",
    "evaluation.backward_pawn_penalty",
}

_DEFENSE_KEYS = {
    "evaluation.pawn_shield_bonus",
    "evaluation.castling_bonus",
    "evaluation.flank_pawn_penalty",
    "evaluation.doubled_pawn_penalty",
    "evaluation.isolated_pawn_penalty",
    "evaluation.backward_pawn_penalty",
    "evaluation.king_ring_base",
    "evaluation.king_ring_defended_scale",
    "evaluation.king_ring_enemy_occupier",
    "evaluation.king_ring_enemy_piece_material_scale",
    "evaluation.early_queen_penalty",
    "evaluation.bishop_pin_penalty",
    "evaluation.rook_pin_penalty",
    "evaluation.knight_pin_penalty",
    "evaluation.pawn_pin_straight_penalty",
    "evaluation.pawn_pin_diagonal_penalty",
    "evaluation.king_attack_weights",
    "evaluation.threat_base",
}


def _base_name(spec: ParamSpec) -> str:
    return spec.name.split("[")[0]


def param_space_for_mode(
    mode: str = "full", *, include_arrays: bool = False, include_pst: bool = True
) -> List[ParamSpec]:
    """Return a filtered parameter space.

    mode: full | phase | offense | defense | pst
    include_arrays: include king_attack_weights/threat_base when available.
    """

    if mode == "pst":
        return pst_param_space()
    if mode == "phase":
        return phase_weight_param_space()

    specs = default_param_space(include_arrays=include_arrays)

    if not include_pst:
        specs = [
            spec
            for spec in specs
            if not _base_name(spec).startswith("evaluation.pst_")
        ]

    if mode == "full":
        return specs

    allowed = _OFFENSE_KEYS if mode == "offense" else _DEFENSE_KEYS
    filtered: List[ParamSpec] = []
    for spec in specs:
        base = _base_name(spec)
        if base in allowed:
            filtered.append(spec)
    return filtered


def param_set_prefixes(mode: str = "full") -> List[str]:
    """Return dotted-prefix filters for param-optimize param sets."""

    if mode == "phase":
        return ["evaluation.phase_weights_mg", "evaluation.phase_weights_eg"]
    if mode == "pst":
        return ["evaluation.pst_midgame", "evaluation.pst_endgame"]
    if mode == "offense":
        return sorted(_OFFENSE_KEYS)
    if mode == "defense":
        return sorted(_DEFENSE_KEYS)
    # full defaults to all evaluation params (exclude search)
    return ["evaluation."]


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

    base_eval = deepcopy(DEFAULT_PARAMS["evaluation"])
    for key, val in base.get("evaluation", {}).items():
        if isinstance(val, dict) and isinstance(base_eval.get(key), dict):
            base_eval[key] = {**base_eval[key], **val}
        else:
            base_eval[key] = val

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
        "evaluation": base_eval,
        "search": base_search,
        "search_nnue": base_search_nnue,
    }
    for key, val in updates.items():
        _set_nested(merged, key, val)
    _clamp_phase_weights(merged["evaluation"])
    return merged
