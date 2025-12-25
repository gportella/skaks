from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, MutableMapping
import re

__all__ = [
    "ParamSpec",
    "DEFAULT_PARAMS",
    "default_param_space",
    "apply_param_updates",
]


@dataclass(frozen=True)
class ParamSpec:
    """Definition of a tunable parameter.

    name uses dotted segments; array entries use e.g. "evaluation.king_attack_weights[0]".
    """

    name: str
    low: int
    high: int
    step: int = 1


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
        "king_attack_weights": [14, 32, 30, 44, 74, 20, 14, 32, 30, 44, 74, 20],
        "threat_base": [0, 12, 30, 30, 45, 180, 540, 12, 30, 30, 45, 180, 540],
        "bishop_pin_penalty": {"base": 12, "mobility": 2},
        "rook_pin_penalty": {"base": 6, "mobility": 1},
        "knight_pin_penalty": {"base": 15, "mobility": 0},
        "pawn_pin_straight_penalty": {"base": 6, "mobility": 2},
        "pawn_pin_diagonal_penalty": {"base": 10, "mobility": 2},
    },
    "search": {
        "aspiration_window_initial": 800,
        "aspiration_window_max": 6400,
        "quiescence_delta_margin": 150,
        "quiescence_max_ply": 12,
        "quiescence_max_noisy_moves": 16,
        "quiescence_zero_gain_skip_index": 6,
        "null_move_reduction": 3,
        "null_move_min_depth": 4,
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
        ParamSpec("search.aspiration_window_initial", 1200, 2500, step=10),
        ParamSpec("search.aspiration_window_max", 4000, 9000, step=20),
        ParamSpec("search.quiescence_delta_margin", 240, 420, step=5),
        ParamSpec("search.quiescence_max_ply", 8, 16),
        ParamSpec("search.quiescence_max_noisy_moves", 12, 30),
        ParamSpec("search.quiescence_zero_gain_skip_index", 0, 6),
        ParamSpec("search.null_move_reduction", 2, 5),
        ParamSpec("search.null_move_min_depth", 4, 9),
    ]

    if include_arrays:
        for idx in range(len(DEFAULT_PARAMS["evaluation"]["king_attack_weights"])):
            specs.append(ParamSpec(f"evaluation.king_attack_weights[{idx}]", 0, 120))
        for idx in range(len(DEFAULT_PARAMS["evaluation"]["threat_base"])):
            specs.append(ParamSpec(f"evaluation.threat_base[{idx}]", 0, 300))

    return specs


_SEGMENT_RE = re.compile(r"^(?P<name>[a-zA-Z0-9_]+)(?:\[(?P<idx>\d+)\])?$")


def _set_nested(target: MutableMapping, dotted: str, value: int) -> None:
    parts = dotted.split(".")
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
    m = _SEGMENT_RE.match(leaf)
    if not m:
        raise ValueError(f"bad key segment: {leaf}")
    name = m.group("name")
    idx = m.group("idx")

    if idx is None:
        cur[name] = value
        return

    idx_int = int(idx)
    if name not in cur:
        raise ValueError(f"array {name} missing in base params")
    arr = list(cur[name])
    if idx_int >= len(arr):
        raise IndexError(f"index {idx_int} out of bounds for {name}")
    arr[idx_int] = value
    cur[name] = arr


def apply_param_updates(base: Dict, updates: MutableMapping[str, int]) -> Dict:
    """Return a deep-ish copy of base with flat-key updates applied."""

    merged = {
        "evaluation": {
            **base.get("evaluation", {}),
        },
        "search": {
            **base.get("search", {}),
        },
    }
    for key, val in updates.items():
        _set_nested(merged, key, int(val))
    return merged
