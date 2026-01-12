"""Utility helpers for score/Elo statistics."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, Tuple

_SCORE_EPS = 1e-9


def _clamp_score(value: float) -> float:
    if value <= _SCORE_EPS:
        return _SCORE_EPS
    if value >= 1.0 - _SCORE_EPS:
        return 1.0 - _SCORE_EPS
    return value


def score_to_elo(score: float) -> float:
    """Convert a fractional score (0-1) to Elo difference."""
    clamped = _clamp_score(score)
    return 400.0 * math.log10(clamped / (1.0 - clamped))


def confidence_to_z(confidence: float) -> float:
    """Return the Z-score for a two-tailed normal confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf((1.0 + confidence) / 2.0)


def score_standard_error(wins: int, losses: int, draws: int) -> float:
    total = wins + losses + draws
    if total <= 0:
        return 0.0
    score = (wins + 0.5 * draws) / total
    variance = max(score * (1.0 - score), 0.0)
    return math.sqrt(variance / total)


def score_confidence_interval(
    wins: int,
    losses: int,
    draws: int,
    *,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total > 0 else 0.0
    if total <= 0 or not 0.0 < confidence < 1.0:
        return score, score
    z_value = confidence_to_z(confidence)
    margin = z_value * score_standard_error(wins, losses, draws)
    return max(0.0, score - margin), min(1.0, score + margin)


def summarize_wld(
    wins: int,
    losses: int,
    draws: int,
    *,
    confidence: float = 0.95,
) -> Dict[str, float]:
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total > 0 else 0.0
    stderr = score_standard_error(wins, losses, draws)
    if total > 0 and 0.0 < confidence < 1.0:
        z_value = confidence_to_z(confidence)
        margin = z_value * stderr
        ci_low = max(0.0, score - margin)
        ci_high = min(1.0, score + margin)
    else:
        z_value = 0.0
        margin = 0.0
        ci_low = score
        ci_high = score
    return {
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "games": int(total),
        "score": float(score),
        "stderr": float(stderr),
        "margin": float(margin),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": float(confidence if 0.0 < confidence < 1.0 else 0.0),
        "z_value": float(z_value),
    }


__all__ = [
    "score_to_elo",
    "score_standard_error",
    "score_confidence_interval",
    "confidence_to_z",
    "summarize_wld",
]
