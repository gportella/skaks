"""Replica exchange Monte Carlo utilities for self-play tuning."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class REXChain:
    params: dict
    score: float


@dataclass
class ReplicaExchangeState:
    temperatures: List[float]
    target_accept: float = 0.2
    adapt_rate: float = 0.05
    seed: Optional[int] = None
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    _accepts: List[int] = field(default_factory=list, init=False, repr=False)
    _trials: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed is not None:
            self._rng.seed(self.seed)
        n = len(self.temperatures)
        self._accepts = [0 for _ in range(n)]
        self._trials = [0 for _ in range(n)]

    def accept_prob(self, idx: int, delta: float) -> float:
        temp = max(1e-9, float(self.temperatures[idx]))
        if delta >= 0:
            return 1.0
        try:
            return math.exp(delta / temp)
        except OverflowError:
            return 0.0

    def should_accept(self, idx: int, candidate_score: float, current_score: float) -> bool:
        delta = candidate_score - current_score
        prob = self.accept_prob(idx, delta)
        self._trials[idx] += 1
        if prob >= 1.0 or self._rng.random() < prob:
            self._accepts[idx] += 1
            return True
        return False

    def acceptance_rate(self, idx: int) -> float:
        trials = self._trials[idx]
        if trials <= 0:
            return 0.0
        return float(self._accepts[idx]) / float(trials)

    def adapt_temperatures(self) -> None:
        if self.adapt_rate <= 0:
            return
        for i in range(len(self.temperatures)):
            rate = self.acceptance_rate(i)
            delta = self.target_accept - rate
            scale = math.exp(self.adapt_rate * delta)
            self.temperatures[i] = max(1e-6, float(self.temperatures[i]) * scale)

    def try_swaps(self, chains: List[REXChain]) -> int:
        swaps = 0
        for i in range(len(chains) - 1):
            a = chains[i]
            b = chains[i + 1]
            ta = max(1e-9, float(self.temperatures[i]))
            tb = max(1e-9, float(self.temperatures[i + 1]))
            delta = (a.score - b.score) * (1.0 / ta - 1.0 / tb)
            prob = 1.0 if delta >= 0 else math.exp(delta)
            if prob >= 1.0 or self._rng.random() < prob:
                chains[i], chains[i + 1] = chains[i + 1], chains[i]
                swaps += 1
        return swaps


def build_temperatures(count: int, t_min: float, t_max: float) -> List[float]:
    count = max(1, int(count))
    if count == 1:
        return [float(t_min)]
    t_min = float(max(1e-6, t_min))
    t_max = float(max(t_min, t_max))
    ratio = (t_max / t_min) ** (1.0 / (count - 1))
    return [t_min * (ratio**i) for i in range(count)]
