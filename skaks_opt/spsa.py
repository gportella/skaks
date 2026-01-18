"""SPSA optimizer utilities for noisy objective tuning."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class SPSAState:
    """Holds SPSA schedule parameters and iteration counter.

    Parameters follow the common SPSA schedule:
      a_k = a / (A + k + 1)^alpha
      c_k = c / (k + 1)^gamma
    """

    a: float = 0.01
    c: float = 0.05
    A: float = 10.0
    alpha: float = 0.602
    gamma: float = 0.101
    iteration: int = 0
    rng_seed: Optional[int] = None
    _rng_state: int = field(default=0, init=False, repr=False)
    _rng: Optional[random.Random] = field(default=None, init=False, repr=False)

    def _next_delta(self, n: int) -> List[float]:
        if self.rng_seed is not None and self._rng_state == 0:
            self._rng_state = int(self.rng_seed)
        if self.rng_seed is None and self._rng is None:
            self._rng = random.Random()

        def next_rand() -> int:
            if self.rng_seed is None:
                return self._rng.getrandbits(31)
            self._rng_state = (1103515245 * self._rng_state + 12345) & 0x7FFFFFFF
            return self._rng_state

        deltas: List[float] = []
        for _ in range(n):
            bit = next_rand() & 1
            deltas.append(1.0 if bit == 1 else -1.0)
        return deltas

    def coefficients(self) -> Tuple[float, float]:
        k = self.iteration
        a_k = self.a / (self.A + k + 1) ** self.alpha
        c_k = self.c / (k + 1) ** self.gamma
        return a_k, c_k

    def propose(self, theta: Sequence[float]) -> Tuple[List[float], List[float], List[float], float]:
        """Return (theta_plus, theta_minus, delta, c_k)."""
        a_k, c_k = self.coefficients()
        deltas = self._next_delta(len(theta))
        theta_plus = [t + c_k * d for t, d in zip(theta, deltas)]
        theta_minus = [t - c_k * d for t, d in zip(theta, deltas)]
        return theta_plus, theta_minus, deltas, c_k

    def update(
        self,
        theta: Sequence[float],
        deltas: Sequence[float],
        y_plus: float,
        y_minus: float,
        *,
        maximize: bool = False,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> List[float]:
        """Update parameters using SPSA gradient estimate.

        Args:
            theta: Current parameter vector.
            deltas: +/-1 perturbation vector used to evaluate y_plus/y_minus.
            y_plus: Objective value at theta_plus.
            y_minus: Objective value at theta_minus.
            maximize: If True, treat objective as a maximization problem.
            bounds: Optional list of (min, max) clamps for each parameter.
        """
        a_k, c_k = self.coefficients()
        # Gradient estimate: g_i = (y_plus - y_minus) / (2 * c_k * delta_i)
        scale = (y_plus - y_minus) / (2.0 * c_k)
        direction = -1.0 if maximize else 1.0
        updated = [
            t - direction * a_k * scale / d if d != 0 else t
            for t, d in zip(theta, deltas)
        ]
        if bounds is not None:
            clamped: List[float] = []
            for val, (low, high) in zip(updated, bounds):
                clamped.append(min(max(val, low), high))
            updated = clamped
        self.iteration += 1
        return updated


def spsa_step(
    state: SPSAState,
    theta: Sequence[float],
    y_plus: float,
    y_minus: float,
    deltas: Sequence[float],
    *,
    maximize: bool = False,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[float]:
    """Functional wrapper around SPSAState.update()."""
    return state.update(
        theta,
        deltas,
        y_plus,
        y_minus,
        maximize=maximize,
        bounds=bounds,
    )