from __future__ import annotations

import warnings

import skaks_opt as _skaks_opt

warnings.warn(
    "tuning.skaks_opt is deprecated; use skaks_opt instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(_skaks_opt, "__all__", [])


def __getattr__(name: str):
    return getattr(_skaks_opt, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_skaks_opt)))
