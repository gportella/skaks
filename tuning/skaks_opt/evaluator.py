from __future__ import annotations

import warnings

warnings.warn(
    "tuning.skaks_opt.evaluator is deprecated; use skaks_opt.evaluator instead.",
    DeprecationWarning,
    stacklevel=2,
)

from skaks_opt.evaluator import *  # noqa: F401,F403
