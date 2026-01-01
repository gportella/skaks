from __future__ import annotations

import warnings

warnings.warn(
    "tuning.skaks_opt.data is deprecated; use skaks_opt.data instead.",
    DeprecationWarning,
    stacklevel=2,
)

from skaks_opt.data import *  # noqa: F401,F403
