from __future__ import annotations

import warnings

from skaks_opt.cli import main as _main

warnings.warn(
    "tuning.skaks_opt.cli is deprecated; use skaks_opt.cli instead.",
    DeprecationWarning,
    stacklevel=2,
)


def main(argv=None) -> None:
    _main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
