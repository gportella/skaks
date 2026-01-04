#!/usr/bin/env python3
"""CLI wrapper around tuning.skaks_opt.regression."""

from __future__ import annotations

import pathlib
import sys


def _ensure_repo_root() -> None:
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent
    for path in (script_dir, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def main() -> None:
    _ensure_repo_root()
    from tuning.skaks_opt import regression

    regression.main()


if __name__ == "__main__":
    main()
