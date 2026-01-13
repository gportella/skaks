"""Param optimizer CLI using the bundled implementation."""

from __future__ import annotations

import argparse

from skaks_opt import param_optimize_impl as _impl


def _configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return _impl.configure_parser(parser)


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "param-optimize",
        help="Arena-based self-play optimizer (baseline vs candidate)",
        description=(
            "Perturb-and-test loop with beam/CMA search, arena binding, and confidence "
            "guards. This is the primary tuning entry point; rerunning with an existing "
            "--output resumes from the saved best YAML/.best.json snapshot, so self-play "
            "runs can be chained without resetting to the original --start-params. GPC"
        ),
    )
    return _configure_parser(parser)


def run_param_optimize(args: argparse.Namespace) -> None:
    _impl.optimize_loop(args)


def main(argv: list[str] | None = None) -> None:
    parser = _configure_parser(
        argparse.ArgumentParser(
            description="Self-play parameter optimizer (beam + repeats)"
        )
    )
    parsed = parser.parse_args(argv)
    run_param_optimize(parsed)


if __name__ == "__main__":
    main()
