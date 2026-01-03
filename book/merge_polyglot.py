#!/usr/bin/env python
"""Merge multiple Polyglot opening books into a single file.

The script prefers entries coming from the largest source book (by file size).
For identical positions and moves coming from books of the same priority, the
entry with the highest (weight, learn) pair is kept. Polyglot keys are used to
identify identical positions.
"""

from __future__ import annotations

import argparse
import logging
import struct
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import chess
import chess.polyglot

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.theme import Theme
except ImportError:  # pragma: no cover - cosmetic dependency
    box = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Theme = None  # type: ignore[assignment]

PROMOTION_TO_POLYGLOT = {
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

ANARCHY_WHISPER = "whispered gambit: ranks unravel, files riot."


def describe_count(value: int) -> str:
    """Return a conversational approximation for large counts."""

    if value >= 2_000_000:
        approx = round(value / 100_000) / 10.0
        text = f"{approx:.1f}".rstrip("0").rstrip(".")
        return f"about {text}M"
    if value >= 1_050_000:
        approx = round(value / 100_000) / 10.0
        text = f"{approx:.1f}".rstrip("0").rstrip(".")
        return f"about {text}M"
    if value >= 900_000:
        return "almost a million"
    if value >= 500_000:
        return "over half a million"
    if value >= 100_000:
        approx = round(value / 50_000) * 50_000
        return f"about {approx // 1_000}k"
    if value >= 10_000:
        approx = round(value / 5_000) * 5_000
        return f"around {approx // 1_000}k"
    if value >= 1_000:
        approx = round(value / 1_000) * 1_000
        return f"roughly {approx // 1_000}k"
    if value >= 100:
        approx = round(value / 10) * 10
        return f"around {approx}"
    return f"about {value}"


def format_lines_summary(value: int) -> Tuple[str, str]:
    """Return the raw count and a short anarchic flourish."""

    return f"{value:,}", f"{describe_count(value)} · {ANARCHY_WHISPER}"

RICH_THEME = (
    Theme(
        {
            "title": "bold white on rgb(28,28,28)",
            "book": "bold cyan",
            "entries": "bold green",
            "total": "bold magenta",
            "warning": "bold red",
            "accent": "italic bright_black",
            "grid": "bright_blue",
            "tri": "rgb(255,128,0)",
            "spark": "rgb(180,0,255)",
            "crimson": "rgb(180,20,20)",
        }
    )
    if Theme
    else None
)

console = Console(theme=RICH_THEME) if Console else None


@dataclass
class AggregatedEntry:
    """In-memory representation of an entry tracked during the merge."""

    key: int
    move: chess.Move
    weight: int
    learn: int
    priority_rank: int
    sources: set[Path] = field(default_factory=set)

    def should_replace(self, other: "AggregatedEntry") -> bool:
        """Return True if self should replace other when conflicts occur."""

        if self.priority_rank != other.priority_rank:
            return self.priority_rank < other.priority_rank
        return (self.weight, self.learn) > (other.weight, other.learn)

    def to_bytes(self) -> bytes:
        """Encode the entry into Polyglot on-disk representation."""

        move_obj = self.move
        if callable(move_obj):  # Defensive: should not trigger in practice.
            move_obj = move_obj()

        raw_move = encode_polyglot_move(move_obj)
        return struct.pack(">QHHI", self.key, raw_move, self.weight, self.learn)


@dataclass
class MergeStats:
    """Collects statistics for reporting at the end of the merge."""

    total_inputs: int = 0
    total_entries_read: int = 0
    total_unique_entries: int = 0
    per_input_entries: Dict[Path, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, int]:
        return {
            "inputs": self.total_inputs,
            "read_entries": self.total_entries_read,
            "unique_entries": self.total_unique_entries,
        }


def discover_input_files(paths: Sequence[Path]) -> List[Path]:
    """Expand the provided paths into a list of .bin files."""

    discovered: List[Path] = []
    for path in paths:
        if path.is_file():
            if path.suffix.lower() == ".bin":
                discovered.append(path)
            else:
                logging.warning("Skipping non-bin file: %s", path)
        elif path.is_dir():
            discovered.extend(sorted(p for p in path.glob("*.bin") if p.is_file()))
        else:
            logging.warning("Skipping missing path: %s", path)
    return discovered


def sort_inputs_by_priority(paths: Iterable[Path]) -> List[Path]:
    """Sort paths so that larger books are processed first."""

    return sorted(paths, key=lambda p: (p.stat().st_size, p.name), reverse=True)


def load_entry(path: Path, priority_rank: int) -> Iterable[AggregatedEntry]:
    """Yield entries for the given book with attached priority metadata."""

    with chess.polyglot.open_reader(str(path)) as reader:
        for raw_entry in reader:
            move_attr = raw_entry.move
            move = move_attr() if callable(move_attr) else move_attr
            # Promote piece flag defaults to 0 when no promotion occurs.
            agg_entry = AggregatedEntry(
                key=raw_entry.key,
                move=move,
                weight=raw_entry.weight,
                learn=raw_entry.learn,
                priority_rank=priority_rank,
                sources={path},
            )
            yield agg_entry


def encode_polyglot_move(move: chess.Move | int) -> int:
    """Encode a move into Polyglot 16-bit representation."""

    if isinstance(move, int):
        return move

    from_sq = move.from_square
    to_sq = move.to_square
    promotion = move.promotion or 0

    promo_code = PROMOTION_TO_POLYGLOT.get(promotion, 0)
    return (from_sq << 10) | (to_sq << 4) | promo_code


def merge_books(
    paths: Sequence[Path],
) -> Tuple[Dict[Tuple[int, chess.Move], AggregatedEntry], MergeStats]:
    """Merge all entries from the provided books."""

    entries: Dict[Tuple[int, chess.Move], AggregatedEntry] = {}
    stats = MergeStats(total_inputs=len(paths))

    for priority_rank, path in enumerate(paths):
        logging.info("Reading %s (priority %s)", path, priority_rank)
        book_entry_count = 0
        for entry in load_entry(path, priority_rank):
            stats.total_entries_read += 1
            book_entry_count += 1
            key = (entry.key, entry.move)
            existing = entries.get(key)
            if existing is None:
                entries[key] = entry
            else:
                if entry.should_replace(existing):
                    entry.sources.update(existing.sources)
                    entries[key] = entry
                else:
                    existing.sources.update(entry.sources)
        stats.per_input_entries[path] = book_entry_count
    stats.total_unique_entries = len(entries)
    return entries, stats


def write_book(
    entries: Dict[Tuple[int, chess.Move], AggregatedEntry], output_path: Path
) -> int:
    """Write the aggregated entries to the Polyglot output file."""

    ordered_entries = sorted(
        entries.values(),
        key=lambda e: (
            e.key,
            e.move.from_square,
            e.move.to_square,
            e.move.promotion or 0,
            e.weight,
            e.learn,
        ),
    )

    with output_path.open("wb") as writer:
        for entry in ordered_entries:
            writer.write(entry.to_bytes())

    return len(ordered_entries)


def render_summary(stats: MergeStats, output_path: Path, output_entries: int) -> None:
    """Display an over-the-top summary of the merge results."""

    sorted_inputs = sorted(
        stats.per_input_entries.items(),
        key=lambda item: (-item[1], item[0].name),
    )

    piece_cycle = cycle(["◼", "♝", "◻", "♗", "◆", "◎", "●", "△"])

    if console is None or Table is None or Panel is None:
        logging.warning(
            "Install 'rich' for the full studio-grade summary: pip install rich"
        )
        print("Polyglot merge, atelier edition:")
        for icon, (path, count) in zip(piece_cycle, sorted_inputs):
            print(f"  {icon} {path.name}: {count:,} entries")
        print(f"  ♝ {output_path.name}: {output_entries:,} unique entries")
        print(
            "  Grand total processed: "
            f"{stats.total_entries_read:,} (unique: {stats.total_unique_entries:,})"
        )
        print("     |\n ┏━━━┻━━━┓\n ┃   △   ┃\n ┗┳━━━━━┳┛\n  ┃     ┃")
        return

    table = Table(
        title="ANALYTICAL LUMINARIUM",
        header_style="title",
        box=None,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Glyph", style="title", justify="center", width=4, no_wrap=True)
    table.add_column("Book", style="book", no_wrap=True, width=22)
    table.add_column("Count", style="entries", justify="right", width=14, no_wrap=True)
    table.add_column("Anarchy", style="entries", justify="left", width=68, no_wrap=True)

    for piece, (path, count) in zip(piece_cycle, sorted_inputs):
        raw_count, phrase = format_lines_summary(count)
        table.add_row(piece, path.name, raw_count, phrase)

    table.add_section()
    raw_count, phrase = format_lines_summary(output_entries)
    table.add_row("♝", f"{output_path.name} · compiled", raw_count, phrase)

    prelude = Panel.fit(
        "[title]⋅ Analytical Spectacles Engaged ⋅[/title]\n[accent]sprocket skylines flicker to life[/accent]",
        border_style="title",
    )

    scatter_top = [
        "       [grid]╵                              ╵                                                  │[/grid]",
        "[grid]│                                                                                            │[/grid]",
    ]
    scatter_mid = "[grid]╰──────────────────────────────── [tri]Δ spectral triangulation Δ[/tri] ────────────────────────────────╯[/grid]"
    console.print(prelude)
    for line in scatter_top:
        console.print(line)
    surround = Panel(
        table,
        border_style="grid",
        title="[spark]╳ lattice bloom ╳[/spark]",
        padding=(0, 1),
    )
    console.print(surround)

    offset_triangle = [
        "            [tri]▲[/tri]",
        "           [tri]╱[/tri] [tri]╲[/tri]",
        "          [tri]╱[/tri] [tri]▲[/tri] [tri]╲[/tri]",
        "         [tri]╱[/tri][tri]▲▲▲[/tri][tri]╲[/tri]",
        "        [tri]╱[/tri]━[tri]▲▲▲▲▲[/tri]━[tri]╲[/tri]",
    ]

    crimson_core = [
        "[crimson]▲[/crimson]",
        "[crimson]╱[/crimson] [crimson]╲[/crimson]",
        "[crimson]╱[/crimson] [crimson]▲[/crimson] [crimson]╲[/crimson]",
        "[crimson]╱[/crimson][crimson]▲▲▲[/crimson][crimson]╲[/crimson]",
        "[crimson]╱[/crimson][crimson]━[/crimson][crimson]▲▲▲▲▲[/crimson][crimson]━[/crimson][crimson]╲[/crimson]",
    ]

    crimson_offsets = [54, 52, 50, 48, 46]

    for left, core, offset in zip(offset_triangle, crimson_core, crimson_offsets):
        right = " " * offset + core
        console.print(f"{left}{right}")

    flourish = (
        "[total]lines traversed:[/total] "
        f"[entries]{stats.total_entries_read:,}[/entries] · "
        "[total]distinct threads curated:[/total] "
        f"[entries]{stats.total_unique_entries:,}[/entries]"
    )
    console.print(flourish)

    console.print(
        "[title]...returning to the atelier, circuitry humming; bishops doodle sigils in the margins.[/title]"
    )

    console.print(
        "[spark]⋰⋱ chromatic residues sparkle along the lattices, lenses aglow ⋰⋱[/spark]"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Polyglot .bin files or directories containing them.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Destination Polyglot book file.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return parser.parse_args(argv)


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.quiet)

    inputs = discover_input_files(args.inputs)
    if not inputs:
        logging.error("No .bin files found among the provided inputs.")
        return 1

    sorted_inputs = sort_inputs_by_priority(inputs)
    logging.info("Merging %d books.", len(sorted_inputs))

    merged_entries, stats = merge_books(sorted_inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_entries = write_book(merged_entries, args.output)
    render_summary(stats, args.output, output_entries)

    logging.info(
        "Merge complete: inputs=%d read_entries=%d unique_entries=%d",
        stats.total_inputs,
        stats.total_entries_read,
        stats.total_unique_entries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
