from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from itertools import cycle
from typing import Iterator, List, Optional

try:  # pragma: no cover - optional rich dependency
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich optional
    Console = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore
    TimeRemainingColumn = None  # type: ignore
    TaskID = int  # type: ignore
    RICH_AVAILABLE = False


TARGET_PHRASE = "kctz rulez 4 lolz"
_GRADIENT_PALETTE = [
    "#1d4ed8",
    "#2563eb",
    "#0ea5e9",
    "#10b981",
    "#8b5cf6",
    "#f59e0b",
    "#f97316",
    "#ef4444",
    "#f43f5e",
]
_PROGRESS_CHARS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
_TRAIL_COLOR = "#1f2937"
_PHRASE_PALETTE = [
    "#a855f7",
    "#38bdf8",
    "#22c55e",
    "#facc15",
    "#fb7185",
    "#f97316",
    "#c084fc",
    "#7dd3fc",
    "#4ade80",
]
_CHESS_FRAMES = ["♔", "♕", "♖", "♗", "♘", "♙", "♚", "♛", "♜", "♝", "♞", "♟"]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{component:02x}" for component in rgb)


def _blend_rgb(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round((1.0 - t) * av + t * bv)) for av, bv in zip(a, b))


def _build_gradient_bar(ratio: float, segments: int = 48) -> str:
    ratio = max(0.0, min(1.0, ratio))
    segments = max(4, segments)
    progress = ratio * segments
    blocks: List[str] = []
    palette_rgb = [_hex_to_rgb(code) for code in _GRADIENT_PALETTE]
    last_index = len(palette_rgb) - 1

    for index in range(segments):
        position = index / max(1, segments - 1)
        scaled = position * last_index
        left = int(math.floor(scaled))
        right = min(last_index, left + 1)
        blend = scaled - left
        base_rgb = _blend_rgb(palette_rgb[left], palette_rgb[right], blend)
        fill_level = max(0.0, min(1.0, progress - index))

        if fill_level <= 0.0:
            blocks.append(f"[{_TRAIL_COLOR}]░[/]")
            continue

        level_index = min(
            len(_PROGRESS_CHARS) - 1,
            int(round(fill_level * (len(_PROGRESS_CHARS) - 1))),
        )
        char = _PROGRESS_CHARS[level_index]

        # Blend partially filled segments toward the palette color
        if fill_level < 1.0:
            blended_rgb = _blend_rgb(base_rgb, _hex_to_rgb(_TRAIL_COLOR), 1.0 - fill_level)
            color = _rgb_to_hex(blended_rgb)
        else:
            color = _rgb_to_hex(base_rgb)

        blocks.append(f"[{color}]{char}[/]")

    return "".join(blocks)


def _initial_phrase_state() -> List[dict]:
    state: List[dict] = []
    for ch in TARGET_PHRASE:
        state.append(
            {"target": ch, "current": ch if ch == " " else "?", "locked": False}
        )
    return state


def _update_phrase(
    ratio: float, time_ratio: float, state: List[dict], rng: random.Random
) -> str:
    ratio = max(0.0, min(1.0, ratio))
    time_ratio = max(0.0, min(1.0, time_ratio))
    progress_strength = min(1.0, (0.1 + 0.9 * max(ratio, time_ratio)) ** 2.4)
    colorful_letters: List[str] = []
    for index, entry in enumerate(state):
        target = entry["target"]
        if target == " ":
            colorful_letters.append(" ")
            continue
        if ratio >= 0.999 or time_ratio >= 0.999:
            entry["locked"] = True
            entry["current"] = target
        if not entry["locked"]:
            threshold = ((index + 1) / max(1, len(state))) ** 1.35
            adjusted = progress_strength
            if adjusted > threshold and rng.random() < 0.6:
                entry["locked"] = True
                entry["current"] = target
            else:
                entry["current"] = rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
        palette_index = int(
            round((max(ratio, time_ratio) ** 0.6) * (len(_PHRASE_PALETTE) - 1))
        )
        if entry["locked"]:
            color = _PHRASE_PALETTE[palette_index]
            colorful_letters.append(f"[bold {color}]{entry['current']}[/]")
        else:
            wobble_index = (index + rng.randint(0, len(_PHRASE_PALETTE) - 1)) % len(
                _PHRASE_PALETTE
            )
            color = _PHRASE_PALETTE[wobble_index]
            colorful_letters.append(f"[{color}]{entry['current']}[/]")
    return "".join(colorful_letters)


@dataclass
class FancyProgress:
    """Context-managed wrapper around Rich progress with animated status."""

    total: int
    rng: random.Random
    description: str = "Processing"
    console: Optional[Console] = None
    spinner: str = "aesthetic"

    def __post_init__(self) -> None:
        if not RICH_AVAILABLE:  # pragma: no cover - safeguarded by callers
            raise RuntimeError("Rich progress display is unavailable")
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        self._phrase_state: List[dict] = _initial_phrase_state()
        self._frames: Iterator[str] = cycle(_CHESS_FRAMES)
        self._start_time: float = 0.0

    def __enter__(self) -> "FancyProgress":
        columns = [
            TextColumn("[bold magenta]{task.fields[chess]}"),
            SpinnerColumn(spinner_name=self.spinner, style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[gradient]}", justify="left"),
            TextColumn("{task.completed}/{task.total}", style="bold cyan"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[phrase]}", style="bold"),
            TextColumn("{task.fields[status]}", style="bold yellow"),
        ]
        console = self.console or Console()
        progress = Progress(*columns, console=console)
        progress.__enter__()
        self._progress = progress
        self._task_id = progress.add_task(
            self.description,
            total=self.total,
            chess=next(self._frames),
            gradient=_build_gradient_bar(0.0),
            phrase=_update_phrase(0.0, 0.0, self._phrase_state, self.rng),
            status="",
        )
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._progress is not None:
            self._progress.__exit__(exc_type, exc, tb)
        self._progress = None

    def update(self, completed: int, message: str, status: str = "") -> None:
        if self._progress is None or self._task_id is None:
            return
        ratio = max(0.0, min(1.0, completed / max(1, self.total)))
        elapsed = time.perf_counter() - self._start_time
        avg_per_item = elapsed / max(1, completed)
        projected_total = avg_per_item * self.total
        time_ratio = elapsed / projected_total if projected_total > 0 else ratio
        self._progress.update(
            self._task_id,
            advance=1,
            description=message,
            chess=next(self._frames),
            gradient=_build_gradient_bar(ratio),
            phrase=_update_phrase(ratio, time_ratio, self._phrase_state, self.rng),
            status=status,
        )
