from __future__ import annotations

from pathlib import Path

import yaml


class _FlowSeq(list):
    pass


def _flow_seq_representer(dumper: yaml.Dumper, data: _FlowSeq):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _flowify(value):
    if isinstance(value, list):
        if any(isinstance(item, list) for item in value):
            return [_flowify(item) for item in value]
        return _FlowSeq([_flowify(item) for item in value])
    if isinstance(value, dict):
        return {key: _flowify(val) for key, val in value.items()}
    return value

MG_BONUS = [
    [[-175, -92, -74, -73], [-77, -41, -27, -15], [-61, -17, 6, 12], [-35, 8, 40, 49], [-34, 13, 44, 51], [-9, 22, 58, 53], [-67, -27, 4, 37], [-201, -83, -56, -26]],
    [[-53, -5, -8, -23], [-15, 8, 19, 4], [-7, 21, -5, 17], [-5, 11, 25, 39], [-12, 29, 22, 31], [-16, 6, 1, 11], [-17, -14, 5, 0], [-48, 1, -14, -23]],
    [[-31, -20, -14, -5], [-21, -13, -8, 6], [-25, -11, -1, 3], [-13, -5, -4, -6], [-27, -15, -4, 3], [-22, -2, 6, 12], [-2, 12, 16, 18], [-17, -19, -1, 9]],
    [[3, -5, -5, 4], [-3, 5, 8, 12], [-3, 6, 13, 7], [4, 5, 9, 8], [0, 14, 12, 5], [-4, 10, 6, 8], [-5, 6, 10, 8], [-2, -2, 1, -2]],
    [[271, 327, 271, 198], [278, 303, 234, 179], [195, 258, 169, 120], [164, 190, 138, 98], [154, 179, 105, 70], [123, 145, 81, 31], [88, 120, 65, 33], [59, 89, 45, -1]],
]

EG_BONUS = [
    [[-96, -65, -49, -21], [-67, -54, -18, 8], [-40, -27, -8, 29], [-35, -2, 13, 28], [-45, -16, 9, 39], [-51, -44, -16, 17], [-69, -50, -51, 12], [-100, -88, -56, -17]],
    [[-57, -30, -37, -12], [-37, -13, -17, 1], [-16, -1, -2, 10], [-20, -6, 0, 17], [-17, -1, -14, 15], [-30, 6, 4, 6], [-31, -20, -1, 1], [-46, -42, -37, -24]],
    [[-9, -13, -10, -9], [-12, -9, -1, -2], [6, -8, -2, -6], [-6, 1, -9, 7], [-5, 8, 7, -6], [6, 1, -7, 10], [4, 5, 20, -5], [18, 0, 19, 13]],
    [[-69, -57, -47, -26], [-55, -31, -22, -4], [-39, -18, -9, 3], [-23, -3, 13, 24], [-29, -6, 9, 21], [-38, -18, -12, 1], [-50, -27, -24, -8], [-75, -52, -43, -36]],
    [[1, 45, 85, 76], [53, 100, 133, 135], [88, 130, 169, 175], [103, 156, 172, 172], [96, 166, 199, 199], [92, 172, 184, 191], [47, 121, 116, 131], [11, 59, 73, 78]],
]

MG_PAWN = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 10, 19, 16, 19, 7, -5],
    [-9, -15, 11, 15, 32, 22, 5, -22],
    [-4, -23, 6, 20, 40, 17, 4, -8],
    [13, 0, -13, 1, 11, -2, -13, 5],
    [5, -12, -7, 22, -8, -5, -15, -8],
    [-7, 7, -3, -13, 5, -16, 10, -8],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

EG_PAWN = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-10, -6, 10, 0, 14, 7, -5, -19],
    [-10, -10, -10, 4, 4, 3, -6, -4],
    [6, -2, -8, -4, -13, -12, -10, -9],
    [10, 5, 4, -5, -5, -5, 14, 9],
    [28, 20, 21, 28, 30, 7, 6, 13],
    [0, -11, 12, 21, 25, 19, 4, 7],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

ORDER = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def mirror_row(row: list[int]) -> list[int]:
    return row + row[::-1]


def expand(table4: list[list[int]]) -> list[list[int]]:
    return [mirror_row(r) for r in table4]


def flip_ranks(table: list[list[int]]) -> list[list[int]]:
    return list(reversed(table))


def build_tables():
    mg_tables = {
        "pawn": flip_ranks(MG_PAWN),
        "knight": flip_ranks(expand(MG_BONUS[0])),
        "bishop": flip_ranks(expand(MG_BONUS[1])),
        "rook": flip_ranks(expand(MG_BONUS[2])),
        "queen": flip_ranks(expand(MG_BONUS[3])),
        "king": flip_ranks(expand(MG_BONUS[4])),
    }
    eg_tables = {
        "pawn": flip_ranks(EG_PAWN),
        "knight": flip_ranks(expand(EG_BONUS[0])),
        "bishop": flip_ranks(expand(EG_BONUS[1])),
        "rook": flip_ranks(expand(EG_BONUS[2])),
        "queen": flip_ranks(expand(EG_BONUS[3])),
        "king": flip_ranks(expand(EG_BONUS[4])),
    }
    return mg_tables, eg_tables


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "texel_fit_all.yaml"
    dst = root / "texel_fit_all_stockfish.yaml"

    with src.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    mg_tables, eg_tables = build_tables()
    eval_block = data.setdefault("evaluation", {})
    eval_block["pst_midgame"] = _flowify(mg_tables)
    eval_block["pst_endgame"] = _flowify(eg_tables)
    data = _flowify(data)
    yaml.add_representer(_FlowSeq, _flow_seq_representer, Dumper=yaml.SafeDumper)

    with dst.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)

    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
