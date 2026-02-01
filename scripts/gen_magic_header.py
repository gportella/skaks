#!/usr/bin/env python3
import struct
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_path = repo_root / "magic_cache.bin"
    header_path = repo_root / "include" / "chess" / "magic_bitboards_cache.hpp"

    data = bin_path.read_bytes()
    if len(data) != 128 * 8:
        raise SystemExit(f"Unexpected size {len(data)} bytes; expected 1024 bytes")

    values = struct.unpack("<128Q", data)
    rook = values[:64]
    bishop = values[64:]

    lines: list[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include \"chess/types.hpp\"")
    lines.append("")
    lines.append("#include <array>")
    lines.append("")
    lines.append("namespace chess {")
    lines.append("")
    lines.append("constexpr std::array<Bitboard, 64> kRookMagics = {")
    for i in range(0, 64, 4):
        chunk = ", ".join(f"0x{v:016x}ULL" for v in rook[i:i + 4])
        lines.append(f"    {chunk},")
    lines.append("};")
    lines.append("")
    lines.append("constexpr std::array<Bitboard, 64> kBishopMagics = {")
    for i in range(0, 64, 4):
        chunk = ", ".join(f"0x{v:016x}ULL" for v in bishop[i:i + 4])
        lines.append(f"    {chunk},")
    lines.append("};")
    lines.append("")
    lines.append("} // namespace chess")
    lines.append("")

    header_path.write_text("\n".join(lines))
    print(f"Wrote {header_path}")


if __name__ == "__main__":
    main()
