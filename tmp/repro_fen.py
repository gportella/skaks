import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
engine = root / "build" / "release" / "src" / "skaks"
fen = "r7/ppp2kp1/5n2/3pP3/8/3BB3/PPP2P1P/R3K3 w Q - 1 20"
script_input = "\n".join(
    ["uci", "isready", "ucinewgame", f"position fen {fen}", "go depth 4", "quit", ""]
)
result = subprocess.run(
    [str(engine), "--uci"], input=script_input, capture_output=True, text=True
)
print("RETURN CODE:", result.returncode)
print("STDOUT:\n" + result.stdout)
print("STDERR:\n" + result.stderr)
