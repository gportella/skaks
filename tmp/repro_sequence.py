import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
engine = root / "build" / "release" / "src" / "skaks"
script_input = "\n".join(
    [
        "uci",
        "isready",
        "ucinewgame",
        "position startpos moves e2e4",
        "go depth 4",
        "quit",
        "",
    ]
)
result = subprocess.run(
    [str(engine), "--uci"], input=script_input, capture_output=True, text=True
)
print("RETURN CODE:", result.returncode)
print("STDOUT:\n" + result.stdout)
print("STDERR:\n" + result.stderr)
