import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
engine_path = root / "build" / "release" / "src" / "skaks"
log_path = root / "tmp" / "uci_repro.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen(
    [str(engine_path), "--uci"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)
commands = [
    "uci",
    "isready",
    "ucinewgame",
    "position startpos moves",
    "go depth 1",
]
with log_path.open("w", encoding="utf-8") as log:
    for cmd in commands:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        time.sleep(0.1)
        while proc.poll() is None and proc.stdout and proc.stdout.readable():
            proc.stdout.flush()
        time.sleep(0.1)
    time.sleep(0.5)
    if proc.poll() is None:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        time.sleep(0.1)
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    return_code = proc.returncode
    log.write("STDOUT:\n")
    log.write(stdout)
    log.write("\nSTDERR:\n")
    log.write(stderr)
    log.write(f"\nRETURN CODE: {return_code}\n")
