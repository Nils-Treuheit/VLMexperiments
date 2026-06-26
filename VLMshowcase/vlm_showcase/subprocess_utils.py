import subprocess
import sys
import threading
import time


def run_with_timer(cmd, timeout=300, label="Processing"):
    stopped = threading.Event()
    start = time.time()
    last_len = 0

    def ticker():
        while not stopped.is_set():
            elapsed = time.time() - start
            msg = f"\r{label} ... ({int(elapsed)}s)"
            nonlocal last_len
            sys.stderr.write(msg + " " * max(0, last_len - len(msg)))
            sys.stderr.flush()
            last_len = len(msg)
            stopped.wait(0.5)

    t = threading.Thread(target=ticker, daemon=True)
    t.start()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc
    finally:
        stopped.set()
        t.join(timeout=2)
        elapsed = time.time() - start
        sys.stderr.write(f"\rDone. {label} in {int(elapsed)}s\n")
        sys.stderr.flush()
