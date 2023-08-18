
from contextlib import contextmanager
import sys

@contextmanager
def silence_log():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr