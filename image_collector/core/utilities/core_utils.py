import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stderr():
    try:
        original_stderr_fd = sys.stderr.fileno()
    except Exception:
        # If sys.stderr does not have a fileno(), just yield.
        yield
        return
    with open(os.devnull, 'w') as devnull:
        old_stderr_fd = os.dup(original_stderr_fd)
        os.dup2(devnull.fileno(), original_stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, original_stderr_fd)
            os.close(old_stderr_fd)