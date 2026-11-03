"""Mirror stdout into a run's own log file."""

from __future__ import annotations

import contextlib
import sys


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def tee_stdout(path):
    """Everything printed inside the block also goes to `path` (overwritten)."""
    with open(path, "w") as log_file:
        previous = sys.stdout
        sys.stdout = _Tee(previous, log_file)
        try:
            yield
        finally:
            sys.stdout = previous
