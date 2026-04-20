from contextlib import contextmanager


@contextmanager
def pipes(*args, **kwargs):
    """Windows-safe no-op replacement for wurlitzer.pipes."""
    yield