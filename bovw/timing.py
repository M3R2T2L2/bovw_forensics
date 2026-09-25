"""Compute-budget logging: wall time, peak CPU RSS and peak GPU memory."""

import resource
import sys
import time
from contextlib import contextmanager


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024  # macOS bytes, Linux KiB


@contextmanager
def budget(record: dict, prefix: str):
    """Adds <prefix>_seconds, <prefix>_peak_rss_mb and (on GPU) <prefix>_peak_gpu_mb to `record`."""
    torch = sys.modules.get("torch")
    cuda = torch is not None and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
            record[f"{prefix}_peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
        record[f"{prefix}_seconds"] = time.perf_counter() - t0
        record[f"{prefix}_peak_rss_mb"] = _peak_rss_mb()
