import threading
import numpy as np
import config


class BalanceBar:
    """Chess.com-style bar. Starts at 0.5 (50/50), ends at 1.0 if every moment is green."""

    def __init__(self, clip_seconds: float):
        self.bucket_s = config.BALANCE_BUCKET_S            # 0.1
        self.n = max(1, int(np.ceil(clip_seconds / self.bucket_s)))
        self.v = np.zeros(self.n, dtype=np.float32)        # +1 green, amber value, -1 red
        self.lock = threading.Lock()                       # pose thread writes, flow thread clears

    def _index(self, ref_t: float) -> int:
        return min(self.n - 1, max(0, int(ref_t / self.bucket_s)))

    def record(self, ref_t: float, band: str) -> None:
        val = {"green": 1.0, "amber": config.BALANCE_AMBER_VALUE, "red": -1.0}[band]
        with self.lock:
            self.v[self._index(ref_t)] = val               # overwrite: rewinds replace, never add

    def clear_from(self, ref_t: float) -> None:
        """Called by flow on pause, with the interrupted section's start time."""
        with self.lock:
            self.v[self._index(ref_t):] = 0.0

    @property
    def value(self) -> float:
        with self.lock:
            return float(0.5 + 0.5 * self.v.sum() / self.n)  # 0..1 = green share

    def reset(self) -> None:                                # Try Again
        with self.lock:
            self.v[:] = 0.0