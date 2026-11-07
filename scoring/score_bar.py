"""Score bar logic (chess.com style): smoothing, banding and the balance bar."""

import math
from collections import deque

import numpy as np

from .config import (
    BALANCE_AMBER_MIN, BALANCE_AMBER_VALUE, BALANCE_BUCKET_S, BALANCE_GREEN_MIN,
    BALANCE_MODE, BALANCE_SMOOTH_S, BALANCE_WARMUP_S,
)


class ScoreSmoother:
    """Average of the overall score over the last `window_s` seconds."""

    def __init__(self, window_s=BALANCE_SMOOTH_S):
        self.window_s = window_s
        self.hist = deque()

    def update(self, value, now):
        self.hist.append((now, value))
        while self.hist and now - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        return sum(v for _, v in self.hist) / len(self.hist)

    def reset(self):
        self.hist.clear()


def balance_band(score):
    """'green' / 'amber' / 'red' for a smoothed overall score (same cut-offs as the colours)."""
    if score >= BALANCE_GREEN_MIN:
        return "green"
    if score >= BALANCE_AMBER_MIN:
        return "amber"
    return "red"


class BalanceBar:
    """Starts at 0.5 (50% green). Ends at 1.0 if every moment of the video is green."""

    def __init__(self, clip_seconds):
        self.n = max(1, int(math.ceil(clip_seconds / BALANCE_BUCKET_S)))
        self.v = np.zeros(self.n, dtype=np.float32)   # +1 green, amber value, -1 red
        self.last = -1                                # furthest bucket played so far

    def record(self, video_t, band):
        i = min(self.n - 1, max(0, int(video_t / BALANCE_BUCKET_S + 1e-9)))
        self.v[i] = {"green": 1.0, "amber": BALANCE_AMBER_VALUE, "red": -1.0}[band]
        self.last = max(self.last, i)

    @property
    def value(self):
        """0..1, the green share of the bar."""
        if BALANCE_MODE == "running":
            warmup = int(BALANCE_WARMUP_S / BALANCE_BUCKET_S)
            denom = min(self.n, max(self.last + 1, warmup, 1))
        else:
            denom = self.n
        return float(min(1.0, max(0.0, 0.5 + 0.5 * self.v.sum() / denom)))

    def reset(self):
        self.v[:] = 0.0
        self.last = -1
