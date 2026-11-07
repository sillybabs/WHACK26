"""scoring/live.py - the small live trackers: smoothing, band hold, weakest joint, streak.
Each takes `now` (seconds) so tests can drive a fake clock."""
from collections import deque
import config
from scoring.common import static_band


class ScoreSmoother:
    def __init__(self, window_s=None):
        self.window_s = window_s if window_s is not None else config.SCORE_SMOOTH_S
        self.hist = deque()

    def update(self, value, now):
        self.hist.append((now, value))
        while self.hist and now - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        return sum(v for _, v in self.hist) / len(self.hist)

    def reset(self):
        self.hist.clear()


class BandTracker:
    """Green / amber / red with a hold: a new band must persist BAND_HOLD_S before it shows."""

    def __init__(self):
        self.band = None
        self.cand = None
        self.since = 0.0

    def update(self, score, now):
        wanted = static_band(score)
        if self.band is None:
            self.band = wanted
        elif wanted == self.band:
            self.cand = None
        elif wanted != self.cand:
            self.cand, self.since = wanted, now
        elif now - self.since >= config.BAND_HOLD_S:
            self.band, self.cand = wanted, None
        return self.band

    def reset(self):
        self.band, self.cand, self.since = None, None, 0.0


class WorstJointTracker:
    def __init__(self, window_s=None):
        self.window_s = window_s if window_s is not None else config.WORST_JOINT_WINDOW_S
        self.hist = deque()

    def update(self, per_joint, now):
        self.hist.append((now, per_joint))
        while self.hist and now - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        tot, cnt = {}, {}
        for _, pj in self.hist:
            for j, e in pj.items():
                tot[j] = tot.get(j, 0.0) + e
                cnt[j] = cnt.get(j, 0) + 1
        return max(tot, key=lambda j: tot[j] / cnt[j]) if tot else None

    def reset(self):
        self.hist.clear()


class StreakCounter:
    """Seconds of consecutive green. Holds on amber, resets on red."""

    def __init__(self):
        self.seconds = 0.0
        self.last = None

    def update(self, band, now):
        dt = 0.0 if self.last is None else now - self.last
        self.last = now
        if band == "green":
            self.seconds += dt
        elif band == "red":
            self.seconds = 0.0
        return self.seconds

    def reset(self):
        self.seconds, self.last = 0.0, None
