"""Run scores: POSITION, TIMING, SHARPNESS, HOLDS and the weighted OVERALL."""

import numpy as np

from .config import (
    EVENT_MATCH_WINDOW_S, EVENT_MAX_GAP_S, EVENT_MERGE_GAP_S, EVENT_MIN_AMPLITUDE,
    EVENT_MIN_MOVE_S, EVENT_MOVE_DPS, EVENT_SMOOTH_S, EVENT_STABLE_MIN_S,
    HOLD_TOL_S, MIN_EVENTS_PER_COMPONENT, SHARPNESS_TOL_S, TIMING_TOL_S,
    WEIGHT_HOLDS, WEIGHT_POSITION, WEIGHT_SHARPNESS, WEIGHT_TIMING,
)

# POSITION  - how closely your segment angles match the reference (the average of
#             the per-frame overall score above).
# TIMING    - do you ARRIVE at each pose at the same time as the reference?
# SHARPNESS - do you complete each movement as fast as the reference?
# HOLDS     - do you keep each pose for as long as the reference?
# All are 0-100. Timing / sharpness / holds are measured from "events" found in each
# segment's angle over time: a movement (angle changing fast enough), its arrival
# (it stops), and the hold (it stays still until the next movement). Your events are
# paired with the reference's in order, using the real video clock.
def _smooth(t, a, win):
    """Moving average of `a` over +-win/2 seconds (t sorted, no NaNs)."""
    lo = np.searchsorted(t, t - win / 2, "left")
    hi = np.searchsorted(t, t + win / 2, "right")
    cs = np.concatenate([[0.0], np.cumsum(a)])
    return (cs[hi] - cs[lo]) / (hi - lo)


def _valid_pieces(t, a):
    """Split a series into pieces with no missing values and no big time gaps.

    Angles are unwrapped so +179 -> -179 counts as a 2 degree move, not 358.
    """
    pieces, start = [], None
    for i in range(len(t)):
        if np.isnan(a[i]):
            if start is not None:
                pieces.append((start, i))
                start = None
        elif start is None:
            start = i
        elif t[i] - t[i - 1] > EVENT_MAX_GAP_S:
            pieces.append((start, i))
            start = i
    if start is not None:
        pieces.append((start, len(t)))
    for s, e in pieces:
        if e - s >= 4:
            yield t[s:e], np.degrees(np.unwrap(np.radians(a[s:e])))


def detect_events(t, a):
    """Find the movements in one angle-over-time series.

    Returns a list of (start_s, arrival_s, hold_s). hold_s is None if the pose was
    still being held when the data ran out (so that hold can't be measured).
    """
    events = []
    for tt, aa in _valid_pieces(np.asarray(t, float), np.asarray(a, float)):
        s = _smooth(tt, aa, EVENT_SMOOTH_S)
        vel = np.zeros(len(tt))
        dt = np.diff(tt)
        vel[1:] = np.abs(np.diff(s) / np.where(dt > 0, dt, np.inf))
        moving = vel > EVENT_MOVE_DPS

        runs, i = [], 0
        while i < len(moving):
            if moving[i]:
                j = i
                while j + 1 < len(moving) and moving[j + 1]:
                    j += 1
                runs.append([i, j])
                i = j + 1
            else:
                i += 1
        merged = []
        for r in runs:
            if merged and tt[r[0]] - tt[merged[-1][1]] < EVENT_MERGE_GAP_S:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        good = [(r0, r1) for r0, r1 in merged
                if tt[r1] - tt[r0] >= EVENT_MIN_MOVE_S
                and np.ptp(s[max(0, r0 - 1):r1 + 1]) >= EVENT_MIN_AMPLITUDE]

        for k, (r0, r1) in enumerate(good):
            start_t, arr_t = float(tt[max(0, r0 - 1)]), float(tt[r1])
            nxt = float(tt[max(0, good[k + 1][0] - 1)]) if k + 1 < len(good) else None
            still_until = nxt if nxt is not None else float(tt[-1])
            if still_until - arr_t < EVENT_STABLE_MIN_S:
                continue            # didn't stay still: not a reliable arrival
            events.append((start_t, arr_t, None if nxt is None else nxt - arr_t))
    return events


def _pair_events(ref_events, user_events):
    """Pair reference and user events in order, by arrival time (within the match window)."""
    pairs, j = [], 0
    for r in ref_events:
        while j < len(user_events) and user_events[j][1] < r[1] - EVENT_MATCH_WINDOW_S:
            j += 1
        if j < len(user_events) and abs(user_events[j][1] - r[1]) <= EVENT_MATCH_WINDOW_S:
            pairs.append((r, user_events[j]))
            j += 1
    return pairs


def _tol_score(diff_s, tol_s):
    """0 s apart -> 100 points, tol_s or more apart -> 0."""
    return max(0.0, min(100.0, 100.0 * (1.0 - diff_s / tol_s)))


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


class RunRecorder:
    """Collects the scored frames of one run and works out position / timing /
    sharpness / holds when the run ends.

    record() - call once per scored frame
    finish() - returns {"overall", "position", "timing", "sharpness", "holds", ...}
    """

    def __init__(self, ref_data, ref_fps):
        self.ref_data = ref_data
        self.ref_fps = ref_fps
        self.t = []        # video time of each scored frame
        self.overall = []  # that frame's overall (position) score
        self.live = []     # {segment: live angle or None}
        self.done = False

    def record(self, t, results, overall):
        self.t.append(float(t))
        self.overall.append(float(overall))
        self.live.append({n: (None if r is None else r["live"]) for n, r in results.items()})

    @property
    def has_data(self):
        return len(self.t) > 0

    def _events(self):
        """(segment, reference event, your event) for every pair that could be matched."""
        ref_t = np.arange(len(self.ref_data)) / self.ref_fps
        user_t = np.array(self.t)
        names = sorted({n for p in self.live for n in p})
        out = []
        for name in names:
            ref_a = np.array([np.nan if ang is None or ang.get(name) is None else ang[name]
                              for _, ang in self.ref_data], float)
            user_a = np.array([np.nan if p.get(name) is None else p[name]
                               for p in self.live], float)
            for r, u in _pair_events(detect_events(ref_t, ref_a), detect_events(user_t, user_a)):
                out.append((name, r, u))
        return out

    def finish(self):
        pairs = self._events()
        timing_l = [_tol_score(abs(u[1] - r[1]), TIMING_TOL_S) for _, r, u in pairs]
        sharp_l = [_tol_score(abs((u[1] - u[0]) - (r[1] - r[0])), SHARPNESS_TOL_S)
                   for _, r, u in pairs]
        hold_l = [_tol_score(abs(u[2] - r[2]), HOLD_TOL_S)
                  for _, r, u in pairs if r[2] is not None and u[2] is not None]

        position = _mean(self.overall)
        timing = _mean(timing_l) if len(timing_l) >= MIN_EVENTS_PER_COMPONENT else None
        sharp = _mean(sharp_l) if len(sharp_l) >= MIN_EVENTS_PER_COMPONENT else None
        holds = _mean(hold_l) if len(hold_l) >= MIN_EVENTS_PER_COMPONENT else None

        # OVERALL = weighted mix of what could be measured. A component that couldn't be
        # measured is left out and the others re-weighted - it never counts as 100.
        comps = {"position": (position, WEIGHT_POSITION), "timing": (timing, WEIGHT_TIMING),
                 "sharpness": (sharp, WEIGHT_SHARPNESS), "holds": (holds, WEIGHT_HOLDS)}
        have = {k: c for k, c in comps.items() if c[0] is not None}
        overall = (sum(v * w for v, w in have.values()) / sum(w for _, w in have.values())
                   if have and position is not None else None)
        return {
            "overall": overall, "position": position, "timing": timing,
            "sharpness": sharp, "holds": holds,
            "unavailable": [k for k, (v, _) in comps.items() if v is None],
            "matched_events": len(pairs), "scored_frames": len(self.t),
            # per-event detail: how much later / slower / longer you were (ms)
            "events": [{
                "segment": n,
                "arrival_offset_ms": round((u[1] - r[1]) * 1000),                 # + = later
                "duration_diff_ms": round(((u[1] - u[0]) - (r[1] - r[0])) * 1000),  # + = slower
                "hold_diff_ms": (None if r[2] is None or u[2] is None
                                 else round((u[2] - r[2]) * 1000)),
            } for n, r, u in pairs],
        }


def finish_run(recorder, complete=True, callback=None):
    """Work out and print the run scores (once per run). Returns the result dict.

    callback - optional function called with the result dict when the run is scored
    (pose_webcam.py passes its RUN_SCORE_CALLBACK here).
    """
    if recorder is None or recorder.done or not recorder.has_data:
        return None
    if not complete and recorder.t[-1] - recorder.t[0] < 2.0:
        return None             # a few frames from a stopped run isn't worth reporting
    recorder.done = True
    res = recorder.finish()
    f = lambda v: "unavailable" if v is None else f"{v:.0f}"
    print("\n=== RUN SCORES (out of 100) ===")
    print(f"OVERALL {f(res['overall'])} | POSITION {f(res['position'])} | "
          f"TIMING {f(res['timing'])} | SHARPNESS {f(res['sharpness'])} | "
          f"HOLDS {f(res['holds'])}")
    if res["unavailable"]:
        print("Not measurable this run (left out of OVERALL):", ", ".join(res["unavailable"]))
    if not complete:
        print("(run was stopped before the video finished)")
    if callback is not None:
        callback(res)
    return res
