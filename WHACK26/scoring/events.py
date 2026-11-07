"""scoring/events.py - Components B, C, D: movement timing, sharpness, hold duration (spec 4-6).

Idea: a joint's angle series is cut into STABLE runs (angle stays within a tolerance band for a
short time) and the gaps between them. A big change between two stable runs is a MOVEMENT:
  movement start = last moment of the first run, target arrival = first moment of the next run.
A stable run longer than HOLD_MIN_S is a HOLD.

Everything here uses the ORIGINAL video timestamps. Nothing is time-shifted, so lateness shows up.
Events that are unclear are simply not detected and not scored (spec 4.1: no forced matches).
"""
import config
from scoring.common import JOINTS, JOINT_PART, swap_lr, section_index_at, linear_score


def _segments(series):
    """Split [(t, value|None)] into continuous pieces at missing values and big time gaps."""
    segs, cur, prev_t = [], [], None
    for t, v in series:
        if v is None:
            if cur:
                segs.append(cur)
                cur = []
            prev_t = None
            continue
        if prev_t is not None and t - prev_t > config.EVENT_MAX_GAP_S:
            if cur:
                segs.append(cur)
                cur = []
        cur.append((t, v))
        prev_t = t
    if cur:
        segs.append(cur)
    return segs


def _smooth(vals, k):
    h = max(0, k // 2)
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - h), min(len(vals), i + h + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def _runs(ts, vs):
    """Stable runs as [first_idx, last_idx, anchor_angle]."""
    tol, min_dur = config.EVENT_STABLE_TOL_DEG, config.EVENT_ARRIVAL_STABLE_S
    n, runs, i = len(vs), [], 0
    while i < n:
        anchor, j = vs[i], i
        while j + 1 < n and abs(vs[j + 1] - anchor) <= tol:
            j += 1
        if ts[j] - ts[i] >= min_dur:
            runs.append([i, j, anchor])
            i = j + 1
        else:
            i += 1
    # a pose that drifted a little past the tolerance is still the same pose: merge
    merged = []
    for r in runs:
        if merged:
            p = merged[-1]
            if abs(r[2] - p[2]) < config.EVENT_MIN_MOVE_DEG and ts[r[0]] - ts[p[1]] < min_dur:
                p[1] = r[1]
                continue
        merged.append(r)
    return merged


def detect_series(series):
    """series = [(t, angle|None)] for ONE joint -> (movement events, holds)."""
    events, holds = [], []
    for seg in _segments(series):
        if len(seg) < 3:
            continue
        ts = [t for t, _ in seg]
        vs = _smooth([v for _, v in seg], config.EVENT_SMOOTH_SAMPLES)
        runs = _runs(ts, vs)
        n = len(seg)
        for i, j, a in runs:
            dur = ts[j] - ts[i]
            # a run touching either end of the data may be cut short, so its length is unknown
            if i > 0 and j < n - 1 and dur >= config.HOLD_MIN_S:
                holds.append({"start_t": ts[i], "end_t": ts[j], "duration": dur})
        for r1, r2 in zip(runs, runs[1:]):
            d_ang = r2[2] - r1[2]
            gap = ts[r2[0]] - ts[r1[1]]
            if abs(d_ang) >= config.EVENT_MIN_MOVE_DEG and gap <= config.EVENT_MAX_MOVE_S:
                events.append({"start_t": ts[r1[1]], "arrival_t": ts[r2[0]], "duration": gap,
                               "direction": 1 if d_ang > 0 else -1})
    return events, holds


def detect_reference(frames):
    """Run once when a clip is loaded: {joint: (events, holds)} on the reference timeline."""
    out = {}
    for j in JOINTS:
        part = JOINT_PART[j]
        series = [(f["t"], f["angles"].get(j) if f["visible"].get(part, True) else None)
                  for f in frames]
        out[j] = detect_series(series)
    return out


def _match(ref_items, user_items, time_of, ok, window):
    """Pair reference and user items in chronological order, nearest first, never forced."""
    pairs, last = [], -1
    for r in ref_items:
        best = None
        for idx in range(last + 1, len(user_items)):
            u = user_items[idx]
            dt = time_of(u) - time_of(r)
            if dt > window:
                break
            if abs(dt) <= window and ok(r, u) and (best is None or abs(dt) < abs(best[1])):
                best = (idx, dt)
        if best is not None:
            pairs.append((r, user_items[best[0]]))
            last = best[0]
    return pairs


def analyse(ref_events, user_series, mirror):
    """Match user events to reference events joint by joint and score them.

    user_series = {joint: [(video_time, angle|None)]}
    Returns {"matches": [...], "holds": [...], "n_ref_events", "n_user_events"}.
    Signed values are user minus reference: positive offset = later, positive duration
    difference = slower / held longer.
    """
    W = config.EVENT_MATCH_WINDOW_S
    matches, hold_matches = [], []
    n_ref = n_user = 0
    for j in JOINTS:
        rj = swap_lr(j) if mirror else j            # mirrored: my left matches their right
        if rj not in ref_events:
            continue
        r_ev, r_hold = ref_events[rj]
        u_ev, u_hold = detect_series(user_series.get(j, []))
        n_ref += len(r_ev)
        n_user += len(u_ev)
        for r, u in _match(r_ev, u_ev, lambda e: e["arrival_t"],
                           lambda a, b: a["direction"] == b["direction"], W):
            offset = u["arrival_t"] - r["arrival_t"]
            ddur = u["duration"] - r["duration"]
            matches.append({
                "joint": j, "section": section_index_at(r["arrival_t"]),
                "ref_arrival": r["arrival_t"], "user_arrival": u["arrival_t"],
                "offset_s": offset,
                "timing": linear_score(abs(offset), config.TIMING_TOLERANCE_S),
                "ref_duration": r["duration"], "user_duration": u["duration"],
                "duration_diff_s": ddur,
                "sharpness": linear_score(abs(ddur), config.SHARPNESS_TOLERANCE_S),
            })
        for r, u in _match(r_hold, u_hold, lambda h: h["start_t"], lambda a, b: True, W):
            diff = u["duration"] - r["duration"]
            hold_matches.append({
                "joint": j, "section": section_index_at(r["start_t"]),
                "ref_start": r["start_t"], "ref_hold": r["duration"],
                "user_hold": u["duration"], "diff_s": diff,
                "holds": linear_score(abs(diff), config.HOLD_TOLERANCE_S),
            })
    return {"matches": matches, "holds": hold_matches,
            "n_ref_events": n_ref, "n_user_events": n_user}
