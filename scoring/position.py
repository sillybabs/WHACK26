"""scoring/position.py - Component A: position & angle accuracy (spec 3)."""
import bisect
import WHACK26.config as config
from scoring.common import JOINTS, JOINT_PART


class Reference:
    """The saved reference clip (data/ref/<clip>.json), searchable by video time."""

    def __init__(self, frames):
        self.frames = sorted(frames, key=lambda f: f["t"])
        self.times = [f["t"] for f in self.frames]
        self.duration = self.times[-1] if self.times else 0.0
        self.events = None            # filled by scoring.events.detect_reference

    def window(self, t, window_s):
        """Reference frames from the last `window_s` seconds up to `t`."""
        lo = bisect.bisect_left(self.times, t - window_s)
        hi = bisect.bisect_right(self.times, t)
        return self.frames[lo:hi]


def reference_quality(frames):
    """Share of reference frames with enough usable joints (spec 15: warn on poor clips)."""
    if not frames:
        return {"tracked_pct": 0.0, "ok": False}
    good = 0
    for f in frames:
        n = sum(1 for j in JOINTS
                if f["angles"].get(j) is not None and f["visible"].get(JOINT_PART[j], True))
        if n >= config.MIN_ACTIVE_JOINTS:
            good += 1
    pct = good / len(frames)
    return {"tracked_pct": round(100 * pct, 1), "ok": pct >= config.MIN_REF_TRACKED}


def active_joints(user_angles, user_visible, ref_angles, ref_visible):
    """Only joints visible AND measured on both sides. Missing = unavailable, never zero error."""
    out = []
    for j in JOINTS:
        part = JOINT_PART[j]
        if not (user_visible.get(part) and ref_visible.get(part)):
            continue
        if user_angles.get(j) is None or ref_angles.get(j) is None:
            continue
        out.append(j)
    return out


def best_match(user_angles, user_visible, ref_window):
    """Compare against every reference frame in the window and keep the lowest weighted error.
    This tolerates small copying delays, so it is for the LIVE POSITION score only.
    The timing score must never use it (spec 3 and 4.2).
    Returns {error, abs, signed, matched_t, active} or None."""
    best = None
    for rf in ref_window:
        active = active_joints(user_angles, user_visible, rf["angles"], rf["visible"])
        if len(active) < config.MIN_ACTIVE_JOINTS:
            continue
        w = {j: config.JOINT_WEIGHTS.get(j, 1.0) for j in active}
        wsum = sum(w.values())
        if wsum <= 0:
            continue
        signed = {j: user_angles[j] - rf["angles"][j] for j in active}
        err = sum(abs(signed[j]) * w[j] for j in active) / wsum      # E_position
        if best is None or err < best["error"]:
            best = {"error": err, "abs": {j: abs(signed[j]) for j in active},
                    "signed": signed, "matched_t": rf["t"], "active": active}
    return best
