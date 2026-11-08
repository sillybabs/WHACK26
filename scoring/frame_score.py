"""Per-frame scoring: angle difference -> points, per-segment compare, overall."""

from .config import (
    BAND1_SCORE_LOW, BAND2_SCORE_LOW, BAND3_SCORE_LOW, COLOR_BAD, COLOR_GOOD,
    COLOR_OK, SCORE_BANDS,
)


def angle_diff(live, ref):
    """Signed difference live - ref, wrapped to -180..180 (so +179 vs -179 = 2)."""
    return (live - ref + 180) % 360 - 180


def score_diff(diff):
    """Turn an angle difference (degrees, sign ignored) into 0-100 points."""
    d = abs(diff)
    for start, end, score_high, score_low in SCORE_BANDS:
        if d <= end:
            t = (d - start) / (end - start)  # 0 at band start, 1 at band end
            return score_high + t * (score_low - score_high)
    return BAND3_SCORE_LOW  # past the last band


def color_for_score(score):
    if score >= BAND1_SCORE_LOW:
        return COLOR_GOOD
    if score >= BAND2_SCORE_LOW:
        return COLOR_OK
    return COLOR_BAD


def compare(live_angles, ref_angles):
    """Score every segment. Returns {name: dict or None}.

    Each dict has: live, ref (angles), diff (signed, wrapped), score (0-100).
    A segment is None if it isn't visible in the live feed or the reference
    (or if the reference has no person in the current frame).
    """
    results = {}
    for name, live in live_angles.items():
        ref = ref_angles.get(name)
        if live is None or ref is None:
            results[name] = None
            continue
        diff = angle_diff(live, ref)
        results[name] = {
            "live": live, "ref": ref, "diff": diff, "score": score_diff(diff),
        }
    return results


def overall_score(results):
    """Average score of all segments that could be compared (None if none)."""
    scores = [r["score"] for r in results.values() if r is not None]
    return sum(scores) / len(scores) if scores else None
