"""scoring/recorder.py - keeps every scored moment, then builds sections, session score,
body-part scores and findings (spec 8, 8.1, 8.2, 9).

Rules it enforces:
  - interrupted sections never count (spec 9, 15)
  - missing data is "unavailable", never 100 and never 0 (spec 8.1)
  - overall = weighted mean of the components that exist, weights renormalised (spec 8.1)
  - body-part scores are NOT folded into the overall score (spec 8.2)
"""
import math
from collections import defaultdict

import config
from scoring import events, findings
from scoring.common import (JOINTS, UNAVAILABLE_PARTS, COMPONENTS, error_to_score, mean,
                            section_index_at, section_bounds, n_sections, static_band)


def combine(scores):
    """Adjusted overall = sum(w_k * s_k) / sum(w_k) over available components (spec 8.1).
    Returns (overall | None, [unavailable component names]).
    Position unavailable -> no confident overall."""
    num = den = 0.0
    missing = []
    for k in COMPONENTS:
        v = scores.get(k)
        if v is None:
            missing.append(k)
            continue
        w = config.COMPONENT_WEIGHTS[k]
        num += w * v
        den += w
    if scores.get("position") is None or den == 0:
        return None, missing
    return num / den, missing


def _stats(samples):
    tracked = [s for s in samples if s["tracked"]]
    n, nt = len(samples), len(tracked)
    joints = {}
    for s in tracked:
        for j, e in s["abs"].items():
            d = joints.setdefault(j, {"n": 0, "sum": 0.0, "signed": 0.0, "max": -1.0, "max_t": 0.0})
            d["n"] += 1
            d["sum"] += e
            d["signed"] += s["signed"][j]
            if e > d["max"]:
                d["max"], d["max_t"] = e, s["t"]
    coverage = nt / n if n else 0.0
    active_frac = mean(s["n_active"] / len(JOINTS) for s in tracked) if tracked else 0.0
    return {"n_samples": n, "n_tracked": nt, "coverage": coverage,
            "mean_p": mean(s["p"] for s in tracked), "joints": joints,
            "confidence": coverage * active_frac}


def _mean_ms(xs):
    return None if not xs else round(1000 * sum(xs) / len(xs), 1)


class RunRecorder:
    def __init__(self, reference):
        self.ref = reference
        if self.ref.events is None:
            self.ref.events = events.detect_reference(self.ref.frames)
        self.samples = []
        self.interrupted = set()

    # ------------------------------------------------------------------ recording
    def reset(self):
        self.samples, self.interrupted = [], set()

    def add_sample(self, t, matched_t, p, error, abs_err, signed_err, user_active, band):
        self.samples.append({"t": t, "section": section_index_at(t), "tracked": True,
                             "matched_t": matched_t, "p": p, "err": error, "abs": abs_err,
                             "signed": signed_err, "user": user_active,
                             "n_active": len(abs_err), "band": band})

    def add_untracked(self, t):
        """A moment we should have scored but couldn't (nothing comparable). Counts against coverage."""
        self.samples.append({"t": t, "section": section_index_at(t), "tracked": False,
                             "user": {}, "n_active": 0})

    def truncate_from(self, t):
        """Drop everything at or after video time t (a rewind is about to replay it)."""
        self.samples = [s for s in self.samples if s["t"] < t]

    def mark_interrupted(self, section_idx):
        self.interrupted.add(section_idx)

    # ------------------------------------------------------------------ timeline for D
    def timeline(self, step_s=1.0):
        """One entry per second: {refT, score, band, worstJoint, section, interrupted}."""
        buckets = defaultdict(list)
        for s in self.samples:
            if s["tracked"]:
                buckets[int(s["t"] // step_s)].append(s)
        out = []
        for k in range(int(math.ceil(self.ref.duration / step_s))):
            ss = buckets.get(k, [])
            sec = section_index_at(k * step_s)
            entry = {"refT": k * step_s, "score": None, "band": None, "worstJoint": None,
                     "section": sec + 1, "interrupted": sec in self.interrupted}
            if ss:
                entry["score"] = round(mean(s["p"] for s in ss), 1)
                entry["band"] = static_band(entry["score"])
                tot = defaultdict(list)
                for s in ss:
                    for j, e in s["abs"].items():
                        tot[j].append(e)
                entry["worstJoint"] = max(tot, key=lambda j: sum(tot[j]) / len(tot[j]))
            out.append(entry)
        return out

    # ------------------------------------------------------------------ the report
    def build_report(self, mirror=None):
        mirror = config.MIRROR if mirror is None else mirror
        ns = n_sections(self.ref.duration)

        live = sorted((s for s in self.samples if s["section"] not in self.interrupted),
                      key=lambda s: s["t"])
        by_sec = defaultdict(list)
        for s in live:
            by_sec[s["section"]].append(s)

        user_series = {j: [(s["t"], s["user"].get(j)) for s in live] for j in JOINTS}
        ev = events.analyse(self.ref.events, user_series, mirror)

        sections = []
        for k in range(ns):
            start, end = section_bounds(k)
            interrupted = k in self.interrupted
            st = _stats(by_sec.get(k, []))
            pos = None
            if (not interrupted and st["n_tracked"] >= config.MIN_SECTION_SAMPLES
                    and st["coverage"] >= config.MIN_POSITION_COVERAGE):
                pos = st["mean_p"]
            m = [] if interrupted else [x for x in ev["matches"] if x["section"] == k]
            h = [] if interrupted else [x for x in ev["holds"] if x["section"] == k]
            ok = config.MIN_EVENTS_SECTION
            comps = {"position": pos,
                     "timing": mean(x["timing"] for x in m) if len(m) >= ok else None,
                     "sharpness": mean(x["sharpness"] for x in m) if len(m) >= ok else None,
                     "holds": mean(x["holds"] for x in h) if len(h) >= ok else None}
            overall, missing = combine(comps)
            valid = pos is not None
            weakest, weakest_err = None, None
            cand = {j: d for j, d in st["joints"].items() if d["n"] >= config.MIN_PART_SAMPLES_SECTION}
            if cand:
                weakest = max(cand, key=lambda j: cand[j]["sum"] / cand[j]["n"])
                weakest_err = cand[weakest]["sum"] / cand[weakest]["n"]
            reason = None if valid else ("interrupted" if interrupted else "not enough tracked data")
            sections.append({
                "index": k, "section": k + 1, "start_s": start, "end_s": end,
                "valid": valid, "excluded_reason": reason,
                "overall": overall, **comps, "unavailable": missing if valid else [],
                "weakest_joint": weakest, "mean_angle_error_deg": weakest_err,
                "arrival_offset_ms": _mean_ms([x["offset_s"] for x in m]) if comps["timing"] is not None else None,
                "movement_duration_difference_ms": _mean_ms([x["duration_diff_s"] for x in m]) if comps["sharpness"] is not None else None,
                "hold_duration_difference_ms": _mean_ms([x["diff_s"] for x in h]) if comps["holds"] is not None else None,
                "confidence": st["confidence"], "coverage": st["coverage"],
                "_joints": st["joints"], "_n_samples": st["n_samples"],
            })

        valid_idx = {s["index"] for s in sections if s["valid"]}
        vs = [x for k in valid_idx for x in by_sec.get(k, [])]
        st = _stats(vs)
        m = [x for x in ev["matches"] if x["section"] in valid_idx]
        h = [x for x in ev["holds"] if x["section"] in valid_idx]
        pos = st["mean_p"] if (valid_idx and st["n_tracked"] >= config.MIN_SESSION_SAMPLES
                               and st["coverage"] >= config.MIN_POSITION_COVERAGE) else None
        comps = {"position": pos,
                 "timing": mean(x["timing"] for x in m) if len(m) >= config.MIN_EVENTS_SESSION else None,
                 "sharpness": mean(x["sharpness"] for x in m) if len(m) >= config.MIN_EVENTS_SESSION else None,
                 "holds": mean(x["holds"] for x in h) if len(h) >= config.MIN_EVENTS_SESSION else None}
        overall, missing = combine(comps)

        excluded = [s for s in sections if not s["valid"]]
        return {
            "scores": {"overall": overall, **comps, "scale": "0-100"},
            "unavailable_components": missing,
            "sections": sections,
            "body_part_scores": self._body_parts(st),
            "top_findings": findings.build(sections, m, h),
            "data_quality": {
                "tracked_frame_percentage": int(round(100 * st["coverage"])),
                "valid_sections": sorted(s["section"] for s in sections if s["valid"]),
                "excluded_sections": [s["section"] for s in excluded],
                "excluded_reasons": {str(s["section"]): s["excluded_reason"] for s in excluded},
                "timing_events_detected": len(m),
                "hold_events_detected": len(h),
                "timing_events_unmatched": max(0, ev["n_ref_events"] - len(ev["matches"])),
                "unavailable_components": missing,
                "confidence": round(st["confidence"], 2),
            },
            "timeline": self.timeline(),
        }

    def _body_parts(self, st):
        out = {}
        for j in JOINTS:
            d = st["joints"].get(j)
            n = d["n"] if d else 0
            conf = round(n / st["n_samples"], 2) if st["n_samples"] else 0.0
            if d and n >= config.MIN_PART_SAMPLES:
                m = d["sum"] / n
                out[j] = {"score": int(round(error_to_score(m))), "mean_angle_error_deg": int(round(m)),
                          "confidence": conf, "tracked_samples": n}
            else:
                out[j] = {"score": None, "metric": "unavailable", "confidence": conf, "tracked_samples": n}
        # No angle exists for these (spec 8.2): unavailable, never a made-up number
        for p in UNAVAILABLE_PARTS:
            out[p] = {"score": None, "metric": "unavailable", "confidence": 0.0, "tracked_samples": 0}
        return out
