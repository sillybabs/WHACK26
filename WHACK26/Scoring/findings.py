"""scoring/findings.py - ranked, evidence-backed findings (spec 11: top_findings).
Python decides what the issues are. The LLM only explains them."""
import config
from scoring.common import error_to_score, joint_direction


def _cand(kind, joint, section, ts, measured, loss, conf, direction=None):
    c = {"type": kind, "body_part": joint, "section": section + 1,
         "timestamp_s": round(ts, 1), "measured_value": measured,
         "confidence": round(conf, 2), "_loss": loss}
    if direction:
        c["direction"] = direction
    return c


def _ms(x):
    return int(round(abs(x) * 1000))


def build(sections, matches, holds):
    valid = {s["index"]: s for s in sections if s["valid"]}
    cands = []

    # position: one candidate per joint per section
    for k, sec in valid.items():
        for j, d in sec["_joints"].items():
            if d["n"] < config.MIN_PART_SAMPLES_SECTION:
                continue
            m = d["sum"] / d["n"]
            conf = d["n"] / max(1, sec["_n_samples"])
            cands.append(_cand("position", j, k, d["max_t"],
                               f"{int(round(m))} degree mean difference from reference",
                               100 - error_to_score(m), conf,
                               joint_direction(j, d["signed"] / d["n"])))

    # timing / sharpness: group matched events by (joint, section)
    groups = {}
    for x in matches:
        if x["section"] in valid:
            groups.setdefault((x["joint"], x["section"]), []).append(x)
    for (j, k), xs in groups.items():
        conf = valid[k]["confidence"]
        off = sum(x["offset_s"] for x in xs) / len(xs)
        worst = max(xs, key=lambda x: abs(x["offset_s"]))
        cands.append(_cand("timing", j, k, worst["ref_arrival"],
                           f"{_ms(off)} ms {'later' if off > 0 else 'earlier'} than reference",
                           100 - sum(x["timing"] for x in xs) / len(xs), conf))
        dd = sum(x["duration_diff_s"] for x in xs) / len(xs)
        worst = max(xs, key=lambda x: abs(x["duration_diff_s"]))
        cands.append(_cand("sharpness", j, k, worst["ref_arrival"],
                           f"movement took {_ms(dd)} ms {'longer' if dd > 0 else 'less time'} than reference",
                           100 - sum(x["sharpness"] for x in xs) / len(xs), conf))

    # holds
    groups = {}
    for x in holds:
        if x["section"] in valid:
            groups.setdefault((x["joint"], x["section"]), []).append(x)
    for (j, k), xs in groups.items():
        diff = sum(x["diff_s"] for x in xs) / len(xs)
        worst = max(xs, key=lambda x: abs(x["diff_s"]))
        cands.append(_cand("holds", j, k, worst["ref_start"],
                           f"hold lasted {_ms(diff)} ms {'longer' if diff > 0 else 'shorter'} than reference",
                           100 - sum(x["holds"] for x in xs) / len(xs), valid[k]["confidence"]))

    cands = [c for c in cands
             if c["_loss"] >= config.FINDING_MIN_LOSS and c["confidence"] >= config.FINDING_MIN_CONFIDENCE]
    cands.sort(key=lambda c: c["_loss"] * c["confidence"], reverse=True)

    out, per_type, seen = [], {}, set()
    for c in cands:
        key = (c["type"], c["body_part"])            # one finding per measure per joint
        if key in seen or per_type.get(c["type"], 0) >= config.FINDINGS_MAX_PER_TYPE:
            continue
        seen.add(key)
        per_type[c["type"]] = per_type.get(c["type"], 0) + 1
        out.append({k: v for k, v in c.items() if not k.startswith("_")})
        if len(out) >= config.FINDINGS_MAX:
            break
    return out
