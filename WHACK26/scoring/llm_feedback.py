"""scoring/llm_feedback.py - the LLM layer (spec 10-13, 15).

Python owns every number. This file only:
  1. packs the measured results into the compact input JSON (spec 11)
  2. asks the model to explain them, with the strict rules (spec 13)
  3. validates the reply against the schema and the input (spec 12, 16)
  4. falls back to deterministic template feedback if anything is wrong or slow (spec 15)
"""
import concurrent.futures
import json
import re

import WHACK26.config as config
from scoring.common import COMPONENTS, normalise_part, pretty, round_half_up

STATUSES = ("strong", "good", "improving", "needs_work", "unavailable")

OUTPUT_EXAMPLE = {
    "headline": "Good overall match - focus on arm shape and timing.",
    "overall_score": 78, "score_scale": "0-100",
    "score_summary": "One or two short sentences.",
    "component_feedback": [{"component": "position", "score": 80, "status": "good", "summary": "..."}],
    "strengths": [{"title": "...", "detail": "..."}],
    "priority_fixes": [{"rank": 1, "title": "...", "body_part": "left_elbow", "section": 2,
                        "timestamp_s": 6.4, "what_happened": "...", "evidence": "...",
                        "drill": "...", "focus_cue": "..."}],
    "next_practice_plan": {"first": "...", "second": "...", "success_target": "..."},
    "body_part_feedback": [{"body_part": "left_elbow", "score": 82, "status": "good", "summary": "..."}],
    "data_caveat": None,
}

SYSTEM_PROMPT = """You are a dance rehearsal coach. The supplied reference video is the standard.
Your job is to explain measured differences and give practical rehearsal advice.

STRICT RULES:
- Return JSON matching the agreed schema, and nothing else.
- Never calculate, alter, round differently, or invent scores. Copy scores exactly.
- Scores are all out of 100.
- Only discuss facts present in the input.
- Do not claim to hear or analyse the music or identify beats.
- Say "later than the reference" rather than "missed the beat".
- Do not call a movement technically wrong; describe how it differs from this reference.
- Prioritise up to three issues using size of error AND confidence.
- Give one specific, achievable drill per priority issue.
- If data is missing or confidence is low, say so and avoid unsupported claims.
- Match the coach personality sliders, but remain clear, constructive and respectful.
- Keep feedback concise and suitable for on-screen text and speech.

SCHEMA RULES:
- component_feedback uses only: position, timing, sharpness, holds.
- status is one of: strong, good, improving, needs_work, unavailable.
- priority_fixes has at most 3 items, and each one must match an entry in top_findings
  (same body_part, section and timestamp_s). If there is no reliable issue, return [].
- body_part_feedback lists every entry of body_part_scores, scored ones from lowest to highest
  score, then unavailable ones with score null and status "unavailable".
- data_caveat is null when there is nothing to flag.

Reply with JSON in exactly this shape:
""" + json.dumps(OUTPUT_EXAMPLE, indent=1)


def status_for(score):
    if score is None:
        return "unavailable"
    if score >= config.STATUS_STRONG:
        return "strong"
    if score >= config.STATUS_GOOD:
        return "good"
    if score >= config.STATUS_IMPROVING:
        return "improving"
    return "needs_work"


# ====================================================================== 1. input JSON
def _i(x):
    return None if x is None else round_half_up(x)


def build_llm_input(report, personality=None):
    sc = report["scores"]
    secs = []
    for s in report["sections"]:
        if not s["valid"]:
            continue
        secs.append({
            "section": s["section"], "start_s": s["start_s"], "end_s": s["end_s"],
            "overall": _i(s["overall"]), "position": _i(s["position"]), "timing": _i(s["timing"]),
            "sharpness": _i(s["sharpness"]), "holds": _i(s["holds"]),
            "weakest_joint": s["weakest_joint"], "mean_angle_error_deg": _i(s["mean_angle_error_deg"]),
            "arrival_offset_ms": _i(s["arrival_offset_ms"]),
            "movement_duration_difference_ms": _i(s["movement_duration_difference_ms"]),
            "hold_duration_difference_ms": _i(s["hold_duration_difference_ms"]),
            "confidence": round(s["confidence"], 2),
        })
    dq = {k: v for k, v in report["data_quality"].items() if k not in ("excluded_reasons",)}
    return {
        "task": "Create concise, actionable dance-coach feedback from measured results.",
        "coach_personality": personality or config.COACH_PERSONALITY,
        "scores": {"overall": _i(sc["overall"]), "position": _i(sc["position"]),
                   "timing": _i(sc["timing"]), "sharpness": _i(sc["sharpness"]),
                   "holds": _i(sc["holds"]), "scale": "0-100"},
        "sections": secs,
        "top_findings": report["top_findings"],
        "body_part_scores": report["body_part_scores"],
        "data_quality": dq,
    }


# ====================================================================== 3. validation
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text):
    return {float(x) for x in _NUM.findall(str(text))}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_llm_output(out, inp):
    """Return a list of problems. Empty list = safe to show."""
    errs = []
    if not isinstance(out, dict):
        return ["reply is not a JSON object"]
    for k in ("headline", "overall_score", "score_scale", "score_summary", "component_feedback",
              "strengths", "priority_fixes", "next_practice_plan", "body_part_feedback", "data_caveat"):
        if k not in out:
            errs.append(f"missing key: {k}")
    if errs:
        return errs

    sc = inp["scores"]
    if out["score_scale"] != "0-100":
        errs.append("score_scale must be '0-100'")
    if out["overall_score"] != sc["overall"]:
        errs.append("overall_score does not match the input")

    texts = [out["headline"], out["score_summary"]]

    # components
    seen = set()
    for c in out["component_feedback"] if isinstance(out["component_feedback"], list) else []:
        name = c.get("component")
        if name not in COMPONENTS:
            errs.append(f"unknown component: {name}")
            continue
        seen.add(name)
        if c.get("score") != sc[name]:
            errs.append(f"{name} score does not match the input")
        if c.get("status") not in STATUSES:
            errs.append(f"{name} has an invalid status")
        if (c.get("score") is None) != (c.get("status") == "unavailable"):
            errs.append(f"{name}: null score and 'unavailable' status must go together")
        texts.append(c.get("summary", ""))
    if seen != set(COMPONENTS):
        errs.append("component_feedback must cover position, timing, sharpness and holds")

    # strengths
    for s in out["strengths"] if isinstance(out["strengths"], list) else []:
        if not isinstance(s, dict) or not isinstance(s.get("title"), str) or not isinstance(s.get("detail"), str):
            errs.append("a strength needs a title and a detail")
        else:
            texts += [s["title"], s["detail"]]

    # priority fixes must map to real findings
    fixes = out["priority_fixes"]
    if not isinstance(fixes, list):
        errs.append("priority_fixes must be a list")
        fixes = []
    if len(fixes) > 3:
        errs.append("more than 3 priority fixes")
    if not inp["top_findings"] and fixes:
        errs.append("priority fixes given but there are no findings")
    for f in fixes:
        ok = any(normalise_part(f.get("body_part")) == normalise_part(t["body_part"])
                 and f.get("section") == t["section"] and _is_num(f.get("timestamp_s"))
                 and abs(f["timestamp_s"] - t["timestamp_s"]) < 0.05
                 for t in inp["top_findings"])
        if not ok:
            errs.append(f"priority fix '{f.get('title')}' does not match any measured finding")
        for k in ("title", "what_happened", "evidence", "drill", "focus_cue"):
            if not isinstance(f.get(k), str):
                errs.append(f"priority fix missing text: {k}")
            else:
                texts.append(f[k])

    # practice plan and caveat
    p = out["next_practice_plan"]
    if not isinstance(p, dict) or not all(isinstance(p.get(k), str) for k in ("first", "second", "success_target")):
        errs.append("next_practice_plan needs first, second and success_target")
    else:
        texts += [p["first"], p["second"], p["success_target"]]
    if out["data_caveat"] is not None:
        if not isinstance(out["data_caveat"], str):
            errs.append("data_caveat must be null or text")
        else:
            texts.append(out["data_caveat"])

    # body parts
    bps = inp["body_part_scores"]
    scored = []
    for b in out["body_part_feedback"] if isinstance(out["body_part_feedback"], list) else []:
        name = b.get("body_part")
        if name not in bps:
            errs.append(f"unknown body part: {name}")
            continue
        if b.get("score") != bps[name]["score"]:
            errs.append(f"{name} score does not match the input")
        if b.get("status") not in STATUSES or (b.get("score") is None) != (b.get("status") == "unavailable"):
            errs.append(f"{name} has an invalid status")
        if b.get("score") is not None:
            scored.append(b["score"])
        texts.append(b.get("summary", ""))
    if scored != sorted(scored):
        errs.append("body_part_feedback must be ordered from lowest to highest score")

    # no numbers that are not in the input
    if config.LLM_STRICT_NUMBERS:
        allowed = _numbers(json.dumps(inp)) | {0.0, 1.0, 2.0, 3.0, 100.0}
        stray = set()
        for t in texts:
            stray |= {n for n in _numbers(t) if n not in allowed}
        if stray:
            errs.append(f"text contains numbers that are not in the input: {sorted(stray)}")
    return errs


# ====================================================================== 4. fallback
def _conf_word(c):
    return "high" if c >= 0.85 else "medium" if c >= 0.7 else "low"


_COMP_TEXT = {
    "position": ("Your overall body positions are close to the reference.",
                 "Your body positions differed from the reference in places.",
                 "There was not enough reliable data to score position."),
    "timing": ("Your movements reached their final positions about when the reference did.",
               "Some movements reach their final positions later than the reference.",
               "No clear timing events could be measured."),
    "sharpness": ("Your movements took about as long to complete as the reference.",
                  "Some movements take longer to complete than the reference.",
                  "No clear movements could be measured for sharpness."),
    "holds": ("You generally hold positions for about the same time as the reference.",
              "Your hold lengths differ from the reference.",
              "No clear holds could be measured."),
}


def _fix_text(f):
    bp, sec, mv = pretty(f["body_part"]), f["section"], f["measured_value"]
    conf = f"{mv}; {_conf_word(f['confidence'])} confidence."
    t = f["type"]
    if t == "position":
        tail = f["direction"].replace(f"user {bp} ", "", 1)
        return {"title": f"{bp.capitalize()} positioning",
                "what_happened": f"Your {bp} was {tail} than the reference.",
                "evidence": conf,
                "drill": f"Repeat section {sec} slowly and focus on matching your {bp} to the reference.",
                "focus_cue": f"Match the final {bp} shape before moving on."}
    if t == "timing":
        late = " later " in f" {mv} "
        return {"title": "Arrive on time" if late else "Hold back for the arrival",
                "what_happened": f"You reached the target position {mv.replace('than reference', 'than the reference')}.",
                "evidence": conf,
                "drill": (f"Loop the transition in section {sec} and aim to reach the final position with the reference."
                          if late else
                          f"Loop the transition in section {sec} and wait to arrive together with the reference."),
                "focus_cue": "Arrive with the choreographer, not after them." if late else "Arrive with the choreographer, not before them."}
    if t == "sharpness":
        longer = " longer " in f" {mv} "
        return {"title": "Movement speed",
                "what_happened": f"Your {bp} {mv.replace('than reference', 'than the reference')}.",
                "evidence": conf,
                "drill": (f"Practise the {bp} move in section {sec} in short bursts, aiming to finish it in the reference time."
                          if longer else
                          f"Practise the {bp} move in section {sec} a little slower, matching the reference duration."),
                "focus_cue": "Finish the move when the reference finishes."}
    return {"title": "Hold length",
            "what_happened": f"Your {bp} {mv.replace('than reference', 'than the reference')}.",
            "evidence": conf,
            "drill": f"Count the hold in section {sec} and stay in the pose until the reference moves.",
            "focus_cue": "Stay in the pose until the reference moves."}


def fallback_feedback(inp):
    """Deterministic template feedback in the exact output schema. Real numbers, no model."""
    sc, finds, dq = inp["scores"], inp["top_findings"], inp["data_quality"]
    comp = []
    for c in COMPONENTS:
        st = status_for(sc[c])
        txt = _COMP_TEXT[c][2 if st == "unavailable" else 0 if st in ("strong", "good") else 1]
        comp.append({"component": c, "score": sc[c], "status": st, "summary": txt})

    fixes = []
    for rank, f in enumerate(finds[:3], 1):
        fixes.append({"rank": rank, "body_part": f["body_part"], "section": f["section"],
                      "timestamp_s": f["timestamp_s"], **_fix_text(f)})

    strong = sorted((c for c in comp if c["status"] in ("strong", "good")),
                    key=lambda c: -c["score"])[:2]
    strengths = [{"title": f"Strong {c['component']}", "detail": c["summary"]} for c in strong]

    o = sc["overall"]
    if o is None:
        headline = "Not enough reliable tracking for a confident overall score."
        summary = "Some of the routine could not be tracked well enough to score it."
    else:
        lead = "Great match" if o >= 85 else "Good overall match" if o >= 70 else "Getting there" if o >= 55 else "Early days"
        headline = f"{lead}" + (f" - focus on {fixes[0]['title'].lower()}." if fixes else ".")
        summary = (f"Your overall score is {o} out of 100."
                   + (" The clearest difference from the reference is listed below." if fixes else ""))

    first = fixes[0]["drill"] if fixes else "Run the routine again to confirm these results."
    second = fixes[1]["drill"] if len(fixes) > 1 else "Pick the section that felt hardest and repeat it slowly."
    target = (f"Improve the section {fixes[0]['section']} {finds[0]['type']} score on the next attempt."
              if fixes else "Keep every section close to the reference on the next attempt.")

    parts = []
    for name, d in inp["body_part_scores"].items():
        st = status_for(d["score"])
        txt = ("There was not enough reliable data to score this body part."
               if st == "unavailable" else
               f"Your {pretty(name)} closely matched the reference." if st in ("strong", "good") else
               f"Your {pretty(name)} position differed more from the reference.")
        parts.append({"body_part": name, "score": d["score"], "status": st, "summary": txt})
    parts.sort(key=lambda p: (p["score"] is None, p["score"] if p["score"] is not None else 0))

    caveats = []
    if dq["excluded_sections"]:
        caveats.append("Sections " + ", ".join(str(s) for s in dq["excluded_sections"])
                       + " were excluded (paused or not tracked well enough).")
    if dq.get("unavailable_components"):
        caveats.append("These could not be measured: " + ", ".join(dq["unavailable_components"]) + ".")
    if dq["tracked_frame_percentage"] < 90:
        caveats.append(f"Only {dq['tracked_frame_percentage']} percent of frames could be tracked.")

    return {"headline": headline, "overall_score": o, "score_scale": "0-100",
            "score_summary": summary, "component_feedback": comp, "strengths": strengths,
            "priority_fixes": fixes,
            "next_practice_plan": {"first": first, "second": second, "success_target": target},
            "body_part_feedback": parts,
            "data_caveat": " ".join(caveats) if caveats else None}


# ====================================================================== 2. the call
def _call_anthropic(inp):
    import anthropic                      # imported here so the rest works without the package
    client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S, max_retries=0)
    msg = client.messages.create(
        model=config.LLM_MODEL, max_tokens=config.LLM_MAX_TOKENS, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(inp)}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _parse(raw):
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    return json.loads(s)


def get_feedback(inp, call_fn=None):
    """Returns (feedback_json, source) where source is 'llm' or 'fallback'.
    Never raises: any timeout, API error, bad JSON or failed validation gives the fallback."""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(call_fn or _call_anthropic, inp)
        out = _parse(fut.result(timeout=config.LLM_TIMEOUT_S))
        if not validate_llm_output(out, inp):
            return out, "llm"
    except Exception:
        pass
    finally:
        ex.shutdown(wait=False)
    return fallback_feedback(inp), "fallback"
