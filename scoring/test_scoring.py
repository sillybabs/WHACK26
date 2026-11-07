"""tests/test_scoring.py - the spec's acceptance table (section 17), on synthetic dancers.
Run from the project root:  python tests/test_scoring.py   (or pytest)"""
import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import WHACK26.config as config
from scoring import loop, llm_feedback
from scoring.common import JOINTS, swap_lr
from scoring.recorder import combine

FPS, DUR = 25, 20.0
PARTS = ["torso", "left_arm", "right_arm", "left_leg", "right_leg"]


def _smooth(x):
    return x * x * (3 - 2 * x)


def level(k, ji):
    return 160.0 if (k + ji) % 2 == 0 else 70.0


def wave(t, ji, start_off=0.0, move=0.3):
    """Hold a pose, move to the next one every second. Transition k spans
    [k + start_off, k + start_off + move]; the reference uses start_off=0, move=0.3."""
    k = int((t - start_off) // 1.0)
    x = (t - start_off) - k
    if x < move:
        return level(k - 1, ji) + (level(k, ji) - level(k - 1, ji)) * _smooth(x / move)
    return level(k, ji)


def ref_frames():
    return [{"t": i / FPS,
             "angles": {j: wave(i / FPS, ji) for ji, j in enumerate(JOINTS)},
             "visible": {p: True for p in PARTS}} for i in range(int(DUR * FPS) + 1)]


def user_angles(t, noise, rnd, delay=0.0, start_off=0.0, move=0.3, still=None, mirror=False):
    out = {}
    for ji, j in enumerate(JOINTS):
        if still is not None:
            v = still
        else:
            src = JOINTS.index(swap_lr(j)) if mirror else ji
            v = wave(t - delay, src, start_off, move)
        out[j] = v + rnd.uniform(-noise, noise)
    return out


class Sim:
    def __init__(self, mirror=False):
        config.MIRROR = mirror
        self.clock = {"t": 1000.0}
        loop._now = lambda: self.clock["t"]
        loop.on_clip_loaded(ref_frames())
        loop.reset_hook()
        loop.notify_state("playing")
        loop.notify_restart()
        self.msgs = []

    def feed(self, t0, t1, hidden=(), **kw):
        rnd = random.Random(1)
        noise = kw.pop("noise", 1.0)
        n0, n1 = int(t0 * FPS), int(t1 * FPS)
        for i in range(n0, n1):
            t = i / FPS
            self.clock["t"] += 1 / FPS
            vis = {p: (p not in hidden) for p in PARTS}
            ang = user_angles(t, noise, rnd, **kw)
            for j in JOINTS:
                if not vis.get(j.replace("elbow", "arm").replace("shoulder", "arm")
                               .replace("knee", "leg").replace("hip", "leg"), True):
                    ang[j] = None
            m = loop.on_snapshot({"video_time": t, "angles": ang, "visible": vis})
            if m:
                self.msgs.append(m)

    def run(self, **kw):
        self.feed(0, DUR, **kw)
        return loop.finish_run()


def bands(msgs):
    b = [m["band"] for m in msgs]
    return {k: b.count(k) / len(b) for k in ("green", "amber", "red")}


def test_combine_matches_spec_examples():
    o, miss = combine({"position": 80, "timing": 60, "sharpness": 70, "holds": 90})
    assert abs(o - 77.5) < 1e-9 and miss == []                      # spec 8 example
    o, miss = combine({"position": 80, "timing": 60, "sharpness": 70, "holds": None})
    assert abs(o - (0.75 * 80 + 0.125 * 60 + 0.0625 * 70) / 0.9375) < 1e-9
    assert miss == ["holds"]                                         # renormalised, not H=100
    assert combine({"position": None, "timing": 90, "sharpness": 90, "holds": 90})[0] is None


def test_close_copy_vs_still():
    good = Sim().run(noise=2.0)
    g_pos = good["report"]["scores"]["position"]
    gm = Sim()
    gm.run(noise=2.0)
    assert bands(gm.msgs)["green"] > 0.8, bands(gm.msgs)
    still = Sim()
    s = still.run(still=115.0)
    s_pos = s["report"]["scores"]["position"]
    assert bands(still.msgs)["red"] > 0.6, bands(still.msgs)
    assert g_pos > 85 and s_pos < 65 and g_pos - s_pos > 30, (g_pos, s_pos)
    print("close copy position", round(g_pos, 1), "| still", round(s_pos, 1))


def test_late_user_timing_shows_lateness_but_position_tolerates_it():
    base = Sim().run(noise=1.0)["report"]["scores"]
    late = Sim().run(noise=1.0, delay=0.25)["report"]
    sc = late["scores"]
    assert sc["position"] > 85, sc                       # best-match window absorbs the lag
    assert sc["timing"] is not None and sc["timing"] < base["timing"] - 20, (sc, base)
    offs = [s["arrival_offset_ms"] for s in late["sections"] if s["arrival_offset_ms"] is not None]
    assert offs and all(180 <= o <= 320 for o in offs), offs
    print("late timing", round(sc["timing"], 1), "vs on-time", round(base["timing"], 1), "offsets", offs)


def test_slow_mover_lowers_sharpness():
    base = Sim().run(noise=1.0)["report"]["scores"]
    slow = Sim().run(noise=1.0, move=0.7)["report"]["scores"]
    assert slow["sharpness"] < base["sharpness"] - 15, (slow, base)
    print("sharpness slow", round(slow["sharpness"], 1), "vs", round(base["sharpness"], 1))


def test_leaving_pose_early_lowers_holds():
    base = Sim().run(noise=1.0)["report"]["scores"]
    early = Sim().run(noise=1.0, start_off=-0.3, move=0.6)["report"]["scores"]
    assert early["holds"] < base["holds"] - 15, (early, base)
    print("holds early", round(early["holds"], 1), "vs", round(base["holds"], 1))


def test_hidden_arm_is_unavailable_not_penalised():
    r = Sim().run(noise=1.0, hidden=("left_arm",))["report"]
    bp = r["body_part_scores"]
    for j in ("left_elbow", "left_shoulder"):
        assert bp[j]["score"] is None and bp[j]["metric"] == "unavailable", bp[j]
    assert bp["right_elbow"]["score"] is not None
    assert r["scores"]["position"] > 85
    for p in ("left_wrist", "right_wrist", "left_foot", "right_foot", "head"):
        assert bp[p]["score"] is None                    # spec 8.2: not implemented -> unavailable


def test_interrupted_section_is_excluded():
    s = Sim()
    s.feed(0, 5.5, noise=1.0)
    loop.notify_state("paused")
    loop.pause_hook(4.0)
    s.clock["t"] += 2
    loop.notify_state("playing")
    loop.notify_restart()
    s.feed(4.0, DUR, noise=1.0)
    res = loop.finish_run()
    dq = res["report"]["data_quality"]
    assert 2 in dq["excluded_sections"] and dq["excluded_reasons"]["2"] == "interrupted", dq
    assert 2 not in dq["valid_sections"] and res["report"]["scores"]["overall"] is not None
    fb, src = loop.get_debrief(res, call_fn=lambda inp: "not json")
    assert src == "fallback" and "2" in fb["data_caveat"], fb["data_caveat"]


def test_mirror_mapping():
    ok = Sim(mirror=True).run(noise=1.0, mirror=True)["report"]["scores"]["position"]
    bad = Sim(mirror=True).run(noise=1.0, mirror=False)["report"]["scores"]["position"]
    assert ok > 85 and bad < ok - 20, (ok, bad)
    config.MIRROR = False


def test_llm_validator_and_fallback():
    config.MIRROR = False
    res = Sim().run(noise=1.0, delay=0.25, hidden=("right_arm",))
    inp = res["llm_input"]
    json.dumps(inp)                                      # must be plain JSON
    fb = llm_feedback.fallback_feedback(inp)
    assert llm_feedback.validate_llm_output(fb, inp) == [], llm_feedback.validate_llm_output(fb, inp)
    scored = [b["score"] for b in fb["body_part_feedback"] if b["score"] is not None]
    assert scored == sorted(scored)
    assert len(fb["priority_fixes"]) <= 3

    bad = copy.deepcopy(fb)                              # invented score
    bad["overall_score"] = 99
    assert llm_feedback.validate_llm_output(bad, inp)
    if bad["component_feedback"]:
        bad = copy.deepcopy(fb)
        bad["component_feedback"][0]["score"] = 12
        assert llm_feedback.validate_llm_output(bad, inp)
    if fb["priority_fixes"]:
        bad = copy.deepcopy(fb)                          # invented timestamp
        bad["priority_fixes"][0]["timestamp_s"] = 77.7
        assert llm_feedback.validate_llm_output(bad, inp)
        bad = copy.deepcopy(fb)                          # invented body part
        bad["priority_fixes"][0]["body_part"] = "head"
        assert llm_feedback.validate_llm_output(bad, inp)
        bad = copy.deepcopy(fb)                          # invented number in prose
        bad["priority_fixes"][0]["evidence"] = "A 313 ms delay."
        assert llm_feedback.validate_llm_output(bad, inp)

    out, src = llm_feedback.get_feedback(inp, call_fn=lambda i: json.dumps(fb))
    assert src == "llm" and out == fb
    _, src = llm_feedback.get_feedback(inp, call_fn=lambda i: "oops")
    assert src == "fallback"
    _, src = llm_feedback.get_feedback(inp, call_fn=lambda i: (_ for _ in ()).throw(RuntimeError("wifi")))
    assert src == "fallback"


def test_no_events_means_unavailable_not_100():
    r = Sim().run(still=115.0)["report"]
    sc = r["scores"]
    # a still person never completes the movements, so timing/sharpness/holds are not invented
    assert sc["timing"] is None or sc["timing"] < 100
    if sc["timing"] is None:
        assert "timing" in r["unavailable_components"]


def test_score_message_shape():
    s = Sim()
    s.feed(0, 3, noise=1.0)
    m = s.msgs[-1]
    for k in ("type", "score", "band", "streak", "weakest_joint", "missing_parts", "section",
              "balance", "joint_scores"):
        assert k in m, k
    assert 0 <= m["score"] <= 100 and 0 <= m["balance"] <= 1 and m["section"] == 1
    assert len(s.msgs) < 3 * FPS                          # quiet window skipped the first 0.5 s


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            failed += 1
            import traceback
            print("FAIL", fn.__name__, "->", repr(e))
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
