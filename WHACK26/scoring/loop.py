"""scoring/loop.py  (Person B) - the conductor.

Server calls:   load_clip(path)          when a clip is chosen
                on_snapshot(snapshot)    for every new body snapshot from A  -> score message | None
                finish_run()             when the video ends                 -> report + LLM input
                get_debrief(result)      -> (feedback JSON, 'llm' | 'fallback')
C (flow/) calls: notify_state(name), notify_restart(rewind_to_s=None), pause_hook(section_start_s),
                reset_hook()

Snapshot shape (from A):
  {"video_time": float,
   "angles":  {"left_elbow": 92.0, "right_elbow": None, ...},      # 8 joints, None = not measurable
   "visible": {"torso": True, "left_arm": True, "right_arm": False, "left_leg": True, "right_leg": True}}
Reference frame shape (data/ref/<clip>.json): {"t", "angles": {...}, "visible": {...}}
"""
import json
import threading
import time

import config
from scoring import events, llm_feedback, position
from scoring.balance import BalanceBar
from scoring.common import (error_to_score, mirror_frame, section_index_at, strip_private)
from scoring.live import BandTracker, ScoreSmoother, StreakCounter, WorstJointTracker
from scoring.recorder import RunRecorder

_lock = threading.RLock()

_ref = None
balance = None
recorder = None
_smoother = ScoreSmoother()
_bands = BandTracker()
_streak = StreakCounter()
_worst = WorstJointTracker()

_flow_state = "waiting"
_last_restart = -1e9
_pending_rewind = None


def _now():
    return time.monotonic()


# ------------------------------------------------------------------ loading
def load_clip(path):
    """Load data/ref/<clip>.json. Returns reference quality so the chooser can warn (spec 15)."""
    with open(path) as f:
        data = json.load(f)
    frames = data["frames"] if isinstance(data, dict) else data
    return on_clip_loaded(frames)


def on_clip_loaded(frames):
    global _ref, balance, recorder
    with _lock:
        _ref = position.Reference(frames)
        _ref.events = events.detect_reference(_ref.frames)
        balance = BalanceBar(_ref.duration or 1.0)
        recorder = RunRecorder(_ref)
        _reset_live_state()
    return position.reference_quality(frames)


def _reset_live_state():
    for t in (_smoother, _bands, _streak, _worst):
        t.reset()


# ------------------------------------------------------------------ hooks for C (flow/)
def notify_state(name):
    """'playing', 'paused', 'calibrating', ... Scoring only runs while 'playing'."""
    global _flow_state
    _flow_state = name


def pause_hook(section_start_s):
    """On pause: the interrupted section is excluded from the session score (spec 9, 15)."""
    global _pending_rewind
    with _lock:
        if recorder is None:
            return
        recorder.mark_interrupted(section_index_at(section_start_s))
        _truncate_from(section_start_s)
        _pending_rewind = section_start_s
        _reset_live_state()


def notify_restart(rewind_to_s=None):
    """When the video restarts. Pass the time it rewound to if it went back further than the
    paused section's start (pause in the first second -> previous section)."""
    global _last_restart, _pending_rewind
    with _lock:
        target = rewind_to_s if rewind_to_s is not None else _pending_rewind
        if target is not None and recorder is not None:
            _truncate_from(target)
        _pending_rewind = None
        _last_restart = _now()
        _reset_live_state()


def reset_hook():
    """Try Again / Reset."""
    global _pending_rewind
    with _lock:
        if balance is not None:
            balance.reset()
        if recorder is not None:
            recorder.reset()
        _pending_rewind = None
        _reset_live_state()


def _truncate_from(t):
    recorder.truncate_from(t)
    balance.clear_from(t)


def is_scoring_allowed():
    """No scoring in calibration, countdown, pause, or the first 0.5 s after a restart."""
    return _flow_state == "playing" and (_now() - _last_restart) > config.RESTART_QUIET_S


# ------------------------------------------------------------------ live scoring
def on_snapshot(snapshot):
    """Returns the `score` message dict, or None (send nothing this frame)."""
    if _ref is None or recorder is None or not is_scoring_allowed():
        return None
    t = snapshot["video_time"]
    window = _ref.window(t, config.WINDOW_S)
    if not window:
        return None
    if config.MIRROR:                          # read every frame so a mid-run toggle just works
        window = [mirror_frame(f) for f in window]

    m = position.best_match(snapshot["angles"], snapshot["visible"], window)
    with _lock:
        if m is None:                          # nothing comparable: counts against coverage
            recorder.add_untracked(t)
            return None
        now = _now()
        p_raw = error_to_score(m["error"])
        score = _smoother.update(p_raw, now)
        band = _bands.update(score, now)
        worst = _worst.update(m["abs"], now)
        streak = _streak.update(band, now)
        recorder.add_sample(t, m["matched_t"], p_raw, m["error"], m["abs"], m["signed"],
                            {j: snapshot["angles"][j] for j in m["active"]}, band)
        balance.record(m["matched_t"], band)
        bal = balance.value

    return {
        "type": "score",
        "score": round(score, 1),              # live POSITION score (timing etc. need whole movements)
        "band": band,
        "streak": round(streak, 1),
        "weakest_joint": worst,
        "missing_parts": [p for p, ok in snapshot["visible"].items() if not ok],
        "section": section_index_at(t) + 1,    # 1-based, like the report
        "balance": round(bal, 4),
        "joint_scores": {j: round(error_to_score(e), 1) for j, e in m["abs"].items()},
        "debug": {"error": round(m["error"], 2), "per_joint": m["abs"]},
    }


# ------------------------------------------------------------------ end of run
def finish_run(personality=None):
    """Call when the video ends. Returns {"report": ..., "llm_input": ...}, both plain JSON."""
    with _lock:
        report = strip_private(recorder.build_report())
    return {"report": report, "llm_input": llm_feedback.build_llm_input(report, personality)}


def get_debrief(result, call_fn=None):
    """Ask the model (4 s limit), validate, or fall back. Returns (feedback_json, source)."""
    return llm_feedback.get_feedback(result["llm_input"], call_fn)
