"""scoring/loop.py  (Person B)

Takes one body snapshot from A, compares it to the reference clip, and returns
the `score` message dict (or None if nothing should be sent this frame).

Assumed data shapes (rename keys below if A's / the prep script's differ):

  snapshot = {
      "video_time": float,                 # seconds, from the page's video clock
      "angles":  {"l_elbow": 92.0, "r_elbow": None, ...},   # None = not measurable
      "visible": {"torso": True, "l_arm": True, "r_arm": False, "l_leg": True, "r_leg": True},
  }

  ref frame (one entry of data/ref/<clip>.json) = {
      "t": float,                          # seconds into the clip
      "angles":  {...same 8 keys...},
      "visible": {...same 5 keys...},
  }

config.py needs:
  WINDOW_S, MIRROR, GREEN_MAX_ERR, AMBER_MAX_ERR, SECTION_SECONDS,
  JOINT_WEIGHTS (dict, optional), BALANCE_BUCKET_S, BALANCE_AMBER_VALUE
Optional extras (defaults used if missing):
  SCORE_SMOOTH_S (0.4), BAND_HOLD_S (0.3), WORST_JOINT_WINDOW_S (3.0),
  RESTART_QUIET_S (0.5), ERR_AT_ZERO_SCORE (50.0)
"""
import bisect
import json
import threading
import time
from collections import deque

import config
from balance import BalanceBar

# ----------------------------------------------------------------------------
# Constants / settings
# ----------------------------------------------------------------------------
JOINTS = ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder",
          "l_knee", "r_knee", "l_hip", "r_hip"]

# Which visibility flag each joint depends on
JOINT_PART = {
    "l_elbow": "l_arm", "l_shoulder": "l_arm",
    "r_elbow": "r_arm", "r_shoulder": "r_arm",
    "l_knee": "l_leg", "l_hip": "l_leg",
    "r_knee": "r_leg", "r_hip": "r_leg",
}

SCORE_SMOOTH_S = getattr(config, "SCORE_SMOOTH_S", 0.4)
BAND_HOLD_S = getattr(config, "BAND_HOLD_S", 0.3)
WORST_WINDOW_S = getattr(config, "WORST_JOINT_WINDOW_S", 3.0)
RESTART_QUIET_S = getattr(config, "RESTART_QUIET_S", 0.5)
ERR_AT_ZERO = getattr(config, "ERR_AT_ZERO_SCORE", 50.0)
WEIGHTS = getattr(config, "JOINT_WEIGHTS", {})   # missing joint -> 1.0


def _now() -> float:
    return time.monotonic()


# ----------------------------------------------------------------------------
# Module state (one run at a time)
# ----------------------------------------------------------------------------
_lock = threading.Lock()

balance = None            # BalanceBar, created when a clip is loaded
_ref_frames = []          # list of ref frames, sorted by t
_ref_times = []           # just the t values, for bisect

_flow_state = "waiting"   # set by C via notify_state()
_last_restart = -1e9      # set by C via notify_restart()

_score_hist = deque()     # (time, raw_score) for smoothing
_worst_hist = deque()     # (time, per_joint dict)
_band = "red"             # band currently shown
_cand_band = None         # band that wants to take over
_cand_since = 0.0
_streak_s = 0.0
_streak_last_t = None


# ----------------------------------------------------------------------------
# Hooks for C (flow/) to call
# ----------------------------------------------------------------------------
def notify_state(name: str) -> None:
    """C calls this whenever the stage changes ('playing', 'paused', ...)."""
    global _flow_state
    _flow_state = name


def notify_restart() -> None:
    """C calls this when the video restarts after a pause (starts the quiet window)."""
    global _last_restart
    _last_restart = _now()


def pause_hook(section_start_s: float) -> None:
    """C calls this on pause: interrupted section does not count against the dancer."""
    if balance is not None:
        balance.clear_from(section_start_s)
    _reset_live_state()


def reset_hook() -> None:
    """C calls this on Try Again / Reset."""
    if balance is not None:
        balance.reset()
    _reset_live_state()


def _reset_live_state() -> None:
    """Clear smoothing / band / streak so a restart does not inherit old values."""
    global _band, _cand_band, _cand_since, _streak_s, _streak_last_t
    with _lock:
        _score_hist.clear()
        _worst_hist.clear()
        _band = "red"
        _cand_band = None
        _cand_since = 0.0
        _streak_s = 0.0
        _streak_last_t = None


# ----------------------------------------------------------------------------
# Loading the reference clip
# ----------------------------------------------------------------------------
def load_clip(path: str) -> None:
    """Load data/ref/<clip>.json and start a fresh run."""
    with open(path, "r") as f:
        data = json.load(f)
    # Handle both a bare list and {"frames": [...]}
    frames = data["frames"] if isinstance(data, dict) else data
    on_clip_loaded(frames)


def on_clip_loaded(ref_frames: list) -> None:
    global balance, _ref_frames, _ref_times
    _ref_frames = sorted(ref_frames, key=lambda fr: fr["t"])
    _ref_times = [fr["t"] for fr in _ref_frames]
    ref_duration_s = _ref_times[-1] if _ref_times else 1.0
    balance = BalanceBar(ref_duration_s)
    _reset_live_state()


# ----------------------------------------------------------------------------
# Small helpers (sections, quiet moments, reference lookup, mirror)
# ----------------------------------------------------------------------------
def section_index_at(t: float) -> int:
    """Fallback section lookup. Swap for C's flow function once it exists."""
    return int(t // config.SECTION_SECONDS)


def is_scoring_allowed() -> bool:
    """No scoring during calibration, countdown, pause, or just after a restart."""
    if _flow_state != "playing":
        return False
    return (_now() - _last_restart) > RESTART_QUIET_S


def get_ref_window(video_time: float, window_s: float) -> list:
    """Reference frames from the last `window_s` seconds up to `video_time`."""
    lo = bisect.bisect_left(_ref_times, video_time - window_s)
    hi = bisect.bisect_right(_ref_times, video_time)
    return _ref_frames[lo:hi]


def _swap_lr(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


def mirror_frame(frame: dict) -> dict:
    """Swap left and right on a reference frame (angles AND visibility)."""
    return {
        "t": frame["t"],
        "angles": {_swap_lr(k): v for k, v in frame["angles"].items()},
        "visible": {_swap_lr(k): v for k, v in frame["visible"].items()},
    }


# ----------------------------------------------------------------------------
# Scoring steps
# ----------------------------------------------------------------------------
def get_active_joints(user_angles, user_visible, ref_angles, ref_visible) -> list:
    """Joints we can score right now: visible and measured on BOTH sides."""
    active = []
    for j in JOINTS:
        part = JOINT_PART[j]
        if not (user_visible.get(part) and ref_visible.get(part)):
            continue
        if user_angles.get(j) is None or ref_angles.get(j) is None:
            continue
        active.append(j)
    return active


def best_match_error(user_angles, ref_window, active):
    """Compare against every ref frame in the window; keep the smallest weighted
    average error. Absorbs people copying the video a moment late.
    Returns (error, per_joint, matched_ref_t) or None if nothing is comparable."""
    best = None
    for rf in ref_window:
        # Joints must be active for THIS ref frame too
        joints = [j for j in active if rf["angles"].get(j) is not None]
        if not joints:
            continue
        per_joint = {j: abs(user_angles[j] - rf["angles"][j]) for j in joints}
        wsum = sum(WEIGHTS.get(j, 1.0) for j in joints)
        if wsum <= 0:
            continue
        err = sum(per_joint[j] * WEIGHTS.get(j, 1.0) for j in joints) / wsum
        if best is None or err < best[0]:
            best = (err, per_joint, rf["t"])
    return best


def error_to_score(err: float) -> float:
    """0 degrees -> 100, ERR_AT_ZERO (50) degrees -> 0, clamped."""
    return max(0.0, min(100.0, 100.0 * (1.0 - err / ERR_AT_ZERO)))


def smooth_score(raw: float) -> float:
    """Average the raw score over the last SCORE_SMOOTH_S seconds."""
    now = _now()
    _score_hist.append((now, raw))
    while _score_hist and now - _score_hist[0][0] > SCORE_SMOOTH_S:
        _score_hist.popleft()
    return sum(s for _, s in _score_hist) / len(_score_hist)


def _raw_band(score: float) -> str:
    green_cut = error_to_score(config.GREEN_MAX_ERR)   # 20 deg -> 60
    amber_cut = error_to_score(config.AMBER_MAX_ERR)   # 40 deg -> 20
    if score >= green_cut:
        return "green"
    if score >= amber_cut:
        return "amber"
    return "red"


def band_for(score: float) -> str:
    """Green / amber / red, but a new band must hold BAND_HOLD_S before it shows."""
    global _band, _cand_band, _cand_since
    now = _now()
    wanted = _raw_band(score)
    if wanted == _band:
        _cand_band = None
    elif wanted != _cand_band:
        _cand_band = wanted
        _cand_since = now
    elif now - _cand_since >= BAND_HOLD_S:
        _band = wanted
        _cand_band = None
    return _band


def track_worst_joint(per_joint: dict):
    """Running average of error per joint over the last few seconds; return the worst."""
    now = _now()
    _worst_hist.append((now, per_joint))
    while _worst_hist and now - _worst_hist[0][0] > WORST_WINDOW_S:
        _worst_hist.popleft()
    totals, counts = {}, {}
    for _, pj in _worst_hist:
        for j, e in pj.items():
            totals[j] = totals.get(j, 0.0) + e
            counts[j] = counts.get(j, 0) + 1
    if not totals:
        return None
    return max(totals, key=lambda j: totals[j] / counts[j])


def update_streak(band: str) -> float:
    """Seconds of consecutive green. Resets on red; amber pauses it."""
    global _streak_s, _streak_last_t
    now = _now()
    dt = 0.0 if _streak_last_t is None else now - _streak_last_t
    _streak_last_t = now
    if band == "green":
        _streak_s += dt
    elif band == "red":
        _streak_s = 0.0
    return _streak_s


# ----------------------------------------------------------------------------
# The conductor: called by the server on every new body snapshot
# ----------------------------------------------------------------------------
def on_snapshot(snapshot: dict):
    """Returns the `score` message dict, or None (send nothing this frame)."""
    if balance is None or not _ref_frames:
        return None

    # 1. Quiet moments
    if not is_scoring_allowed():
        return None

    video_time = snapshot["video_time"]

    # 2. Reference frames for the last WINDOW_S
    ref_window = get_ref_window(video_time, config.WINDOW_S)
    if not ref_window:
        return None

    # 3. Mirror on the reference, if enabled (read each frame so a mid-run toggle works)
    if config.MIRROR:
        ref_window = [mirror_frame(f) for f in ref_window]

    # 4. Joints visible on both sides (use newest ref frame for the visibility check)
    newest = ref_window[-1]
    active = get_active_joints(snapshot["angles"], snapshot["visible"],
                               newest["angles"], newest["visible"])
    if not active:
        return None

    # 5. Best-match error across the window
    result = best_match_error(snapshot["angles"], ref_window, active)
    if result is None:
        return None
    error, per_joint, matched_ref_t = result

    # 6. Score, band, weakest joint, streak
    with _lock:
        score = smooth_score(error_to_score(error))
        band = band_for(score)
        worst = track_worst_joint(per_joint)
        streak = update_streak(band)

    # 7. Balance bar
    balance.record(matched_ref_t, band)

    # 8. The score message
    return {
        "type": "score",
        "score": round(score, 1),
        "band": band,
        "streak": round(streak, 1),
        "weakest_joint": worst,
        "missing_parts": [p for p, ok in snapshot["visible"].items() if not ok],
        "section": section_index_at(video_time),
        "balance": round(balance.value, 4),
        # handy for A's/D's debug panel
        "debug": {"error": round(error, 2), "per_joint": per_joint},
    }