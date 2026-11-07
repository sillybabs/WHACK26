"""scoring/common.py - names, maths and small helpers shared by every scoring file."""
import math
import WHACK26.config as config

JOINTS = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_knee", "right_knee", "left_hip", "right_hip",
]
# Which visibility flag each joint depends on
JOINT_PART = {
    "left_elbow": "left_arm", "left_shoulder": "left_arm",
    "right_elbow": "right_arm", "right_shoulder": "right_arm",
    "left_knee": "left_leg", "left_hip": "left_leg",
    "right_knee": "right_leg", "right_hip": "right_leg",
}
# No angle exists for these in the 8-angle input (spec 8.2): shown as unavailable
UNAVAILABLE_PARTS = ["left_wrist", "right_wrist", "left_foot", "right_foot", "head"]
COMPONENTS = ["position", "timing", "sharpness", "holds"]


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def linear_score(diff, tolerance):
    """0 difference -> 100, `tolerance` or more -> 0 (spec 3-6)."""
    return clamp(100.0 * (1.0 - diff / tolerance))


def error_to_score(err_deg):
    return linear_score(err_deg, config.ERR_AT_ZERO_SCORE)


def round_half_up(x):
    return int(math.floor(x + 0.5))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def swap_lr(name):
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def mirror_frame(frame):
    """Swap left and right on a reference frame (angles AND visibility)."""
    return {
        "t": frame["t"],
        "angles": {swap_lr(k): v for k, v in frame["angles"].items()},
        "visible": {swap_lr(k): v for k, v in frame["visible"].items()},
    }


# ---- sections (0-based inside the code, 1-based in anything a person or the LLM sees)
def section_seconds():
    return 8 * 60.0 / config.BPM if config.BPM else config.SECTION_SECONDS


def section_index_at(t):
    return max(0, int((t + 1e-9) // section_seconds()))


def section_bounds(idx):
    s = section_seconds()
    return idx * s, (idx + 1) * s


def n_sections(duration):
    return max(1, int(math.ceil(duration / section_seconds() - 1e-9)))


def static_band(score):
    if score >= config.GREEN_MIN_SCORE:
        return "green"
    if score >= config.AMBER_MIN_SCORE:
        return "amber"
    return "red"


def pretty(name):
    return name.replace("_", " ")


def normalise_part(name):
    return str(name).strip().lower().replace(" ", "_")


def joint_direction(joint, signed_err):
    """signed_err = user - reference. Wording stays about the reference, not 'right/wrong'."""
    if joint.endswith("elbow") or joint.endswith("knee"):
        tail = "more bent" if signed_err < 0 else "more extended"
    else:
        tail = "at a smaller angle" if signed_err < 0 else "at a larger angle"
    return f"user {pretty(joint)} {tail}"


def strip_private(obj):
    """Drop keys starting with '_' (internal stats) so a report can be sent as JSON."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj
