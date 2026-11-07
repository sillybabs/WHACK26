"""config.py  (owned by B)

Every number the scoring code uses lives here, so tuning never means editing logic.
Values marked (spec) come straight from Scoring + LLM Feedback Spec v3.
Everything else is a v1 starting value to tune with real dancers.
"""

# ---------------------------------------------------------------- run structure
SECTION_SECONDS = 4.0          # (spec 9) 4 s sections
BPM = None                     # if set, section length = 8 * 60 / BPM
WINDOW_S = 0.5                 # (spec 3) look-back for the LIVE position score only
MIRROR = True
AUTO_PAUSE = True
RESTART_QUIET_S = 0.5          # (spec 15) no scoring for 0.5 s after a restart

# ---------------------------------------------------------------- position (spec 3)
ERR_AT_ZERO_SCORE = 50.0       # P = clamp(100 * (1 - E / 50), 0, 100)
SCORE_SMOOTH_S = 0.4           # rolling smoothing of the live score
MIN_ACTIVE_JOINTS = 3          # fewer comparable joints than this = frame not scored
JOINT_WEIGHTS = {              # (spec 3) equal to start, configurable
    "left_elbow": 1.0, "right_elbow": 1.0,
    "left_shoulder": 1.0, "right_shoulder": 1.0,
    "left_knee": 1.0, "right_knee": 1.0,
    "left_hip": 1.0, "right_hip": 1.0,
}

# ---------------------------------------------------------------- live bands (spec 9)
GREEN_MIN_SCORE = 80           # green 80-100
AMBER_MIN_SCORE = 60           # amber 60-79, red 0-59
BAND_HOLD_S = 0.3              # a new band must hold this long before it shows
WORST_JOINT_WINDOW_S = 3.0
# Old names, kept so other people's code does not break (degrees of error at the cut)
GREEN_MAX_ERR = ERR_AT_ZERO_SCORE * (1 - GREEN_MIN_SCORE / 100)   # 10 deg
AMBER_MAX_ERR = ERR_AT_ZERO_SCORE * (1 - AMBER_MIN_SCORE / 100)   # 20 deg

# ---------------------------------------------------------------- events (spec 4-6)
TIMING_TOLERANCE_S = 0.50      # T_i = clamp(100 * (1 - dt / 0.5))
SHARPNESS_TOLERANCE_S = 0.50   # S_i
HOLD_TOLERANCE_S = 0.50        # H_i
EVENT_SMOOTH_SAMPLES = 5       # moving-average width before detecting events
EVENT_MAX_GAP_S = 0.20         # a bigger gap in the data splits the series
EVENT_STABLE_TOL_DEG = 5.0     # "stable" = within +-5 deg of where the run started
EVENT_ARRIVAL_STABLE_S = 0.20  # must stay stable this long to count as arrived
EVENT_MIN_MOVE_DEG = 20.0      # smaller changes are not a movement
EVENT_MAX_MOVE_S = 1.5         # slower than this is not one clear movement
HOLD_MIN_S = 0.30              # a stable pose this long is a hold
EVENT_MATCH_WINDOW_S = 1.0     # user event must be this close to a reference event

# ---------------------------------------------------------------- combining (spec 8)
COMPONENT_WEIGHTS = {"position": 0.75, "timing": 0.125, "sharpness": 0.0625, "holds": 0.0625}

# ---------------------------------------------------------------- data quality (spec 8.1, 15)
MIN_POSITION_COVERAGE = 0.50   # below this, no confident overall score
MIN_SECTION_SAMPLES = 5
MIN_SESSION_SAMPLES = 20
MIN_EVENTS_SECTION = 1         # fewer matched events = component unavailable
MIN_EVENTS_SESSION = 2
MIN_PART_SAMPLES = 10          # body-part score needs this many samples (session)
MIN_PART_SAMPLES_SECTION = 3
MIN_REF_TRACKED = 0.95         # warn if less of the reference clip is tracked
LANDMARK_METRIC_ENABLED = False  # wrists / feet / head need landmarks (spec 8.2)

# ---------------------------------------------------------------- findings (spec 11)
FINDINGS_MAX = 6
FINDINGS_MAX_PER_TYPE = 3
FINDING_MIN_LOSS = 15          # ignore issues that cost fewer than this many points
FINDING_MIN_CONFIDENCE = 0.60

# ---------------------------------------------------------------- feedback wording
STATUS_STRONG = 85
STATUS_GOOD = 70
STATUS_IMPROVING = 55

# ---------------------------------------------------------------- LLM (spec 10-13)
LLM_MODEL = "claude-haiku-5-5"
LLM_TIMEOUT_S = 4.0            # see note in the summary: the new JSON is big
LLM_MAX_TOKENS = 1500
LLM_STRICT_NUMBERS = True      # reject replies that contain numbers not in the input
COACH_PERSONALITY = {"kindness": 0.8, "technicality": 0.7, "energy": 0.6, "directness": 0.5}

# ---------------------------------------------------------------- balance bar (unchanged)
BALANCE_BUCKET_S = 0.1
BALANCE_AMBER_VALUE = 0.0
