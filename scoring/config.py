"""
Tunable settings for the scoring package (moved unchanged from pose_webcam.py).

Edit the numbers here to change how the live scoring behaves.
"""

# =============================================================================
# FRAME SCORING - point ranges stored as variables
# =============================================================================
# Each band covers a range of angle differences (in degrees). At the START of
# the band you get *_SCORE_HIGH points, at the END you get *_SCORE_LOW points,
# and in between the score falls linearly.

# Band 1: 0 -> 20 degrees difference  =>  100 -> 60 points
BAND1_MAX_DIFF = 20
BAND1_SCORE_HIGH = 100
BAND1_SCORE_LOW = 60

# Band 2: 20 -> 40 degrees difference  =>  59 -> 20 points
BAND2_MAX_DIFF = 40
BAND2_SCORE_HIGH = 59
BAND2_SCORE_LOW = 20

# Band 3: 40 -> BAND3_MAX_DIFF degrees difference  =>  19 -> 0 points
# (BAND3_MAX_DIFF = 60 is my assumption; beyond it the score stays at 0)
BAND3_MAX_DIFF = 60
BAND3_SCORE_HIGH = 19
BAND3_SCORE_LOW = 0

# (start_diff, end_diff, score_at_start, score_at_end) for each band
SCORE_BANDS = [
    (0,              BAND1_MAX_DIFF, BAND1_SCORE_HIGH, BAND1_SCORE_LOW),
    (BAND1_MAX_DIFF, BAND2_MAX_DIFF, BAND2_SCORE_HIGH, BAND2_SCORE_LOW),
    (BAND2_MAX_DIFF, BAND3_MAX_DIFF, BAND3_SCORE_HIGH, BAND3_SCORE_LOW),
]

# Display colours (BGR). Used for the skeleton and score text.
COLOR_GOOD = (0, 200, 0)      # score >= BAND1_SCORE_LOW  (green)
COLOR_OK = (0, 165, 255)      # score >= BAND2_SCORE_LOW  (orange)
COLOR_BAD = (0, 0, 255)       # anything lower            (red)


# =============================================================================
# SCORE BAR (chess.com style) - tuning values
# =============================================================================
BALANCE_BUCKET_S = 0.1
BALANCE_AMBER_VALUE = 0.0    # orange counts as neutral; try 0.3 if the bar feels too harsh
BALANCE_SMOOTH_S = 0.4       # smooth the overall score this long before it is banded
# Cut-offs for what pushes the bar. These are the team spec's bands (green 80+,
# amber 60-79, red below 60), NOT the skeleton colours, so the bar reacts to a
# mediocre stretch instead of only to a disaster. (Using the skeleton cut-offs of
# 60 / 20 makes red almost impossible, so the bar could only ever go up.)
BALANCE_GREEN_MIN = 80
BALANCE_AMBER_MIN = 60
# How the bar is scaled:
#   "running"    - divided by the time played SO FAR (but at least BALANCE_WARMUP_S),
#                  so it rises and falls with how you are doing. Ends the same as
#                  "whole_clip" once the video is over.
#   "whole_clip" - divided by the whole video length: steadier but very slow, and it
#                  can only reach full green at the very end.
BALANCE_MODE = "running"
BALANCE_WARMUP_S = 3.0       # "running" only: damps the swings in the first seconds

# =============================================================================
# RUN SCORES - position, timing, sharpness, holds
# =============================================================================
# OVERALL = 0.75 position + 0.125 timing + 0.0625 sharpness + 0.0625 holds. If one
# can't be measured (e.g. no clear holds) it's left out and the others re-weighted.
WEIGHT_POSITION = 0.75
WEIGHT_TIMING = 0.125
WEIGHT_SHARPNESS = 0.0625
WEIGHT_HOLDS = 0.0625

# Events are found in each segment's angle over time: a movement, its arrival, a hold.
EVENT_SMOOTH_S = 0.15        # smooth the angle this long before measuring speed
EVENT_MOVE_DPS = 30.0        # degrees/second above which a segment counts as moving
EVENT_MIN_MOVE_S = 0.10      # a movement must last at least this long
EVENT_MIN_AMPLITUDE = 10.0   # ...and change the angle by at least this many degrees
EVENT_MERGE_GAP_S = 0.10     # movements closer than this are treated as one
EVENT_STABLE_MIN_S = 0.15    # must stay still this long afterwards to count as arrived
EVENT_MATCH_WINDOW_S = 1.0   # pair your arrival with the reference one within this gap
EVENT_MAX_GAP_S = 0.25       # a gap in the data longer than this splits it into pieces
TIMING_TOL_S = 0.50          # arrival 0.5 s apart = 0 points (0 s = 100)
SHARPNESS_TOL_S = 0.50       # movement duration 0.5 s apart = 0 points
HOLD_TOL_S = 0.50            # hold duration 0.5 s apart = 0 points
MIN_EVENTS_PER_COMPONENT = 1 # fewer matched events than this -> "unavailable"
