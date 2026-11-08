"""
pose_compare_video.py - copy a pose from a reference VIDEO and get scored live.

HOW IT WORKS
------------
1. START-UP: the reference .mp4 is read once from start to finish. Every frame
   is run through MediaPipe Pose Landmarker and we store, per frame, the
   skeleton and the angle of 11 body segments (see SEGMENTS below). This takes a
   little while for long videos (a progress % is printed).
2. PLAYBACK: the reference video plays next to your live webcam feed. After a
   short countdown it plays in real time, so you can follow along with it.
3. SCORING: for every webcam frame we work out your 11 live angles and compare
   them with the angles of the reference frame that is on screen at that moment.
   For each segment: difference = how far your angle is from the reference
   angle (always 0-180 degrees, wrap-around handled). That difference is turned
   into a score from 0 to 100 using the point bands below.
4. DISPLAY: live feed on the left, reference video on the right. Your skeleton
   is coloured per segment (green = good, orange = so-so, red = off). Each
   segment's score is written on the body, and a table lists live angle,
   reference angle, difference and score per segment, plus the overall score
   (the average of the segments) and AVG, the average overall score since the
   video (re)started. In THIS version the big SCORE and AVG texts are hidden
   (SHOW_OVERALL_SCORE / SHOW_AVG_SCORE = False); everything else is the same.
5. RUN SCORES: while the reference plays, every scored frame is recorded. When the
   run ends (video finishes / loops, or you quit) the console shows POSITION,
   TIMING, SHARPNESS, HOLDS (each 0-100) and OVERALL - see "RUN SCORES" below.
6. SCORE BAR (hidden in this version: SHOW_SCORE_BAR = False): a vertical bar on
   the right edge of the live feed starts 50% green
   / 50% red, like the evaluation bar on chess.com. Matching the video pushes it
   towards green, missing pushes it towards red, and a perfect run ends fully
   green. See the "SCORE BAR" section for how it is worked out.

ANGLE CONVENTION
----------------
Each segment is a line between two landmarks, ordered so it points "downwards"
when you stand upright (e.g. shoulder -> elbow). Its angle is measured against
straight down, as seen on screen:
    0 deg = pointing straight down,  + = clockwise,  - = anticlockwise.
Because only angle DIFFERENCES matter, the sign convention is just used
consistently for both the live feed and the reference.

SCORING (all tunable in scoring/config.py)
----------------------------------------------
    difference 0-20 deg   -> 100 down to 60 points   (band 1)
    difference 20-40 deg  -> 59 down to 20 points    (band 2)
    difference 40+ deg    -> 19 down to 0 points     (band 3)
Inside a band the score falls in a straight line. Examples:
    10 deg off -> 80,  20 deg -> 60,  30 deg -> 39.5,  40 deg -> 20,  50 deg -> 9.5
Band 3 needs an end point (the brief says "more than 40"), so I assumed it
reaches 0 points at 60 deg difference and stays 0 beyond that. Change
BAND3_MAX_DIFF (in scoring/config.py) if you want a gentler or harsher fall-off.

SETUP
-----
    pip install mediapipe opencv-python

HOW TO RUN (no terminal needed)
-------------------------------
1. Put your reference video (.mp4) in the same folder as this script.
2. Set REFERENCE_VIDEO in the CONFIG section below to the file name
   (or a full path, e.g. "C:/Users/me/Videos/pose.mp4" - use forward slashes).
3. Run the script from your editor (IDLE, Spyder, PyCharm, VS Code "Run" ...).
   From a Jupyter notebook or Python prompt instead:
       import pose_compare_video
       pose_compare_video.main()                  # uses REFERENCE_VIDEO
       pose_compare_video.main("other_video.mp4") # or pass a different video
Relative file names are looked up next to this script, so it doesn't matter
which folder your editor runs it from.

CONTROLS (click the video window first so it has focus)
-------------------------------------------------------
    q or ESC  quit
    SPACE     pause / resume the reference video
    r         restart the reference video (with the countdown); resets the bar

USAGE NOTES
-----------
* The reference video is mirrored when MIRROR = True, exactly like the live
  feed, so you copy it like looking in a mirror: if the person in the video
  raises the arm that appears on the right of the picture, you raise your arm on
  the right of the screen.
* Use a reference filmed from a similar viewpoint (front-on vs front-on). The
  angles are 2D screen angles, so a side-on video won't match a front-on you.
* Segments that aren't visible (below MIN_VISIBILITY) in either the live feed or
  the current reference frame are shown as "--" and left out of the overall
  score. If the reference has no person in a frame, nothing is scored.
* The video's sound is not played.
"""

import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# The scoring code lives in /scoring, next to this /sandbox folder. Make the folder
# above this script importable so "import scoring" works wherever you run from.
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from scoring import (  # noqa: E402
    BalanceBar, RunRecorder, ScoreSmoother, balance_band, color_for_score, compare,
    overall_score,
)
from scoring import finish_run as _finish_run  # noqa: E402
from scoring.config import COLOR_BAD, COLOR_GOOD  # noqa: E402  (used by draw_score_bar)

# =============================================================================
# CONFIG
# =============================================================================
REFERENCE_VIDEO = "reference.mp4"  # <-- your reference video: file name (next to this
                                   #     script) or a full path
LOOP_REFERENCE = True     # replay the video (with a new countdown) when it ends
COUNTDOWN_SECONDS = 3     # get-ready time before the reference starts playing
REF_PROCESS_HEIGHT = 480  # reference frames are shrunk to this height for the
                          # start-up analysis (faster; doesn't change the angles)

CAMERA_INDEX = 0          # try 1 if you have multiple cameras
MODEL_VARIANT = "lite"    # "lite" (fastest), "full", or "heavy" (most accurate)
MIRROR = True             # selfie-style view (applied to the reference too)
MIN_VISIBILITY = 0.5      # ignore landmarks the model isn't confident about
MAX_POSES = 1             # track at most this many people in the live feed
TARGET_FPS = 30           # set to None for uncapped
PRINT_SCORES = True       # also print the scores to the console
PRINT_INTERVAL = 0.5      # seconds between console prints

# =============================================================================
# SCORING SETTINGS
# =============================================================================
# The point bands, colours, score-bar tuning and run-score weights all live in
# scoring/config.py now. Edit them there.

# =============================================================================
# SCORE BAR (chess.com style)
# =============================================================================
# The bar starts 50% green / 50% red. Every moment of the reference video that
# you match (green) pushes it towards green, every moment you miss (red) pushes
# it towards red. A perfect run ends fully green; a run of misses ends fully red.
#
#   bar = 0.5 + 0.5 * (sum of moments / moments counted)   (see BALANCE_MODE)
#   green moment = +1, orange = BALANCE_AMBER_VALUE, red = -1
#
# Moments are stored by REFERENCE VIDEO TIME (0.1 s buckets), so replaying a
# stretch overwrites it instead of adding to it. Pausing never counts.
SHOW_SCORE_BAR = False       # False = the bar is not displayed (set True to bring it back)
SHOW_PART_SCORES = True      # write each segment's score on the body as well as in the table
SHOW_OVERALL_SCORE = False   # the big "SCORE: nn" text at the bottom left of the live feed
SHOW_AVG_SCORE = False       # the "AVG: nn" text above it
# (BALANCE_* tuning values for the bar are in scoring/config.py)

# Optional hook so other code can receive the scores. Set it to a function that
# takes one dict:  {"time", "overall", "parts": {segment: score or None}, "balance"}
# It is called once per frame while the reference is playing and a person is seen.
SCORE_CALLBACK = None

# =============================================================================
# RUN SCORE HOOK
# =============================================================================
# Optional hook: set to a function taking one dict (the run scores above) - called
# once when a run ends.
RUN_SCORE_CALLBACK = None

# =============================================================================
# MODEL
# =============================================================================
# Folder this script lives in (falls back to the current folder in a notebook)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, f"pose_landmarker_{MODEL_VARIANT}.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{MODEL_VARIANT}/float16/latest/"
    f"pose_landmarker_{MODEL_VARIANT}.task"
)

# Standard 33-point BlazePose skeleton connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# =============================================================================
# SEGMENTS - which body parts we measure
# =============================================================================
# Each segment is a line between two landmarks, ordered so it points "downwards"
# when you stand upright. A landmark can be an int, or a tuple of ints meaning
# "midpoint of these". For left/right pairs the right landmark index is the
# left index + 1.
LIMB_SEGMENTS = [
    ("upper arm", 11, 13),   # shoulder -> elbow
    ("forearm",   13, 15),   # elbow -> wrist
    ("torso",     11, 23),   # shoulder -> hip
    ("thigh",     23, 25),   # hip -> knee
    ("shin",      25, 27),   # knee -> ankle (use 31 / 32 for toes instead)
]
HEAD_SEGMENT = ("head", (7, 8), (11, 12))  # ear midpoint -> shoulder midpoint


# =============================================================================
# =============================================================================
# SCORING HOOKS
# =============================================================================
def finish_run(recorder, complete=True):
    """Score the finished run (see scoring/run_scores.py) and call RUN_SCORE_CALLBACK."""
    return _finish_run(recorder, complete, callback=RUN_SCORE_CALLBACK)


# =============================================================================
# GEOMETRY
# =============================================================================
def _point(lms, spec, w, h):
    """Pixel position of a landmark (or midpoint), or None if not visible."""
    idxs = spec if isinstance(spec, tuple) else (spec,)
    xs, ys = [], []
    for i in idxs:
        lm = lms[i]
        if lm.visibility is not None and lm.visibility < MIN_VISIBILITY:
            return None
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _angle(p, q):
    """Angle of p->q vs straight down on screen; clockwise is positive."""
    dx, dy = q[0] - p[0], q[1] - p[1]  # image y points down
    return math.degrees(math.atan2(-dx, dy))


def get_segments(lms, w, h):
    """Return {segment name: (start_point, end_point)} in pixels (None if hidden)."""
    segs = {}
    # The frame is mirrored before detection, so the model's "left" landmarks
    # are your right side when MIRROR is on. Swap so labels match your body.
    sides = (("L", 1 if MIRROR else 0), ("R", 0 if MIRROR else 1))
    for label, offset in sides:
        for name, a, b in LIMB_SEGMENTS:
            segs[f"{label} {name}"] = (
                _point(lms, a + offset, w, h),
                _point(lms, b + offset, w, h),
            )
    name, a, b = HEAD_SEGMENT
    segs[name] = (_point(lms, a, w, h), _point(lms, b, w, h))
    return segs


def segment_angles(segs):
    """{segment name: angle in degrees, or None if not visible}."""
    return {
        name: (_angle(p, q) if p and q else None)
        for name, (p, q) in segs.items()
    }



# =============================================================================
# DRAWING
# =============================================================================
def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        visible = (lm.visibility is None) or (lm.visibility >= MIN_VISIBILITY)
        pts.append((int(lm.x * w), int(lm.y * h)) if visible else None)

    for a, b in POSE_CONNECTIONS:
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_segment_colors(frame, segs, results):
    """Re-draw each measured segment in green / orange / red by its score."""
    for name, r in results.items():
        if r is None:
            continue
        p, q = segs[name]
        cv2.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                 color_for_score(r["score"]), 5, cv2.LINE_AA)


def _text(frame, s, pos, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(frame, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def draw_part_scores(frame, segs, results):
    """Write each segment's score on the body, halfway along the segment."""
    for name, r in results.items():
        if r is None:
            continue
        p, q = segs[name]
        pos = (int((p[0] + q[0]) / 2) + 6, int((p[1] + q[1]) / 2))
        s = f"{r['score']:.0f}"
        _text(frame, s, pos, (0, 0, 0), 0.55, 4)                      # dark outline
        _text(frame, s, pos, color_for_score(r["score"]), 0.55, 2)    # coloured number


def draw_score_bar(frame, value):
    """Vertical green/red bar on the right edge. value = green share, 0..1."""
    h, w = frame.shape[:2]
    bar_w = 26
    x1 = w - 14
    x0 = x1 - bar_w
    y0, y1 = 70, h - 70
    fill_y = int(y1 - (y1 - y0) * value)

    cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_BAD, -1)         # red = misses
    cv2.rectangle(frame, (x0, fill_y), (x1, y1), COLOR_GOOD, -1)    # green rises from the bottom
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    mid = (y0 + y1) // 2
    cv2.line(frame, (x0 - 5, mid), (x1 + 5, mid), (255, 255, 255), 2)  # the 50/50 line

    # The percentage is written as a number too, so it never relies on colour alone
    label = f"{value * 100:.0f}%"
    _text(frame, label, (w - 76, y0 - 12), (0, 0, 0), 0.7, 4)
    _text(frame, label, (w - 76, y0 - 12), (255, 255, 255), 0.7, 2)


def draw_panel(frame, results, overall):
    """Table of live angle / reference angle / difference / score per segment."""
    x0, y0, row_h = 5, 45, 21
    rows = len(results) + 1
    box_w, box_h = 340, row_h * rows + 10

    # Semi-transparent dark background so the text is readable
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cols = [x0 + 6, x0 + 115, x0 + 170, x0 + 225, x0 + 285]
    y = y0 + row_h
    for label, x in zip(("segment", "live", "ref", "diff", "score"), cols):
        _text(frame, label, (x, y), (200, 200, 200), 0.45)

    for name, r in results.items():
        y += row_h
        if r is None:
            _text(frame, name, (cols[0], y))
            _text(frame, "--", (cols[1], y), (150, 150, 150))
            continue
        col = color_for_score(r["score"])
        _text(frame, name, (cols[0], y))
        _text(frame, f"{r['live']:+.0f}", (cols[1], y))
        _text(frame, f"{r['ref']:+.0f}", (cols[2], y))
        _text(frame, f"{r['diff']:+.0f}", (cols[3], y))
        _text(frame, f"{r['score']:.0f}", (cols[4], y), col, 0.55, 2)

    if SHOW_OVERALL_SCORE and overall is not None:
        h = frame.shape[0]
        _text(frame, f"SCORE: {overall:.0f}", (10, h - 15),
              color_for_score(overall), 1.0, 3)


# =============================================================================
# SETUP HELPERS
# =============================================================================
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {os.path.basename(MODEL_PATH)} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def resolve_path(path):
    """Relative paths are looked for next to this script first."""
    if not os.path.isabs(path):
        candidate = os.path.join(BASE_DIR, path)
        if os.path.exists(candidate):
            return candidate
    return path


# =============================================================================
# REFERENCE VIDEO
# =============================================================================
def analyse_reference(path):
    """Run the whole reference video through the pose model once.

    Returns (data, fps) where data[i] = (pose, angles) for frame i:
        pose   - the 33 landmarks (for drawing), or None if nobody was found
        angles - {segment name: angle}, or None if nobody was found
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open the reference video: {path}\n"
                 f"Check REFERENCE_VIDEO at the top of the script.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:  # missing / NaN / nonsense
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print(f"Analysing reference video ({total or '?'} frames, {fps:.1f} fps) ...")
    data = []
    last_ts = -1
    next_report = 10
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)  # same treatment as the live feed
            if frame.shape[0] > REF_PROCESS_HEIGHT:
                scale = REF_PROCESS_HEIGHT / frame.shape[0]
                frame = cv2.resize(
                    frame, (max(1, int(frame.shape[1] * scale)), REF_PROCESS_HEIGHT))
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int(len(data) * 1000 / fps), last_ts + 1)  # must increase
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                data.append((pose, segment_angles(get_segments(pose, w, h))))
            else:
                data.append((None, None))

            if total and len(data) * 100 // total >= next_report:
                print(f"  {next_report}%")
                next_report += 10
    cap.release()

    if not data:
        sys.exit(f"The reference video has no readable frames: {path}")

    found = sum(1 for pose, _ in data if pose is not None)
    print(f"Done: {len(data)} frames, person found in {found} of them.")
    if found == 0:
        sys.exit("No person found anywhere in the reference video. Try a "
                 "clearer, full-body video.")
    return data, fps


class ReferencePlayer:
    """Plays the reference video in real time and says which frame is current.

    Playback is driven by the clock, not by how fast the main loop runs, so the
    reference always stays in time even if the webcam loop is slow (frames are
    skipped to catch up). Call update(dt) once per loop.
    """

    def __init__(self, path, data, fps):
        self.data = data          # per-frame (pose, angles) from analyse_reference
        self.fps = fps
        self.n = len(data)
        self.cap = cv2.VideoCapture(path)
        self.paused = False
        self.restart()

    def restart(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.index = -1           # index of the frame currently shown
        self.frame = None         # that frame (raw, un-mirrored)
        self.time = 0.0           # seconds of video played so far
        self.countdown = float(COUNTDOWN_SECONDS)
        self.finished = False

    @property
    def playing(self):
        """True while the video is actually running (scores count then)."""
        return self.countdown <= 0 and not self.paused and not self.finished

    @property
    def current(self):
        """(pose, angles) of the frame on screen, or (None, None)."""
        return self.data[self.index] if self.index >= 0 else (None, None)

    def update(self, dt):
        """Advance the clock by dt seconds. Returns True if the video looped."""
        if not self.paused:
            if self.countdown > 0:
                self.countdown = max(0.0, self.countdown - dt)
            elif not self.finished:
                self.time += dt

        looped = False
        target = int(self.time * self.fps)
        if target >= self.n:
            if LOOP_REFERENCE:
                self.restart()
                looped = True
                target = 0
            else:
                target = self.n - 1
                self.finished = True
        self._seek(target)
        return looped

    def _seek(self, target):
        """Move forward to frame `target`, skipping (not decoding) frames in between."""
        while self.index < target:
            self.index += 1
            if self.index == target:
                ok, frame = self.cap.read()
                if ok:
                    self.frame = frame
            elif not self.cap.grab():
                self.index = target  # video ended earlier than expected
                break

    def render(self, height):
        """The current reference frame as an image `height` pixels tall."""
        if self.frame is None:
            img = np.zeros((height, int(height * 4 / 3), 3), np.uint8)
        else:
            img = cv2.flip(self.frame, 1) if MIRROR else self.frame.copy()
            scale = height / img.shape[0]
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), height))

        pose, _ = self.current
        if pose is not None:
            draw_skeleton(img, pose)
        else:
            _text(img, "No person in this frame", (10, 60), (0, 0, 255), 0.6, 2)

        _text(img, "REFERENCE", (10, 30), (255, 255, 255), 0.8, 2)
        if self.finished:
            status = "FINISHED"
        elif self.paused:
            status = "PAUSED"
        else:
            status = f"{self.time:.1f}s / {self.n / self.fps:.1f}s"
        _text(img, status, (10, height - 15), (255, 255, 255), 0.7, 2)
        return img

    def release(self):
        self.cap.release()


# =============================================================================
# MAIN
# =============================================================================
def main(video_path=None):
    """Run the live comparison. video_path defaults to REFERENCE_VIDEO."""
    ensure_model()
    path = resolve_path(video_path or REFERENCE_VIDEO)
    data, ref_fps = analyse_reference(path)
    player = ReferencePlayer(path, data, ref_fps)
    balance = BalanceBar(player.n / player.fps)   # the chess.com-style bar
    smoother = ScoreSmoother()
    recorder = RunRecorder(data, ref_fps)         # position / timing / sharpness / holds

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX / permissions.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS or 0)  # hint to the camera (not always honoured)
    frame_time = 1.0 / TARGET_FPS if TARGET_FPS else 0

    start = time.time()
    prev = start              # for the FPS counter
    last_loop = start         # for the reference clock
    last_print = 0.0
    score_sum, score_count = 0.0, 0   # running average since the video (re)started

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            loop_start = time.time()
            dt = loop_start - last_loop
            last_loop = loop_start

            ok, frame = cap.read()
            if not ok:
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Advance the reference video; a loop restarts the running average and the bar
            if player.update(dt):
                finish_run(recorder)                  # the run that just ended
                recorder = RunRecorder(data, ref_fps)
                score_sum, score_count = 0.0, 0
                balance.reset()
                smoother.reset()
            elif player.finished:                     # LOOP_REFERENCE = False: video ended
                finish_run(recorder)
            _, ref_angles = player.current

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)  # must increase each frame
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                _text(frame, "No person detected", (10, 70),
                      (0, 0, 255), 0.7, 2)

            for pose in result.pose_landmarks:
                draw_skeleton(frame, pose)

                segs = get_segments(pose, w, h)
                results = compare(segment_angles(segs), ref_angles or {})
                overall = overall_score(results)

                draw_segment_colors(frame, segs, results)
                if SHOW_PART_SCORES:
                    draw_part_scores(frame, segs, results)
                draw_panel(frame, results, overall)

                # Only count scores while the reference is really playing
                if overall is not None and player.playing:
                    score_sum += overall
                    score_count += 1
                    recorder.record(player.time, results, overall)

                    # Score bar: smooth the overall score, band it, store it at this video time
                    smooth = smoother.update(overall, time.time())
                    balance.record(player.time, balance_band(smooth))

                    # Hand the per-body-part scores to anyone who asked for them
                    if SCORE_CALLBACK is not None:
                        SCORE_CALLBACK({
                            "time": player.time,
                            "overall": overall,
                            "parts": {n: (r["score"] if r else None)
                                      for n, r in results.items()},
                            "balance": balance.value,
                        })

                if PRINT_SCORES and time.time() - last_print >= PRINT_INTERVAL:
                    last_print = time.time()
                    parts = [
                        f"{n}: {r['score']:.0f}" if r else f"{n}: --"
                        for n, r in results.items()
                    ]
                    total = f"{overall:.0f}" if overall is not None else "--"
                    bar_txt = f"BAR {balance.value * 100:.0f}% | " if SHOW_SCORE_BAR else ""
                    print(f"TOTAL {total} | {bar_txt}" + " | ".join(parts))

            if SHOW_SCORE_BAR:
                draw_score_bar(frame, balance.value)

            if SHOW_AVG_SCORE and score_count:
                avg = score_sum / score_count
                _text(frame, f"AVG: {avg:.0f}", (10, h - 55),
                      color_for_score(avg), 0.8, 2)

            if player.countdown > 0:
                _text(frame, str(math.ceil(player.countdown)),
                      (w // 2 - 40, h // 2 + 50), (0, 255, 255), 4, 8)
                _text(frame, "Get ready!", (w // 2 - 80, h // 2 - 60),
                      (0, 255, 255), 1.0, 3)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            _text(frame, f"FPS: {fps:.0f}", (10, 30), (255, 255, 255), 0.8, 2)

            cv2.imshow("Pose compare (live | reference video)",
                       np.hstack([frame, player.render(h)]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                player.paused = not player.paused
            elif key == ord("r"):
                player.restart()
                score_sum, score_count = 0.0, 0
                balance.reset()
                smoother.reset()
                recorder = RunRecorder(data, ref_fps)   # discard the run so far

            # Cap the frame rate by sleeping off any leftover frame time
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    cap.release()
    player.release()
    cv2.destroyAllWindows()
    finish_run(recorder, complete=False)   # a run stopped early is still reported
    if score_count:
        print(f"Average score: {score_sum / score_count:.0f}")
        if SHOW_SCORE_BAR:
            print(f"Final score bar: {balance.value * 100:.0f}% green")


if __name__ == "__main__":
    main()