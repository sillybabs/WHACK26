"""
Real-time MediaPipe Pose Landmarker demo.
Shows your webcam feed with a skeleton overlay. Press 'q' or ESC to quit.

Setup:
    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install mediapipe opencv-python
    python pose_webcam.py
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Config -----------------------------------------------------------------
CAMERA_INDEX = 0          # try 1 if you have multiple cameras
MODEL_VARIANT = "lite"    # "lite" (fastest), "full", or "heavy" (most accurate)
MIRROR = True             # selfie-style view
MIN_VISIBILITY = 0.5      # hide landmarks the model isn't confident about

MODEL_PATH = f"pose_landmarker_{MODEL_VARIANT}.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{MODEL_VARIANT}/float16/latest/{MODEL_PATH}"
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


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        visible = (lm.visibility is None) or (lm.visibility >= MIN_VISIBILITY)
        pts.append((int(lm.x * w), int(lm.y * h)) if visible else None)

    for a, b in POSE_CONNECTIONS:
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 3, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(frame, p, 5, (0, 0, 255), -1, cv2.LINE_AA)


def main():
    ensure_model()

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX / permissions.")

    start = time.time()
    prev = start

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)  # must increase each frame
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for pose in result.pose_landmarks:
                draw_skeleton(frame, pose)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("MediaPipe Pose Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()