import math
import pathlib
import threading
from dataclasses import dataclass

import cv2
import mediapipe as mp

from schema import AWAY_HAND, Hand, Pose
from velocity import VelocityTracker

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "hand_landmarker.task"

# landmark indices
WRIST = 0
THUMB_TIP = 4
POINTER_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGER_BASES = [2, 6, 10, 14, 18]


def create_image(frame_bgr):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )


def create_options(result_callback):
    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )


class ResultHolder:
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()

    def update(self, result, *_):
        with self._lock:
            self._result = result

    def get(self):
        with self._lock:
            return self._result


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _hand_axis(lm) -> tuple[float, float]:
    dx = lm[9].x - lm[WRIST].x
    dy = lm[9].y - lm[WRIST].y
    mag = math.hypot(dx, dy) or 1e-6
    return dx / mag, dy / mag


def _project(lm, idx: int, ax: tuple[float, float]) -> float:
    return (lm[idx].x - lm[WRIST].x) * ax[0] + (lm[idx].y - lm[WRIST].y) * ax[1]


def _finger_up(lm, tip: int, base: int, axis: tuple[float, float]) -> bool:
    return _project(lm, tip, axis) > _project(lm, base, axis)


def classify_pose(lms: list) -> Pose:
    axis = _hand_axis(lms)
    hand_size = _dist(lms[WRIST], lms[MIDDLE_TIP]) or 1e-6
    fingers = [
        _finger_up(lms, THUMB_TIP, FINGER_BASES[0], axis),
        _finger_up(lms, POINTER_TIP, FINGER_BASES[1], axis),
        _finger_up(lms, MIDDLE_TIP, FINGER_BASES[2], axis),
        _finger_up(lms, RING_TIP, FINGER_BASES[3], axis),
        _finger_up(lms, PINKY_TIP, FINGER_BASES[4], axis),
    ]
    _, pointer, middle, ring, pinky = fingers
    if _dist(lms[THUMB_TIP], lms[POINTER_TIP]) / hand_size < 0.25:
        return Pose.PINCH
    if sum(fingers) >= 4:
        return Pose.FLAT
    if pointer and not middle and not ring and not pinky:
        return Pose.POINT
    return Pose.IDLE


def _build_hand(lm, tracker: VelocityTracker) -> Hand:
    wrist = lm[WRIST]
    vx, vy = tracker.update(wrist.x, wrist.y)
    ux, uy = _hand_axis(lm)
    theta = round(math.atan2(ux, -uy), 4)
    return Hand(
        x=round(wrist.x, 6),
        y=round(wrist.y, 6),
        vx=round(vx, 2),
        vy=round(vy, 2),
        theta=theta,
        pose=classify_pose(lm),
    )


@dataclass
class MediaPipeTrackingOutput:
    left_hand: Hand
    right_hand: Hand
    draw_lm_left: list | None
    draw_lm_right: list | None
    last_lm_left: list | None
    last_lm_right: list | None
    blob_offset_left: tuple[float, float]
    blob_offset_right: tuple[float, float]
    detected: bool


def track_with_mediapipe(
    result,
    frame_bgr,
    ts_s: float,
    tracker_left: VelocityTracker,
    tracker_right: VelocityTracker,
    blob_left,
    blob_right,
) -> MediaPipeTrackingOutput:
    if not (result and result.hand_landmarks):
        return MediaPipeTrackingOutput(
            left_hand=AWAY_HAND,
            right_hand=AWAY_HAND,
            draw_lm_left=None,
            draw_lm_right=None,
            last_lm_left=None,
            last_lm_right=None,
            blob_offset_left=(0.0, 0.0),
            blob_offset_right=(0.0, 0.0),
            detected=False,
        )

    left_hand = AWAY_HAND
    right_hand = AWAY_HAND
    draw_lm_left = None
    draw_lm_right = None
    last_lm_left = None
    last_lm_right = None
    blob_offset_left = (0.0, 0.0)
    blob_offset_right = (0.0, 0.0)

    for i, lm in enumerate(result.hand_landmarks):
        side = result.handedness[i][0].category_name.lower()
        palm_nodes = [lm[idx] for idx in [WRIST] + FINGER_BASES]
        pcx = sum(n.x for n in palm_nodes) / len(palm_nodes)
        pcy = sum(n.y for n in palm_nodes) / len(palm_nodes)
        wrist = lm[WRIST]
        offset = (wrist.x - pcx, wrist.y - pcy)

        if side == "left":
            left_hand = _build_hand(lm, tracker_left)
            last_lm_left = lm
            draw_lm_left = lm
            blob_left.notify(ts_s, pcx, pcy)
            blob_left.update_skin_palette(frame_bgr, lm)
            blob_offset_left = offset
        elif side == "right":
            right_hand = _build_hand(lm, tracker_right)
            last_lm_right = lm
            draw_lm_right = lm
            blob_right.notify(ts_s, pcx, pcy)
            blob_right.update_skin_palette(frame_bgr, lm)
            blob_offset_right = offset

    return MediaPipeTrackingOutput(
        left_hand=left_hand,
        right_hand=right_hand,
        draw_lm_left=draw_lm_left,
        draw_lm_right=draw_lm_right,
        last_lm_left=last_lm_left,
        last_lm_right=last_lm_right,
        blob_offset_left=blob_offset_left,
        blob_offset_right=blob_offset_right,
        detected=True,
    )
