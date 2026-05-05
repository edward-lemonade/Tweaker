import argparse
import cv2
import mediapipe as mp
import math
import time
import threading
import pathlib

from schema    import Hand, Pose, AWAY_HAND
from velocity  import VelocityTracker
from messaging import emit

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = pathlib.Path(__file__).parent / "hand_landmarker.task"

# landmark indices
WRIST = 0
THUMB_TIP = 4
POINTER_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGER_BASES = [2, 6, 10, 14, 18]

def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def _hand_axis(lm) -> tuple[float, float]:
    # wrist to middle finger unit vector
    dx = lm[9].x - lm[WRIST].x
    dy = lm[9].y - lm[WRIST].y
    mag = math.hypot(dx, dy) or 1e-6
    return dx / mag, dy / mag

def _project(lm, idx: int, ax: tuple[float, float]) -> float:
    # projection of landmark idx onto axis ax, relative to wrist
    return (lm[idx].x - lm[WRIST].x) * ax[0] + (lm[idx].y - lm[WRIST].y) * ax[1]

def _finger_up(lm, tip: int, base: int, axis: tuple[float, float]) -> bool:
    return _project(lm, tip, axis) > _project(lm, base, axis)

def classify_pose(lms: list) -> Pose:
    axis = _hand_axis(lms)
    hand_size = _dist(lms[WRIST], lms[MIDDLE_TIP]) or 1e-6
    fingers = [
        _finger_up(lms, THUMB_TIP,      FINGER_BASES[0], axis),
        _finger_up(lms, POINTER_TIP,    FINGER_BASES[1], axis),
        _finger_up(lms, MIDDLE_TIP,     FINGER_BASES[2], axis),
        _finger_up(lms, RING_TIP,       FINGER_BASES[3], axis),
        _finger_up(lms, PINKY_TIP,      FINGER_BASES[4], axis),
    ]
    _, pointer, middle, ring, pinky = fingers

    # pinch
    if _dist(lms[THUMB_TIP], lms[POINTER_TIP]) / hand_size < 0.25:
        return Pose.PINCH
    
    # flat
    if sum(fingers) >= 4:
        return Pose.FLAT
    
    # point
    if pointer and not middle and not ring and not pinky:
        return Pose.POINT
    
    # idle
    return Pose.IDLE

def _build_hand(lm, tracker: VelocityTracker) -> Hand:
    pointer = lm[POINTER_TIP]
    vx, vy  = tracker.update(pointer.x, pointer.y)
    return Hand(
        x    = round(pointer.x, 6),
        y    = round(pointer.y, 6),
        vx   = round(vx, 2),
        vy   = round(vy, 2),
        pose = classify_pose(lm),
    )

class _ResultHolder:
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
 
    def update(self, result, *_):
        with self._lock:
            self._result = result
 
    def get(self):
        with self._lock:
            return self._result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run  python download_model.py  first."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker_left  = VelocityTracker(frame_w, frame_h)
    tracker_right = VelocityTracker(frame_w, frame_h)
    holder = _ResultHolder()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=holder.update,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            ts_ms = int(time.monotonic() * 1000)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            landmarker.detect_async(mp_image, ts_ms)

            # build hands from latest result
            left_hand = AWAY_HAND
            right_hand = AWAY_HAND

            result = holder.get()
            if result and result.hand_landmarks:
                for i, lm in enumerate(result.hand_landmarks):
                    # handedness[i].category_name is "Left" or "Right" (mirrored after flip)
                    side = result.handedness[i][0].category_name.lower()
                    if side == 'left':
                        left_hand  = _build_hand(lm, tracker_left)
                    elif side == 'right':
                        right_hand = _build_hand(lm, tracker_right)
            else:
                tracker_left.reset()
                tracker_right.reset()

            # Emit
            emit(args.port, 'GESTURE', {
                'leftHand':  left_hand.as_dict(),
                'rightHand': right_hand.as_dict(),
            })

    cap.release()


if __name__ == "__main__":
    main()