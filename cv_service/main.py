import cv2
import mediapipe as mp
import math
import time
import threading
import pathlib

from schema   import Hand, Pose, AWAY_HAND
from velocity import VelocityTracker
import overlay

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
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),            # thumb
    (0,5),(5,6),(6,7),(7,8),            # index
    (5,9),(9,10),(10,11),(11,12),       # middle
    (9,13),(13,14),(14,15),(15,16),     # ring
    (13,17),(17,18),(18,19),(19,20),    # pinky
    (0,17),                             # palm base
]


def draw_skeleton(frame, landmarks, w: int, h: int) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 80), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (50, 150, 50),   1,  cv2.LINE_AA)

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

    tracker = VelocityTracker(frame_w, frame_h)
    holder = _ResultHolder()
    tick_hz = cv2.getTickFrequency()
    prev_t = cv2.getTickCount()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
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

            # build hand from latest result
            result = holder.get()
            if result and result.hand_landmarks:
                lm = result.hand_landmarks[0]
                pointer = lm[POINTER_TIP]
                vx, vy = tracker.update(pointer.x, pointer.y)
                hand = Hand(
                    x = round(pointer.x, 6),
                    y = round(pointer.y, 6),
                    vx = round(vx, 2),
                    vy = round(vy, 2),
                    pose = classify_pose(lm),
                )
                draw_skeleton(frame, lm, frame_w, frame_h)
            else:
                tracker.reset()
                hand = AWAY_HAND

            # Emit
            print(hand.as_dict())

            # FPS
            now    = cv2.getTickCount()
            fps    = tick_hz / (now - prev_t)
            prev_t = now

            overlay.draw(frame, hand, fps)
            cv2.imshow("Hand Gesture Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("o"), ord("O")):
                overlay.toggle()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()