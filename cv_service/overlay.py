import cv2
from schema import Hand, Pose

_OVERLAY_BG = (15, 15, 15)
_TEXT = (220, 220, 220)
_ACCENT = (80, 220, 120)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_POSE_LABEL = {
    Pose.FLAT: "FLAT",
    Pose.PINCH: "PINCH",
    Pose.POINT: "POINT",
    Pose.IDLE: "IDLE",
    Pose.AWAY: "AWAY",
}

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]

_enabled: bool = True
def enable() -> None: global _enabled; _enabled = True
def disable() -> None: global _enabled; _enabled = False
def toggle() -> bool: global _enabled; _enabled = not _enabled

def is_enabled() -> bool:
    return _enabled

def draw(frame, hand: Hand, fps: float) -> None:
    if not _enabled:
        return

    h, w = frame.shape[:2]

    # Semi-transparent bottom banner
    bg = frame.copy()
    cv2.rectangle(bg, (0, h - 96), (w, h), _OVERLAY_BG, -1)
    cv2.addWeighted(bg, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, _POSE_LABEL[hand.pose],
                (18, h - 58), _FONT, 1.1, _ACCENT, 2, cv2.LINE_AA)

    detail = (f"x={hand.x:.3f}  y={hand.y:.3f}  "
              f"vx={hand.vx:+.0f}  vy={hand.vy:+.0f} px/s")
    cv2.putText(frame, detail,
                (18, h - 20), _FONT, 0.58, _TEXT, 1, cv2.LINE_AA)

    fps_txt = f"FPS {fps:.1f}"
    tw = cv2.getTextSize(fps_txt, _FONT, 0.58, 1)[0][0]
    cv2.putText(frame, fps_txt,
                (w - tw - 14, h - 20), _FONT, 0.58, _TEXT, 1, cv2.LINE_AA)

    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 36), _OVERLAY_BG, -1)
    cv2.putText(frame, "Hand Gesture Capture  |  Q/Esc quit  O toggle overlay",
                (12, 24), _FONT, 0.58, _TEXT, 1, cv2.LINE_AA)
    
def draw_landmarks(frame, hand_landmarks) -> None:
    if not _enabled or hand_landmarks is None:
        return

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], _ACCENT, 2, cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, _TEXT, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, _ACCENT, 1, cv2.LINE_AA)