import dataclasses

import cv2

from schema import AWAY_HAND, Pose

MAX_DR_VELOCITY_NORM = 20.0
WRIST = 0
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
_motion_state: dict[int, dict[str, float]] = {}


def update_motion_state(vel_tracker, vx: float, vy: float, ts_ms: int) -> None:
    key = id(vel_tracker)
    prev = _motion_state.get(key)
    if prev is None:
        _motion_state[key] = {
            "vx": vx,
            "vy": vy,
            "ax": 0.0,
            "ay": 0.0,
            "t_ms": float(ts_ms),
        }
        return

    dt_s = (ts_ms - prev["t_ms"]) / 1000.0
    if dt_s <= 0:
        prev["vx"] = vx
        prev["vy"] = vy
        prev["t_ms"] = float(ts_ms)
        return

    prev["ax"] = (vx - prev["vx"]) / dt_s
    prev["ay"] = (vy - prev["vy"]) / dt_s
    prev["vx"] = vx
    prev["vy"] = vy
    prev["t_ms"] = float(ts_ms)


def reset_motion_state(vel_tracker) -> None:
    _motion_state.pop(id(vel_tracker), None)

def handle_missing_hand(
    ts_ms: int,
    last_seen_ms: int,
    away_after_ms: int,
    last_hand,
    last_lm,
    vel_tracker,
    blob_tracker,
):
    if last_hand.pose == Pose.AWAY:
        return last_hand, last_lm, None, None, "dr"

    if (ts_ms - last_seen_ms) > away_after_ms:
        vel_tracker.reset()
        reset_motion_state(vel_tracker)
        blob_tracker.reset()
        return AWAY_HAND, None, None, None, "dr"

    return None


def track_with_dead_reckoning(last_hand, last_lm, vel_tracker, dt_s: float, frame_w: int, frame_h: int):
    vel_tracker.reset()
    state = _motion_state.get(id(vel_tracker))
    vx_px = state["vx"] if state is not None else last_hand.vx
    vy_px = state["vy"] if state is not None else last_hand.vy
    ax_px = state["ax"] if state is not None else 0.0
    ay_px = state["ay"] if state is not None else 0.0

    vx_norm = vx_px / max(frame_w, 1)
    vy_norm = vy_px / max(frame_h, 1)
    ax_norm = ax_px / max(frame_w, 1)
    ay_norm = ay_px / max(frame_h, 1)
    dx = _cap_velocity(vx_norm * dt_s + 0.5 * ax_norm * dt_s * dt_s)
    dy = _cap_velocity(vy_norm * dt_s + 0.5 * ay_norm * dt_s * dt_s)
    updated = dataclasses.replace(
        last_hand,
        x=round(last_hand.x + dx, 6),
        y=round(last_hand.y + dy, 6),
    )

    draw = _translate_landmarks(last_lm, dx, dy) if last_lm is not None else None
    new_lm = draw if draw is not None else last_lm
    return updated, new_lm, draw, None, "dr"


def _cap_velocity(v: float) -> float:
    return max(-MAX_DR_VELOCITY_NORM, min(MAX_DR_VELOCITY_NORM, v))


class _LM:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _translate_landmarks(lms, dx: float, dy: float):
    return [_LM(lm.x + dx, lm.y + dy, lm.z) for lm in lms]


def draw_dead_reckoning_debug(frame_bgr, prev_lm, dr_lm, label: str) -> None:
    if prev_lm is None or dr_lm is None:
        return

    h, w = frame_bgr.shape[:2]
    old_pts = [(int(lm.x * w), int(lm.y * h)) for lm in prev_lm]
    new_pts = [(int(lm.x * w), int(lm.y * h)) for lm in dr_lm]
    if len(old_pts) <= WRIST or len(new_pts) <= WRIST:
        return

    for a, b in _CONNECTIONS:
        cv2.line(frame_bgr, old_pts[a], old_pts[b], (40, 90, 180), 1, cv2.LINE_AA)
    for a, b in _CONNECTIONS:
        cv2.line(frame_bgr, new_pts[a], new_pts[b], (0, 140, 255), 2, cv2.LINE_AA)

    old_wrist = old_pts[WRIST]
    new_wrist = new_pts[WRIST]
    cv2.arrowedLine(frame_bgr, old_wrist, new_wrist, (0, 180, 255), 2, cv2.LINE_AA, tipLength=0.25)

    dx = new_wrist[0] - old_wrist[0]
    dy = new_wrist[1] - old_wrist[1]
    pred = new_wrist
    for _ in range(3):
        nxt = (pred[0] + dx, pred[1] + dy)
        cv2.line(frame_bgr, pred, nxt, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(frame_bgr, nxt, 2, (0, 220, 255), -1, cv2.LINE_AA)
        pred = nxt

    cv2.putText(
        frame_bgr,
        f"{label}:DR",
        (new_wrist[0] + 8, new_wrist[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
