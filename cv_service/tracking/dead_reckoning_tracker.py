import dataclasses

from schema import AWAY_HAND, Pose

MAX_DR_VELOCITY = 1.5


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
        blob_tracker.reset()
        return AWAY_HAND, None, None, None, "dr"

    return None


def track_with_dead_reckoning(last_hand, last_lm, vel_tracker, dt_s: float):
    vel_tracker.reset()
    dx = _cap_velocity(last_hand.vx) * dt_s
    dy = _cap_velocity(last_hand.vy) * dt_s
    updated = dataclasses.replace(
        last_hand,
        x=round(last_hand.x + dx, 6),
        y=round(last_hand.y + dy, 6),
    )

    draw = _translate_landmarks(last_lm, dx, dy) if last_lm is not None else None
    new_lm = draw if draw is not None else last_lm
    return updated, new_lm, draw, None, "dr"


def _cap_velocity(v: float) -> float:
    return max(-MAX_DR_VELOCITY, min(MAX_DR_VELOCITY, v))


class _LM:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _translate_landmarks(lms, dx: float, dy: float):
    return [_LM(lm.x + dx, lm.y + dy, lm.z) for lm in lms]
