import time

class VelocityTracker:

    def __init__(self, frame_w: int, frame_h: int, alpha: float = 0.3):
        self._w     = frame_w
        self._h     = frame_h
        self._alpha = alpha
        self._px:  float | None = None
        self._py:  float | None = None
        self._vx:  float        = 0.0
        self._vy:  float        = 0.0
        self._t:   float | None = None

    def update(self, nx: float, ny: float) -> tuple[float, float]:
        # Feed normalised (x,y), return smoothed (vx,vy) in px/s.
        now = time.monotonic()
        px  = nx * self._w
        py  = ny * self._h

        if self._px is not None and self._t is not None:
            dt = now - self._t
            if dt > 0:
                raw_vx = (px - self._px) / dt
                raw_vy = (py - self._py) / dt
                self._vx = self._alpha * raw_vx + (1 - self._alpha) * self._vx
                self._vy = self._alpha * raw_vy + (1 - self._alpha) * self._vy

        self._px, self._py, self._t = px, py, now
        return self._vx, self._vy

    def reset(self) -> None:
        # Call when the tracked subject disappears; zeroes velocity.
        self._px = self._py = self._t = None
        self._vx = self._vy = 0.0