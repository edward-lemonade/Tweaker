import collections
import dataclasses
import math

import cv2
import numpy as np


class SkinBlobTracker:
    """
    Lightweight fallback that finds skin-coloured blobs when MediaPipe drops
    the hand (e.g. during motion blur).

    Maintains a rolling history of confirmed centroids (fed from both
    MediaPipe results and successful blob hits via notify()).  Each call to
    find() fits a weighted-least-squares linear velocity to that history,
    projects it forward to the current timestamp, and centres the ROI search
    window on the predicted position rather than the last known one.
    """

    _LOWER1 = np.array([0, 40, 60], dtype=np.uint8)
    _UPPER1 = np.array([25, 255, 255], dtype=np.uint8)
    _LOWER2 = np.array([160, 40, 60], dtype=np.uint8)
    _UPPER2 = np.array([180, 255, 255], dtype=np.uint8)
    _YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
    _YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

    BASE_ROI_FRAC = 0.18
    MAX_ROI_FRAC = 0.35
    MIN_BLOB_FRAC = 0.001
    MIN_BLOB_PX = 800
    HISTORY_LEN = 8

    def __init__(self, frame_w: int, frame_h: int):
        self._fw = frame_w
        self._fh = frame_h
        self._history: collections.deque = collections.deque(maxlen=self.HISTORY_LEN)
        self._debug_blob: tuple[np.ndarray, tuple[int, int]] | None = None
        self._skin_palette: tuple[int, int, int, int, int, int, int, int, int, int] | None = None
        self._skin_color_history: collections.deque = collections.deque(maxlen=16)
        self._skin_color_bgr: tuple[int, int, int] | None = None

    def notify(self, ts_s: float, nx: float, ny: float) -> None:
        self._history.append((ts_s, nx, ny))

    def find(
        self,
        frame_bgr: np.ndarray,
        ts_s: float,
        last_x: float,
        last_y: float,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        fw, fh = self._fw, self._fh

        search_x, search_y = self._predict(ts_s, last_x, last_y)
        r_frac = self._roi_radius_frac()

        r = int(r_frac * fw)
        cx = int(search_x * fw)
        cy = int(search_y * fh)
        x0 = max(0, cx - r)
        x1 = min(fw, cx + r)
        y0 = max(0, cy - r)
        y1 = min(fh, cy + r)

        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        center = (cx - x0, cy - y0)
        mask = self._make_skin_mask(roi, center)

        roi_h, roi_w = mask.shape
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, center, min(r, roi_w // 2, roi_h // 2), 255, -1)
        mask = cv2.bitwise_and(mask, circle_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._debug_blob = None
            return None

        roi_area = (x1 - x0) * (y1 - y0)
        center_pt = (cx - x0, cy - y0)

        if self._skin_palette is not None:
            candidates = []
            h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, cr_lo, cr_hi, cb_lo, cb_hi = self._skin_palette
            h_c = self._hue_center(h_lo, h_hi)
            s_c = (s_lo + s_hi) / 2.0
            v_c = (v_lo + v_hi) / 2.0
            cr_c = (cr_lo + cr_hi) / 2.0
            cb_c = (cb_lo + cb_hi) / 2.0
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                #if area < self.MIN_BLOB_PX:
                #    continue
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                hsv_mean = cv2.mean(hsv_roi, mask=contour_mask)
                ycrcb_mean = cv2.mean(ycrcb_roi, mask=contour_mask)
                dh = min(abs(hsv_mean[0] - h_c), 180 - abs(hsv_mean[0] - h_c)) / 180.0
                ds = abs(hsv_mean[1] - s_c) / 255.0
                dv = abs(hsv_mean[2] - v_c) / 255.0
                dcr = abs(ycrcb_mean[1] - cr_c) / 255.0
                dcb = abs(ycrcb_mean[2] - cb_c) / 255.0
                score = dh + ds + dv + dcr + dcb
                #if cv2.pointPolygonTest(cnt, center_pt, False) < 0:
                #    score += 0.75
                candidates.append((score, area, cnt))

            if candidates:
                candidates.sort(key=lambda x: (x[0], -x[1]))
                best = candidates[0][2]
            else:
                best = max(contours, key=cv2.contourArea)
        else:
            best = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(best)
        #if area < self.MIN_BLOB_FRAC * roi_area or area < self.MIN_BLOB_PX:
        #    self._debug_blob = None
        #    return None

        moments = cv2.moments(best)
        if moments["m00"] == 0:
            self._debug_blob = None
            return None

        blob_cx = (moments["m10"] / moments["m00"] + x0) / fw
        blob_cy = (moments["m01"] / moments["m00"] + y0) / fh
        if math.hypot(blob_cx - search_x, blob_cy - search_y) > r_frac:
            self._debug_blob = None
            return None

        self._debug_blob = (best, (x0, y0))
        return (search_x, search_y), (blob_cx, blob_cy)

    def reset(self) -> None:
        self._history.clear()
        self._debug_blob = None

    def update_skin_palette(self, frame_bgr: np.ndarray, landmarks: list) -> None:
        frame_h, frame_w = frame_bgr.shape[:2]
        sample_pts = []
        for lm in landmarks:
            x = int(round(lm.x * frame_w))
            y = int(round(lm.y * frame_h))
            if 0 <= x < frame_w and 0 <= y < frame_h:
                sample_pts.append((x, y))
        if not sample_pts:
            return

        sampled_colors = []
        for x, y in sample_pts:
            x0 = max(0, x - 2)
            x1 = min(frame_w, x + 3)
            y0 = max(0, y - 2)
            y1 = min(frame_h, y + 3)
            patch = frame_bgr[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            sampled_colors.append(patch.reshape(-1, 3))
        if not sampled_colors:
            return

        sampled_colors = np.vstack(sampled_colors)
        avg_bgr = np.round(sampled_colors.mean(axis=0)).astype(int)
        self._skin_color_history.append(avg_bgr)
        history = np.vstack(self._skin_color_history)
        self._skin_color_bgr = tuple(np.round(history.mean(axis=0)).astype(int).tolist())

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        patch = np.zeros((0, 3), dtype=np.int32)
        for x, y in sample_pts:
            x0 = max(0, x - 2)
            x1 = min(frame_w, x + 3)
            y0 = max(0, y - 2)
            y1 = min(frame_h, y + 3)
            pixels = hsv[y0:y1, x0:x1].reshape(-1, 3)
            patch = np.vstack((patch, pixels)) if patch.size else pixels
        if patch.size == 0:
            return

        h = patch[:, 0].astype(np.int32)
        s = patch[:, 1].astype(np.int32)
        v = patch[:, 2].astype(np.int32)
        ycrcb_patch = np.vstack(
            [
                ycrcb[y0:y1, x0:x1].reshape(-1, 3)
                for x, y in sample_pts
                for x0 in [max(0, x - 2)]
                for x1 in [min(frame_w, x + 3)]
                for y0 in [max(0, y - 2)]
                for y1 in [min(frame_h, y + 3)]
            ]
        )
        if ycrcb_patch.size == 0:
            return

        cr = ycrcb_patch[:, 1].astype(np.int32)
        cb = ycrcb_patch[:, 2].astype(np.int32)

        if np.median(s) < 40 or np.median(v) < 35:
            return

        h_med = int(np.median(h))
        s_med = int(np.median(s))
        v_med = int(np.median(v))
        cr_med = int(np.median(cr))
        cb_med = int(np.median(cb))

        h_std = int(np.clip(np.std(h), 4, 10))
        s_std = int(np.clip(np.std(s), 10, 25))
        v_std = int(np.clip(np.std(v), 10, 30))
        cr_std = int(np.clip(np.std(cr), 3, 8))
        cb_std = int(np.clip(np.std(cb), 3, 8))

        self._skin_palette = (
            max(0, h_med - max(5, h_std)),
            min(180, h_med + max(5, h_std)),
            max(40, s_med - max(20, s_std)),
            min(255, s_med + max(20, s_std)),
            max(30, v_med - max(20, v_std)),
            min(255, v_med + max(20, v_std)),
            max(133, cr_med - max(6, cr_std)),
            min(173, cr_med + max(6, cr_std)),
            max(77, cb_med - max(6, cb_std)),
            min(127, cb_med + max(6, cb_std)),
        )

    def _make_skin_mask(self, roi: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

        if self._skin_palette is not None:
            h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, cr_lo, cr_hi, cb_lo, cb_hi = self._skin_palette
            mask_hsv = self._hue_in_range(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
            mask_ycrcb = cv2.inRange(
                ycrcb,
                np.array([0, cr_lo, cb_lo], dtype=np.uint8),
                np.array([255, cr_hi, cb_hi], dtype=np.uint8),
            )
        else:
            sampled = self._sample_skin_palette(hsv, ycrcb, center)
            if sampled is None:
                mask_hsv = cv2.inRange(hsv, self._LOWER1, self._UPPER1)
                mask_hsv |= cv2.inRange(hsv, self._LOWER2, self._UPPER2)
                mask_ycrcb = cv2.inRange(ycrcb, self._YCRCB_LOWER, self._YCRCB_UPPER)
            else:
                h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, cr_lo, cr_hi, cb_lo, cb_hi = sampled
                mask_hsv = self._hue_in_range(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
                mask_ycrcb = cv2.inRange(
                    ycrcb,
                    np.array([0, cr_lo, cb_lo], dtype=np.uint8),
                    np.array([255, cr_hi, cb_hi], dtype=np.uint8),
                )

        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _sample_skin_palette(
        self,
        hsv: np.ndarray,
        ycrcb: np.ndarray,
        center: tuple[int, int],
    ) -> tuple[int, int, int, int, int, int, int, int, int, int] | None:
        sample_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        radius = max(10, min(hsv.shape[1], hsv.shape[0]) // 8)
        cv2.circle(sample_mask, center, radius, 255, -1)

        sample_hsv = hsv[sample_mask == 255]
        sample_ycrcb = ycrcb[sample_mask == 255]
        if sample_hsv.shape[0] < 50:
            return None

        h = sample_hsv[:, 0].astype(np.int32)
        s = sample_hsv[:, 1].astype(np.int32)
        v = sample_hsv[:, 2].astype(np.int32)
        cr = sample_ycrcb[:, 1].astype(np.int32)
        cb = sample_ycrcb[:, 2].astype(np.int32)
        if np.median(s) < 40 or np.median(v) < 40:
            return None

        h_med = int(np.median(h))
        s_med = int(np.median(s))
        v_med = int(np.median(v))
        cr_med = int(np.median(cr))
        cb_med = int(np.median(cb))

        h_std = int(np.clip(np.std(h), 5, 15))
        s_std = int(np.clip(np.std(s), 20, 40))
        v_std = int(np.clip(np.std(v), 20, 50))
        cr_std = int(np.clip(np.std(cr), 4, 12))
        cb_std = int(np.clip(np.std(cb), 4, 12))

        return (
            max(0, h_med - max(10, h_std)),
            min(180, h_med + max(10, h_std)),
            max(40, s_med - max(25, s_std)),
            min(255, s_med + max(25, s_std)),
            max(30, v_med - max(30, v_std)),
            min(255, v_med + max(30, v_std)),
            max(133, cr_med - max(10, cr_std)),
            min(173, cr_med + max(10, cr_std)),
            max(77, cb_med - max(10, cb_std)),
            min(127, cb_med + max(10, cb_std)),
        )

    def _hue_in_range(
        self,
        hsv: np.ndarray,
        h_lo: int,
        h_hi: int,
        s_lo: int,
        s_hi: int,
        v_lo: int,
        v_hi: int,
    ) -> np.ndarray:
        if h_lo <= h_hi:
            return cv2.inRange(
                hsv,
                np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                np.array([h_hi, s_hi, v_hi], dtype=np.uint8),
            )

        mask_lo = cv2.inRange(
            hsv,
            np.array([0, s_lo, v_lo], dtype=np.uint8),
            np.array([h_hi, s_hi, v_hi], dtype=np.uint8),
        )
        mask_hi = cv2.inRange(
            hsv,
            np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
            np.array([180, s_hi, v_hi], dtype=np.uint8),
        )
        return cv2.bitwise_or(mask_lo, mask_hi)

    def _hue_center(self, h_lo: int, h_hi: int) -> float:
        if h_lo <= h_hi:
            return float(h_lo + h_hi) / 2.0
        mid = (h_lo + h_hi + 180) / 2.0
        return mid % 180.0

    def debug_draw(
        self,
        frame_bgr: np.ndarray,
        search_xy: tuple[float, float],
        found_xy: tuple[float, float] | None,
    ) -> None:
        fw, fh = self._fw, self._fh
        r = int(self._roi_radius_frac() * fw)
        cx = int(search_xy[0] * fw)
        cy = int(search_xy[1] * fh)
        cv2.circle(frame_bgr, (cx, cy), r, (0, 180, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(frame_bgr, (cx, cy), (0, 180, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

        pts = list(self._history)
        for i in range(1, len(pts)):
            p0 = (int(pts[i - 1][1] * fw), int(pts[i - 1][2] * fh))
            p1 = (int(pts[i][1] * fw), int(pts[i][2] * fh))
            alpha = i / len(pts)
            cv2.line(frame_bgr, p0, p1, (0, int(120 * alpha), int(255 * alpha)), 1, cv2.LINE_AA)

        if self._debug_blob is not None:
            contour, offset = self._debug_blob
            overlay = frame_bgr.copy()
            cv2.drawContours(overlay, [contour], -1, (0, 140, 255), -1, cv2.LINE_AA, offset=offset)
            cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

        if self._skin_color_bgr is not None:
            rect_w, rect_h = 40, 24
            rx, ry = 10, 10
            cv2.rectangle(frame_bgr, (rx, ry), (rx + rect_w, ry + rect_h), self._skin_color_bgr, -1)
            cv2.rectangle(frame_bgr, (rx, ry), (rx + rect_w, ry + rect_h), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, "S", (rx + 4, ry + rect_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if found_xy is not None:
            bx = int(found_xy[0] * fw)
            by = int(found_xy[1] * fh)
            cv2.drawMarker(frame_bgr, (bx, by), (0, 255, 128), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    def debug_draw_swatch(
        self,
        frame_bgr: np.ndarray,
        origin: tuple[int, int],
        label: str = "S",
    ) -> None:
        if self._skin_color_bgr is None:
            return

        rect_w, rect_h = 40, 24
        rx, ry = origin
        cv2.rectangle(frame_bgr, (rx, ry), (rx + rect_w, ry + rect_h), self._skin_color_bgr, -1)
        cv2.rectangle(frame_bgr, (rx, ry), (rx + rect_w, ry + rect_h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (rx + 4, ry + rect_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _predict(self, ts_s: float, fallback_x: float, fallback_y: float) -> tuple[float, float]:
        if len(self._history) < 2:
            if self._history:
                return self._history[-1][1], self._history[-1][2]
            return fallback_x, fallback_y

        times = np.array([h[0] for h in self._history])
        xs = np.array([h[1] for h in self._history])
        ys = np.array([h[2] for h in self._history])

        t0 = times[-1]
        t = times - t0
        dt = ts_s - t0
        n = len(t)
        w = np.arange(1, n + 1, dtype=float)

        sw = w.sum()
        swt = (w * t).sum()
        swt2 = (w * t * t).sum()
        denom = sw * swt2 - swt * swt
        if abs(denom) < 1e-12:
            return float(xs[-1]), float(ys[-1])

        def _fit(arr):
            swa = (w * arr).sum()
            swat = (w * arr * t).sum()
            slope = (sw * swat - swt * swa) / denom
            intercept = (swa - slope * swt) / sw
            return float(np.clip(slope * dt + intercept, 0.0, 1.0))

        return _fit(xs), _fit(ys)

    def _roi_radius_frac(self) -> float:
        if len(self._history) < 2:
            return self.BASE_ROI_FRAC
        h0, h1 = self._history[-2], self._history[-1]
        elapsed = max(h1[0] - h0[0], 1e-6)
        speed = math.hypot(h1[1] - h0[1], h1[2] - h0[2]) / elapsed
        return min(self.BASE_ROI_FRAC * (1.0 + speed), self.MAX_ROI_FRAC)


def track_with_blob(
    frame,
    ts_s: float,
    last_hand,
    last_lm,
    vel_tracker,
    blob_tracker: SkinBlobTracker,
    blob_offset: tuple[float, float],
):
    ox, oy = blob_offset
    last_cx = last_hand.x - ox
    last_cy = last_hand.y - oy
    blob_result = blob_tracker.find(frame, ts_s, last_cx, last_cy)
    if blob_result is None:
        return None

    search_xy, (bx, by) = blob_result
    wx, wy = bx + ox, by + oy
    vx, vy = vel_tracker.update(wx, wy)
    updated = dataclasses.replace(
        last_hand,
        x=round(wx, 6),
        y=round(wy, 6),
        vx=round(vx, 2),
        vy=round(vy, 2),
    )

    draw = None
    new_lm = last_lm
    if last_lm is not None:
        dx = wx - last_hand.x
        dy = wy - last_hand.y
        draw = _translate_landmarks(last_lm, dx, dy)
        new_lm = draw

    blob_tracker.notify(ts_s, bx, by)
    return updated, new_lm, draw, (search_xy, (bx, by)), "blob"


class _LM:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _translate_landmarks(lms, dx: float, dy: float):
    return [_LM(lm.x + dx, lm.y + dy, lm.z) for lm in lms]
