import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

import overlay
from messaging import emit
from schema import AWAY_HAND, Pose
from tracking.blob_tracker import SkinBlobTracker, track_with_blob
from tracking.dead_reckoning_tracker import handle_missing_hand, track_with_dead_reckoning
from tracking.mediapipe_tracker import MODEL_PATH, ResultHolder, create_image, create_options, track_with_mediapipe
from velocity import VelocityTracker

HandLandmarker = mp.tasks.vision.HandLandmarker

AWAY_AFTER_MS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--debug", action="store_true", help="Show video capture window with overlay")
    parser.add_argument(
        "--hands-only",
        action="store_true",
        help="Draw hands on black instead of video feed (requires --debug)",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=-6,
        help="Manual exposure value (log2 s, e.g. -6 ≈ 1/64 s). Pass 0 to keep auto.",
    )
    parser.add_argument(
        "--sharpen-strength",
        type=float,
        default=0.8,
        help="Unsharp-mask blend weight (0 = off). Default 0.8.",
    )
    parser.add_argument("--no-blob", action="store_true", help="Disable skin-blob fallback tracker.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun  python download_model.py  first.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if args.exposure != 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker_left = VelocityTracker(frame_w, frame_h)
    tracker_right = VelocityTracker(frame_w, frame_h)
    blob_left = SkinBlobTracker(frame_w, frame_h)
    blob_right = SkinBlobTracker(frame_w, frame_h)
    holder = ResultHolder()
    options = create_options(holder.update)

    last_left = AWAY_HAND
    last_right = AWAY_HAND
    last_seen_ms = int(time.monotonic() * 1000)
    last_lm_left = None
    last_lm_right = None
    blob_offset_left = (0.0, 0.0)
    blob_offset_right = (0.0, 0.0)
    fps_timer = time.monotonic()
    fps = 0.0
    mode_left = "mp"
    mode_right = "mp"

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            ts_ms = int(time.monotonic() * 1000)
            ts_s = ts_ms / 1000.0

            if args.sharpen_strength > 0:
                blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.5, sigmaY=2.5)
                frame = cv2.addWeighted(frame, 1.0 + args.sharpen_strength, blur, -args.sharpen_strength, 0)

            landmarker.detect_async(create_image(frame), ts_ms)
            result = holder.get()

            left_hand = AWAY_HAND
            right_hand = AWAY_HAND
            draw_lm_left = None
            draw_lm_right = None
            blob_dbg_left = None
            blob_dbg_right = None

            mp_output = track_with_mediapipe(
                result=result,
                frame_bgr=frame,
                ts_s=ts_s,
                tracker_left=tracker_left,
                tracker_right=tracker_right,
                blob_left=blob_left,
                blob_right=blob_right,
            )

            if mp_output.detected:
                left_hand = mp_output.left_hand
                right_hand = mp_output.right_hand
                draw_lm_left = mp_output.draw_lm_left
                draw_lm_right = mp_output.draw_lm_right
                last_lm_left = mp_output.last_lm_left
                last_lm_right = mp_output.last_lm_right
                blob_offset_left = mp_output.blob_offset_left
                blob_offset_right = mp_output.blob_offset_right
                last_left = left_hand
                last_right = right_hand
                last_seen_ms = ts_ms
                mode_left = "mp"
                mode_right = "mp"
            else:
                dt = (ts_ms - last_seen_ms) / 1000.0

                missing = handle_missing_hand(
                    ts_ms=ts_ms,
                    last_seen_ms=last_seen_ms,
                    away_after_ms=AWAY_AFTER_MS,
                    last_hand=last_left,
                    last_lm=last_lm_left,
                    vel_tracker=tracker_left,
                    blob_tracker=blob_left,
                )
                if missing is not None:
                    last_left, last_lm_left, draw_lm_left, blob_dbg_left, mode_left = missing
                else:
                    blob_hit = None
                    if not args.no_blob:
                        blob_hit = track_with_blob(
                            frame=frame,
                            ts_s=ts_s,
                            last_hand=last_left,
                            last_lm=last_lm_left,
                            vel_tracker=tracker_left,
                            blob_tracker=blob_left,
                            blob_offset=blob_offset_left,
                        )
                    if blob_hit is not None:
                        last_left, last_lm_left, draw_lm_left, blob_dbg_left, mode_left = blob_hit
                    else:
                        last_left, last_lm_left, draw_lm_left, blob_dbg_left, mode_left = track_with_dead_reckoning(
                            last_hand=last_left,
                            last_lm=last_lm_left,
                            vel_tracker=tracker_left,
                            dt_s=dt,
                        )

                missing = handle_missing_hand(
                    ts_ms=ts_ms,
                    last_seen_ms=last_seen_ms,
                    away_after_ms=AWAY_AFTER_MS,
                    last_hand=last_right,
                    last_lm=last_lm_right,
                    vel_tracker=tracker_right,
                    blob_tracker=blob_right,
                )
                if missing is not None:
                    last_right, last_lm_right, draw_lm_right, blob_dbg_right, mode_right = missing
                else:
                    blob_hit = None
                    if not args.no_blob:
                        blob_hit = track_with_blob(
                            frame=frame,
                            ts_s=ts_s,
                            last_hand=last_right,
                            last_lm=last_lm_right,
                            vel_tracker=tracker_right,
                            blob_tracker=blob_right,
                            blob_offset=blob_offset_right,
                        )
                    if blob_hit is not None:
                        last_right, last_lm_right, draw_lm_right, blob_dbg_right, mode_right = blob_hit
                    else:
                        last_right, last_lm_right, draw_lm_right, blob_dbg_right, mode_right = track_with_dead_reckoning(
                            last_hand=last_right,
                            last_lm=last_lm_right,
                            vel_tracker=tracker_right,
                            dt_s=dt,
                        )

                left_hand = last_left
                right_hand = last_right

            emit(
                args.port,
                "GESTURE",
                {
                    "leftHand": left_hand.as_dict(),
                    "rightHand": right_hand.as_dict(),
                },
            )

            if args.debug:
                now = time.monotonic()
                fps = 1.0 / (now - fps_timer) if fps_timer else fps
                fps_timer = now

                if args.hands_only:
                    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

                if draw_lm_left is not None:
                    overlay.draw_landmarks(frame, draw_lm_left)
                if draw_lm_right is not None:
                    overlay.draw_landmarks(frame, draw_lm_right)

                if blob_dbg_left is not None:
                    blob_left.debug_draw(frame, blob_dbg_left[0], blob_dbg_left[1])
                else:
                    blob_left.debug_draw_swatch(frame, (10, 40), "L")

                if blob_dbg_right is not None:
                    blob_right.debug_draw(frame, blob_dbg_right[0], blob_dbg_right[1])
                else:
                    rect_w = 40
                    blob_right.debug_draw_swatch(frame, (frame_w - rect_w - 10, 40), "R")

                label = f"L:{mode_left}  R:{mode_right}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

                hands_present = (
                    (left_hand.pose != Pose.AWAY or right_hand.pose != Pose.AWAY)
                    and (ts_ms - last_seen_ms) < AWAY_AFTER_MS
                )
                if not hands_present:
                    red = frame.copy()
                    red[:] = (0, 0, 80)
                    cv2.addWeighted(red, 0.4, frame, 0.6, 0, frame)

                debug_hand = right_hand if right_hand.pose != Pose.AWAY else left_hand
                overlay.draw(frame, debug_hand, fps)
                cv2.imshow("Hand Gesture Capture", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("o"):
                    overlay.toggle()

    cap.release()
    if args.debug:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
