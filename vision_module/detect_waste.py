"""Detect 'Hazard' and 'Waste Item' objects by color in a webcam or video stream.

Install:
    pip install opencv-python numpy

Run:
    python detect_waste.py                      # default webcam (index 0)
    python detect_waste.py --source 1           # second webcam
    python detect_waste.py --source clip.mp4    # saved video file

Keys: q or Esc quits, m toggles the mask view, space pauses.
"""
import argparse
import sys
import time

import cv2
import numpy as np

# HSV ranges (OpenCV scale: H 0-179, S 0-255, V 0-255). Tune these for your
# lighting and camera. Each label can have several ranges (red wraps around 0).
LABELS = {
    "Hazard": {
        "color": (60, 80, 255),  # BGR, drawn as red
        "ranges": [
            ((0, 120, 70), (10, 255, 255)),     # red
            ((170, 120, 70), (179, 255, 255)),  # red (wrap-around)
            ((11, 120, 120), (35, 255, 255)),   # orange and yellow
        ],
    },
    "Waste Item": {
        "color": (230, 190, 60),  # BGR, drawn as cyan-blue
        "ranges": [
            ((40, 80, 60), (85, 255, 255)),     # green
            ((95, 80, 60), (130, 255, 255)),    # blue
        ],
    },
}

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")
    return cap


def build_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low), np.array(high))
    # Remove speckle noise, then fill small holes in the blobs.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)


def detect(frame: np.ndarray, min_area: int):
    """Return (detections, combined_mask). Each detection is (label, x, y, w, h)."""
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    detections = []
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    for label, cfg in LABELS.items():
        mask = build_mask(hsv, cfg["ranges"])
        combined |= mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append((label, x, y, w, h))
    return detections, combined


def draw(frame, detections, frame_no, fps):
    for label, x, y, w, h in detections:
        color = LABELS[label]["color"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(y - th - 8, 0)
        cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 8), color, -1)
        cv2.putText(frame, label, (x + 4, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    hud = f"Frame {frame_no}   FPS {fps:5.1f}   Detections {len(detections)}"
    cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index or path to a video file")
    p.add_argument("--min-area", type=int, default=800, help="ignore blobs smaller than this many pixels")
    p.add_argument("--width", type=int, default=640, help="resize frames to this width (0 keeps original)")
    p.add_argument("--print-every", type=int, default=1, help="print detections every N frames")
    args = p.parse_args()

    cap = open_source(args.source)
    frame_no, fps, last = 0, 0.0, time.perf_counter()
    show_mask, paused = False, False
    frame = mask = None

    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                print("End of stream.")
                break
            if args.width and frame.shape[1] != args.width:
                scale = args.width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_no += 1

            detections, mask = detect(frame, args.min_area)

            if detections and frame_no % args.print_every == 0:
                for label, x, y, w, h in detections:
                    cx, cy = x + w // 2, y + h // 2
                    print(f"frame={frame_no:05d} {label:<10} x={x} y={y} w={w} h={h} center=({cx},{cy})")

            # Smooth FPS with an exponential moving average.
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now

            view = frame.copy()
            draw(view, detections, frame_no, fps)

        shown = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else view
        cv2.imshow("Waste detector", shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()