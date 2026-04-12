import os
import math
import cv2
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.70
CAMERA_ID = 1
# ───────────────────────────────────────────────────────────────

# Load YOLO OBB model once at module level
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(_project_root, "models", "best.pt"))


def detect_and_annotate(frame, confidence=DEFAULT_CONFIDENCE):
    """Run YOLO OBB inference on a frame. Returns (annotated_frame, results).

    Each result is a tuple: (cx, cy, angle, label, conf)
    - cx, cy: centroid pixel coordinates
    - angle: orientation in degrees (converted from YOLO radians)
    - label: class name string
    - conf: confidence float 0.0-1.0
    """
    raw_results = model(frame, conf=confidence, verbose=False)
    result = raw_results[0]

    # YOLO renders the OBBs natively
    annotated = result.plot()

    detections = []
    if result.obb is not None:
        for box in result.obb:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            if label.lower() == "noise":
                continue

            xywhr = box.xywhr[0]
            cx, cy = float(xywhr[0]), float(xywhr[1])
            angle = math.degrees(float(xywhr[4]))
            conf = float(box.conf[0])

            detections.append((cx, cy, angle, label, conf))

            # Draw centroid dot
            cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            # Draw small label with centroid and angle
            text = f"{label} {conf*100:.1f}% ({int(cx)},{int(cy)}) {angle:.1f} deg"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            tx, ty = int(cx) - tw // 2, int(cy) - 10
            cv2.rectangle(annotated, (tx, ty - th - 4), (tx + tw + 4, ty), (0, 255, 0), -1)
            cv2.putText(annotated, text, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    return annotated, detections
