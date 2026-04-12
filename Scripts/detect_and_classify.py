import os
import sys
import math
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.70
CAMERA_ID = 1
# ───────────────────────────────────────────────────────────────

# Load YOLO OBB model once at module level
model = YOLO(os.path.join(script_dir, "..", "models", "best.pt"))


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


def process_image(image_path):
    """Run detection on a single image. Returns list of location strings 'cx cy angle'."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    annotated, results = detect_and_annotate(frame)
    locations = [f"{cx} {cy} {angle:.1f}" for (cx, cy, angle, _, _) in results]

    cv2.imshow("Detected Objects", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return locations


def process_camera():
    """Run detection on live camera feed. Returns locations from last captured frame."""
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}.")
        return []

    print("Camera loaded. Press 'q' to quit.")

    cv2.namedWindow("Detect & Classify")
    cv2.createTrackbar("Confidence %", "Detect & Classify", int(DEFAULT_CONFIDENCE * 100), 100, lambda x: None)

    locations = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        conf_pct = cv2.getTrackbarPos("Confidence %", "Detect & Classify")
        confidence = max(conf_pct / 100.0, 0.01)

        annotated, results = detect_and_annotate(frame, confidence)
        locations = [f"{cx} {cy} {angle:.1f}" for (cx, cy, angle, _, _) in results]

        cv2.imshow("Detect & Classify", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return locations


def main():
    if len(sys.argv) > 1:
        locations = process_image(sys.argv[1])
    else:
        locations = process_camera()

    print("\nDetected Locations (Centroid X, Centroid Y, Angle):")
    for loc in locations:
        print(loc)


if __name__ == "__main__":
    main()
