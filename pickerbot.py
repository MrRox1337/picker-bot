import os
import sys
import json
import argparse

import cv2

from pickerbot_lib import CONFIG, resolve
from pickerbot_lib.detection import detect_and_annotate
from pickerbot_lib.calibration import load_calibration_data, calculate_homography, pixel_to_world, run_calibration_gui
from pickerbot_lib.sender import connect, disconnect, epsonPickAll


def translate_points(pixel_locations, H):
    """Convert pixel location strings 'cx cy angle' to world coordinate strings 'wx wy z angle'."""
    world_locations = []
    for loc in pixel_locations:
        px, py, angle = loc.split()
        wx, wy = pixel_to_world(H, float(px), float(py))
        world_locations.append(f"{wx} {wy} {CONFIG['robot_z']} {angle}")
    return world_locations

def load_boq():
    """Reads boq.txt and limits the robot payload to requested kit quantities."""
    boq_path = resolve("boq.txt")
    if not os.path.exists(boq_path):
        return None

    boq = {}
    with open(boq_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    boq[parts[0].strip().lower()] = int(parts[1].strip())
                except ValueError:
                    continue

    return boq if boq else None

def filter_by_boq(results, boq):
    """Filters AI detections strictly against the required BOQ constraint."""
    if boq is None:
        return [f"{cx} {cy} {angle:.1f}" for (cx, cy, angle, _, _) in results]

    filtered_locations = []
    current_boq = boq.copy()

    for cx, cy, angle, label, conf in results:
        lbl = label.lower()
        if current_boq.get(lbl, 0) > 0:
            filtered_locations.append(f"{cx} {cy} {angle:.1f}")
            current_boq[lbl] -= 1

    return filtered_locations


def run_detection(src):
    """Run YOLO OBB detection on a camera frame or image file. Returns filtered pixel locations."""
    boq = load_boq()
    if boq is not None:
        print(f"-> BOQ Logistics Mode ACTIVE: {json.dumps(boq)}")
    else:
        print("-> BOQ Not Found/Empty. Defaulting to Greedy Collection Mode.")

    if src == "camera":
        cap = cv2.VideoCapture(CONFIG["webcam_id"])
        if not cap.isOpened():
            print(f"Error: Could not open camera {CONFIG['webcam_id']}.")
            sys.exit(1)

        print("Camera loaded. Press 'c' to capture and pick, 'q' to quit.")
        cv2.namedWindow("PickerBot")
        cv2.createTrackbar("Confidence %", "PickerBot", int(CONFIG["min_confidence"] * 100), 100, lambda x: None)

        locations = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            conf_pct = cv2.getTrackbarPos("Confidence %", "PickerBot")
            confidence = max(conf_pct / 100.0, 0.01)

            annotated, results = detect_and_annotate(frame, confidence)
            locations = filter_by_boq(results, boq)

            cv2.imshow("PickerBot", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                break
            elif key == ord('q'):
                locations = []
                break

        cap.release()
        cv2.destroyAllWindows()
        return locations
    else:
        frame = cv2.imread(src)
        if frame is None:
            print(f"Error: Could not read image '{src}'")
            sys.exit(1)

        annotated, results = detect_and_annotate(frame, CONFIG["min_confidence"])
        locations = filter_by_boq(results, boq)

        cv2.imshow("PickerBot - Detected Objects", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return locations


def resolve_src(arg_src):
    """Resolve the input source. CLI arg wins; otherwise fall back to config.json."""
    if arg_src is not None:
        return arg_src

    mode = CONFIG.get("input_mode", "camera").lower()
    if mode in ("camera", "webcam"):
        return "camera"
    return resolve(CONFIG["test_image_path"])


def main():
    parser = argparse.ArgumentParser(description="PickerBot: Detect, classify, and pick objects.")
    parser.add_argument("--src", default=None, help="'camera' for live feed, or path to an image file. Defaults to config.json input_mode.")
    parser.add_argument("--calibrate", action="store_true", help="Run height recalibration using ruler.jpg")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration_gui()
        return

    default_csv = resolve("data/calibration/calibration_pixels.csv")
    csv_file = resolve(CONFIG["calibration_file"]) if CONFIG.get("calibration_file") else default_csv
    src_pts, dst_pts = load_calibration_data(csv_file)
    H = calculate_homography(src_pts, dst_pts)
    print(f"Homography calibration loaded from {os.path.basename(csv_file)}.\n")

    src = resolve_src(args.src)
    pixel_locations = run_detection(src)

    if not pixel_locations:
        print("No objects detected. Nothing to pick.")
        return

    epson_points = translate_points(pixel_locations, H)

    print(f"\nDetected {len(epson_points)} object(s):")
    for loc in epson_points:
        print(f"  {loc}")

    if not CONFIG.get("enable_epson_tcp", False):
        print("\n[DRY RUN] enable_epson_tcp is disabled in config.json — skipping TCP dispatch.")
        return

    try:
        connect(CONFIG["epson_ip"], CONFIG["epson_port"])
        epsonPickAll(epson_points)
    finally:
        disconnect()


if __name__ == "__main__":
    main()
