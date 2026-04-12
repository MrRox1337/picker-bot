import os
import sys
import json
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from detect_and_classify import detect_and_annotate, DEFAULT_CONFIDENCE
from teleop_mouse import load_calibration_data, calculate_homography, pixel_to_world, run_calibration_gui
from pickerbot_sender import connect, disconnect, epsonPickAll
import cv2

# ── Load configuration from config.json ───────────────────────
config_path = os.path.join(script_dir, "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

def translate_points(pixel_locations, H):
    """Convert pixel location strings 'cx cy angle' to world coordinate strings 'wx wy z angle'."""
    world_locations = []
    for loc in pixel_locations:
        px, py, angle = loc.split()
        wx, wy = pixel_to_world(H, float(px), float(py))
        world_locations.append(f"{wx} {wy} {config["robot_z"]} {angle}")
    return world_locations


def run_detection(src):
    """Run YOLO OBB detection on a camera frame or image file. Returns pixel locations."""
    if src == "camera":
        cap = cv2.VideoCapture(config["webcam_id"])
        if not cap.isOpened():
            print(f"Error: Could not open camera {config["webcam_id"]}.")
            sys.exit(1)

        print("Camera loaded. Press 'c' to capture and pick, 'q' to quit.")
        cv2.namedWindow("PickerBot")
        cv2.createTrackbar("Confidence %", "PickerBot", int(DEFAULT_CONFIDENCE * 100), 100, lambda x: None)

        locations = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            conf_pct = cv2.getTrackbarPos("Confidence %", "PickerBot")
            confidence = max(conf_pct / 100.0, 0.01)

            annotated, results = detect_and_annotate(frame, confidence)
            locations = [f"{cx} {cy} {angle:.1f}" for (cx, cy, angle, _, _) in results]

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

        annotated, results = detect_and_annotate(frame)
        locations = [f"{cx} {cy} {angle:.1f}" for (cx, cy, angle, _, _) in results]

        cv2.imshow("PickerBot - Detected Objects", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return locations


def main():
    parser = argparse.ArgumentParser(description="PickerBot: Detect, classify, and pick objects.")
    parser.add_argument("--src", default="camera", help="'camera' for live feed, or path to an image file")
    parser.add_argument("--calibrate", action="store_true", help="Run height recalibration using ruler.jpg")
    args = parser.parse_args()

    # 0. Run calibration if requested
    if args.calibrate:
        run_calibration_gui()
        return

    # 1. Load homography calibration (use config override if available, else default)
    default_csv = os.path.join(script_dir, "..", "Utils", "Calibration", "calibration_pixels.csv")
    csv_file = os.path.join(script_dir, "..", config.get("calibration_file", "")) if config.get("calibration_file") else default_csv
    src_pts, dst_pts = load_calibration_data(csv_file)
    H = calculate_homography(src_pts, dst_pts)
    print(f"Homography calibration loaded from {os.path.basename(csv_file)}.\n")

    # 2. Detect objects
    pixel_locations = run_detection(args.src)

    if not pixel_locations:
        print("No objects detected. Nothing to pick.")
        return

    # 3. Convert pixel locations to world coordinates
    epson_points = translate_points(pixel_locations, H)

    print(f"\nDetected {len(epson_points)} object(s):")
    for loc in epson_points:
        print(f"  {loc}")

    # 4. Connect and batch pick
    try:
        connect(config["epson_ip"], config["epson_port"])
        epsonPickAll(epson_points)
    finally:
        disconnect()


if __name__ == "__main__":
    main()
