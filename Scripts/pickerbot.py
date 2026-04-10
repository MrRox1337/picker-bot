import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from detect_and_classify import detect_objects, classify_and_annotate, DEFAULT_THRESH_VALUE
from keras_inference import ModuleClassifier
from teleop_mouse import load_calibration_data, calculate_homography, pixel_to_world
from pickerbot_sender import connect, disconnect, epsonPickAll
import numpy as np
import cv2

# ── Configuration ──────────────────────────────────────────────
EPSON_IP = "127.0.0.1"
EPSON_PORT = 2001
ROBOT_Z = 360
CAMERA_ID = 1
# ───────────────────────────────────────────────────────────────

def pixels_to_world_locations(pixel_locations, H):
    """Convert pixel location strings 'cx cy angle' to world coordinate strings 'wx wy z angle'."""
    world_locations = []
    for loc in pixel_locations:
        px, py, angle = loc.split()
        wx, wy = pixel_to_world(H, float(px), float(py))
        world_locations.append(f"{wx} {wy} {ROBOT_Z} {angle}")
    return world_locations


def run_detection(src, classifier):
    """Run object detection on a camera frame or image file. Returns pixel locations and annotated frame."""
    if src == "camera":
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print(f"Error: Could not open camera {CAMERA_ID}.")
            sys.exit(1)

        print("Camera loaded. Press 'c' to capture and pick, 'q' to quit.")
        cv2.namedWindow("PickerBot")
        cv2.createTrackbar("Thresh", "PickerBot", DEFAULT_THRESH_VALUE, 255, lambda x: None)

        locations = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            thresh_val = cv2.getTrackbarPos("Thresh", "PickerBot")
            boxes, thresh_view = detect_objects(frame, thresh_val)
            annotated, results = classify_and_annotate(frame.copy(), boxes, classifier)

            cv2.imshow("PickerBot", annotated)
            cv2.imshow("Threshold", thresh_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                locations = [f"{cx} {cy} {angle:.1f}" for (_, _, _, _, cx, cy, angle, _, _) in results]
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return locations
    else:
        frame = cv2.imread(src)
        if frame is None:
            print(f"Error: Could not read image '{src}'")
            sys.exit(1)

        boxes, _ = detect_objects(frame, DEFAULT_THRESH_VALUE)
        annotated, results = classify_and_annotate(frame, boxes, classifier)
        locations = [f"{cx} {cy} {angle:.1f}" for (_, _, _, _, cx, cy, angle, _, _) in results]

        cv2.imshow("PickerBot - Detected Objects", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return locations


def main():
    parser = argparse.ArgumentParser(description="PickerBot: Detect, classify, and pick objects.")
    parser.add_argument("--src", default="camera", help="'camera' for live feed, or path to an image file")
    args = parser.parse_args()

    # 1. Load classifier
    model_folder = os.path.join(script_dir, "..", "models", "cw_keras")
    classifier = ModuleClassifier(model_folder)
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    classifier.predict(dummy)
    print("Model warmed up and ready.\n")

    # 2. Load homography calibration
    csv_file = os.path.join(script_dir, "..", "Utils", "Calibration", "calibration_pixels.csv")
    src_pts, dst_pts = load_calibration_data(csv_file)
    H = calculate_homography(src_pts, dst_pts)
    print("Homography calibration loaded.\n")

    # 3. Detect objects
    pixel_locations = run_detection(args.src, classifier)

    if not pixel_locations:
        print("No objects detected. Nothing to pick.")
        return

    # 4. Convert pixel locations to world coordinates
    world_locations = pixels_to_world_locations(pixel_locations, H)

    print(f"\nDetected {len(world_locations)} object(s):")
    for loc in world_locations:
        print(f"  {loc}")

    # 5. Connect and batch pick
    try:
        connect(EPSON_IP, EPSON_PORT)
        epsonPickAll(world_locations)
    finally:
        disconnect()


if __name__ == "__main__":
    main()
