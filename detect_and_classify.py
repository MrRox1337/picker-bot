import sys
import cv2

from pickerbot_lib import CONFIG
from pickerbot_lib.detection import detect_and_annotate


def process_image(image_path):
    """Run detection on a single image. Returns list of location strings 'cx cy angle'."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    annotated, results = detect_and_annotate(frame, CONFIG["min_confidence"])
    locations = [f"{cx} {cy} {angle:.1f}" for (cx, cy, angle, _, _) in results]

    cv2.imshow("Detected Objects", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return locations


def process_camera():
    """Run detection on live camera feed. Returns locations from last captured frame."""
    cap = cv2.VideoCapture(CONFIG["webcam_id"])
    if not cap.isOpened():
        print(f"Error: Could not open camera {CONFIG['webcam_id']}.")
        return []

    print("Camera loaded. Press 'q' to quit.")

    cv2.namedWindow("Detect & Classify")
    cv2.createTrackbar("Confidence %", "Detect & Classify", int(CONFIG["min_confidence"] * 100), 100, lambda x: None)

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
