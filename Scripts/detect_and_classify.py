import os
import sys
import cv2
import numpy as np

# Add Scripts directory to path so we can import keras_inference
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from keras_inference import ModuleClassifier

# ── Configuration ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 50000
DEFAULT_THRESH_VALUE = 100
CAMERA_ID = 1
# ───────────────────────────────────────────────────────────────

def detect_objects(frame, thresh_val):
    """Find contour bounding boxes in the frame using the same pipeline as cv_discovery.py."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h, contour))
    return boxes, thresh


def classify_and_annotate(frame, boxes, classifier):
    """Crop each bounding box, classify it, and draw the label on the frame."""
    results = []
    frame_h, frame_w = frame.shape[:2]
    for (x, y, w, h, contour) in boxes:
        # Place the crop onto a white canvas at original pixel size so the
        # classifier sees it the same way as the training images — no upscaling blur
        crop = frame[y:y+h, x:x+w]
        canvas = np.full((frame_h, frame_w, 3), 255, dtype=np.uint8)
        canvas_cx, canvas_cy = frame_w // 2, frame_h // 2
        paste_x = canvas_cx - w // 2
        paste_y = canvas_cy - h // 2
        canvas[paste_y:paste_y+h, paste_x:paste_x+w] = crop
        label, confidence = classifier.predict(canvas)

        # Skip empty or low-confidence detections
        if label == "empty" or confidence < CONFIDENCE_THRESHOLD:
            continue

        # Centroid of the bounding box
        centroid_x = x + w // 2
        centroid_y = y + h // 2

        # Orientation angle from the minimum area rotated rectangle
        rect = cv2.minAreaRect(contour)
        (_, _), (rect_w, rect_h), angle = rect
        # Normalize so angle is relative to horizontal X axis [0, 180)
        if rect_w < rect_h:
            angle = angle + 90

        results.append((x, y, w, h, centroid_x, centroid_y, angle, label, confidence))

        # Draw green upright bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Draw red rotated bounding box aligned with item orientation
        rot_box = cv2.boxPoints(rect)
        rot_box = np.int32(rot_box)
        cv2.drawContours(frame, [rot_box], 0, (0, 0, 255), 2)

        # Draw centroid dot
        cv2.circle(frame, (centroid_x, centroid_y), 4, (0, 0, 255), -1)

        # Draw label with centroid and angle
        text = f"{label} {confidence*100:.1f}% ({centroid_x},{centroid_y}) {angle:.1f} deg"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), (0, 255, 0), -1)
        cv2.putText(frame, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    return frame, results


def process_image(image_path, classifier):
    """Run detection and classification on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image '{image_path}'")
        sys.exit(1)

    boxes, _ = detect_objects(frame, DEFAULT_THRESH_VALUE)
    annotated, results = classify_and_annotate(frame, boxes, classifier)

    locations = [f"{cx} {cy} {angle:.1f}" for (x, y, w, h, cx, cy, angle, label, conf) in results]
    print(locations)

    cv2.imshow("Detected Objects", annotated)
    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_camera(classifier):
    """Run detection and classification on a live camera feed."""
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}.")
        return

    print("Camera loaded. Press 'q' to quit.")

    cv2.namedWindow("Detect & Classify")
    cv2.createTrackbar("Thresh", "Detect & Classify", DEFAULT_THRESH_VALUE, 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        thresh_val = cv2.getTrackbarPos("Thresh", "Detect & Classify")
        boxes, thresh_view = detect_objects(frame, thresh_val)
        annotated, _ = classify_and_annotate(frame, boxes, classifier)

        cv2.imshow("Detect & Classify", annotated)
        cv2.imshow("Threshold", thresh_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    model_folder = os.path.join(script_dir, "..", "models", "cw_keras")
    classifier = ModuleClassifier(model_folder)

    # Warm up the model (first prediction is always slow)
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    classifier.predict(dummy)
    print("Model warmed up and ready.\n")

    if len(sys.argv) > 1:
        process_image(sys.argv[1], classifier)
    else:
        process_camera(classifier)


if __name__ == "__main__":
    main()
