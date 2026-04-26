import numpy as np
import cv2
import csv
import os
import sys
import json
import math

from pickerbot_lib.config import PROJECT_ROOT as _project_root

# --- CALIBRATION FUNCTIONS ---
def load_calibration_data(csv_filename):
    src_pixels, dst_world = [], []
    if not os.path.exists(csv_filename):
        print(f"Error: Calibration file '{csv_filename}' not found.")
        sys.exit(1)

    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) == 4:
                try:
                    src_pixels.append([float(row[0]), float(row[1])])
                    dst_world.append([float(row[2]), float(row[3])])
                except ValueError:
                    continue
    return np.array(src_pixels, dtype=np.float32), np.array(dst_world, dtype=np.float32)

def calculate_homography(src_pts, dst_pts):
    # Changed from cv2.RANSAC to 0 (Regular method using all points)
    # This prevents the algorithm from treating corner points as outliers
    H, status = cv2.findHomography(src_pts, dst_pts, 0)
    return H

def pixel_to_world(H, pixel_x, pixel_y):
    point = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, H)
    return round(world_point[0][0][0], 3), round(world_point[0][0][1], 3)

def run_calibration_gui():
    """Open ruler.jpg, let user click two points 20mm apart, and scale calibration data accordingly.

    Scales all pixel coordinates around point 39 (grid center) based on the ratio
    between the clicked pixel distance and the expected pixel distance for 20mm
    from the original calibration. Saves the scaled calibration to a new CSV and
    updates config.json to reference it.
    """
    # Load ruler image
    ruler_path = os.path.join(_project_root, "data", "test-samples", "ruler.jpg")
    ruler_img = cv2.imread(ruler_path)
    if ruler_img is None:
        print(f"Error: Could not load ruler image at {ruler_path}")
        return False

    # Scale to 720p to match the calibration resolution
    ruler_img = cv2.resize(ruler_img, (1280, 720))

    # Load original (standard) calibration data
    csv_path = os.path.join(_project_root, "data", "calibration", "calibration_pixels.csv")
    src_pts, dst_pts = load_calibration_data(csv_path)

    # Point 39 is index 38 (0-indexed) — center of the 11x7 grid
    center_px = src_pts[38].copy()

    # Compute reference pixel distances for 20mm from the standard calibration.
    # Grid is 11 columns of 7 rows each. Adjacent columns are 20mm apart (World_X),
    # adjacent rows are 20mm apart (World_Y).

    # Horizontal reference (X pixel distance per 20mm world)
    h_dists = []
    for col in range(10):
        for row in range(7):
            idx1 = col * 7 + row
            idx2 = (col + 1) * 7 + row
            h_dists.append(abs(src_pts[idx2][0] - src_pts[idx1][0]))
    ref_h = np.mean(h_dists)

    # Vertical reference (Y pixel distance per 20mm world)
    v_dists = []
    for col in range(11):
        for row in range(6):
            idx1 = col * 7 + row
            idx2 = col * 7 + row + 1
            v_dists.append(abs(src_pts[idx2][1] - src_pts[idx1][1]))
    ref_v = np.mean(v_dists)

    print(f"Reference calibration: {ref_h:.1f}px/20mm (horizontal), {ref_v:.1f}px/20mm (vertical)")

    # --- GUI: click two points 20mm apart on the ruler ---
    clicked_points = []
    display = ruler_img.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
            clicked_points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            if len(clicked_points) == 2:
                cv2.line(display, clicked_points[0], clicked_points[1], (0, 255, 0), 2)
                dx = clicked_points[1][0] - clicked_points[0][0]
                dy = clicked_points[1][1] - clicked_points[0][1]
                px_dist = math.sqrt(dx * dx + dy * dy)
                mid = ((clicked_points[0][0] + clicked_points[1][0]) // 2,
                       (clicked_points[0][1] + clicked_points[1][1]) // 2)
                cv2.putText(display, f"{px_dist:.1f}px", (mid[0] + 10, mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, display)

    window_name = "Calibration - Click two points 20mm apart, Enter to confirm, R to reset, Q to cancel"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)

    print("Click two points on the ruler that are exactly 20mm apart.")
    print("Press Enter to confirm, R to reset, Q to cancel.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(clicked_points) == 2:  # Enter
            break
        elif key == ord('r'):
            clicked_points.clear()
            display = ruler_img.copy()
            cv2.imshow(window_name, display)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return False

    cv2.destroyAllWindows()

    # Compute scale factor, accounting for ruler orientation
    dx = clicked_points[1][0] - clicked_points[0][0]
    dy = clicked_points[1][1] - clicked_points[0][1]
    new_dist = math.sqrt(dx * dx + dy * dy)
    theta = math.atan2(abs(dy), abs(dx))

    # Expected pixel distance at this angle using the original calibration
    expected_dist = math.sqrt((ref_h * math.cos(theta)) ** 2 + (ref_v * math.sin(theta)) ** 2)
    scale = new_dist / expected_dist

    print(f"\nMeasured: {new_dist:.1f}px for 20mm (angle {math.degrees(theta):.1f} deg)")
    print(f"Expected: {expected_dist:.1f}px at that angle from standard calibration")
    print(f"Scale factor: {scale:.4f}")

    # Scale all pixel coordinates around point 39's pixel position
    scaled_pts = []
    for pt in src_pts:
        offset = pt - center_px
        scaled_pts.append(center_px + offset * scale)
    scaled_pts = np.array(scaled_pts, dtype=np.float32)

    # Save scaled calibration to new CSV (world coordinates unchanged)
    scaled_csv = os.path.join(_project_root, "data", "calibration", "calibration_pixels_scaled.csv")
    with open(scaled_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Pixel_X", "Pixel_Y", "World_X", "World_Y"])
        for i in range(len(scaled_pts)):
            writer.writerow([
                round(float(scaled_pts[i][0]), 1),
                round(float(scaled_pts[i][1]), 1),
                int(dst_pts[i][0]),
                int(dst_pts[i][1])
            ])

    print(f"Scaled calibration saved to {scaled_csv}")

    # Update config.json to point to the scaled calibration file
    config_path = os.path.join(_project_root, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    config["calibration_file"] = "data/calibration/calibration_pixels_scaled.csv"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print("config.json updated to use scaled calibration.\n")
    return True
