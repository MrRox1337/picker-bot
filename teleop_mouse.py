import os
import sys
import json
import cv2
from pickerbot_lib.sender import connect, disconnect, epsonMove
from pickerbot_lib.calibration import load_calibration_data, calculate_homography, pixel_to_world

# --- 1. EPSON TCP/IP SETTINGS ---
ip_adddress = "127.0.0.1" # simulator robot
port = 2001
robot_z = 360 # default z height for picking


def mouse_click(event, x, y, flags, param):
    """Callback function for mouse clicks on the OpenCV window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        H_matrix, z_height, img_display = param

        # Translate pixel to world coordinates
        world_x, world_y = pixel_to_world(H_matrix, x, y)
        print(f"\nTarget Selected - Pixel: ({x}, {y}) | EPSON World: X={world_x}, Y={world_y}")

        # Draw a visual marker for the click (blue circle)
        cv2.circle(img_display, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.imshow("Interactive PickerBot (Click to Move, 'q' to quit)", img_display)

        # Send MOVE command
        epsonMove(world_x, world_y, z_height)

def main():
    # Load calibration matrix
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load config to check for scaled calibration
    config_path = os.path.join(project_root, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Use scaled calibration if available, otherwise default
    default_csv = os.path.join(project_root, "data", "calibration", "calibration_pixels.csv")
    csv_file = os.path.join(project_root, config["calibration_file"]) if config.get("calibration_file") else default_csv

    print("1. Loading calibration matrix...")
    src_pts, dst_pts = load_calibration_data(csv_file)
    H_matrix = calculate_homography(src_pts, dst_pts)
    print(f"   Using: {os.path.basename(csv_file)}")

    # Connect to Robot
    try:
        connect(ip_adddress, port)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("\n3. System Ready! Loading Image UI...")

    # Load ruler.jpg if using scaled calibration, otherwise graph_paper.jpg
    if config.get("calibration_file"):
        img_path = os.path.join(project_root, "data", "test-samples", "ruler.jpg")
    else:
        img_path = os.path.join(project_root, "data", "calibration", "graph_paper.jpg")
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not load image at {img_path}")
        sys.exit(1)

    # Ensure it matches the 720p calibration size
    img = cv2.resize(img, (1280, 720))

    # Draw the calibration points (labels) on the image using data from the CSV
    for i, (px, py) in enumerate(src_pts):
        px, py = int(px), int(py)
        cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, str(i + 1), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    img_display = img.copy()
    window_name = "Interactive PickerBot (Click to Move, 'q' to quit)"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click, param=(H_matrix, robot_z, img_display))

    cv2.imshow(window_name, img_display)
    print("Click anywhere on the image to send the robot to that location.")
    print("Press 'q' on your keyboard to quit.")

    # Wait loop
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nClosing connection.")
    disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
