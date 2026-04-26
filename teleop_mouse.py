import os
import sys
import cv2

from pickerbot_lib import CONFIG, resolve
from pickerbot_lib.sender import connect, disconnect, epsonMove
from pickerbot_lib.calibration import load_calibration_data, calculate_homography, pixel_to_world


def mouse_click(event, x, y, flags, param):
    """Callback function for mouse clicks on the OpenCV window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        H_matrix, z_height, img_display = param

        world_x, world_y = pixel_to_world(H_matrix, x, y)
        print(f"\nTarget Selected - Pixel: ({x}, {y}) | EPSON World: X={world_x}, Y={world_y}")

        cv2.circle(img_display, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.imshow("Interactive PickerBot (Click to Move, 'q' to quit)", img_display)

        epsonMove(world_x, world_y, z_height)

def main():
    default_csv = resolve("data/calibration/calibration_pixels.csv")
    csv_file = resolve(CONFIG["calibration_file"]) if CONFIG.get("calibration_file") else default_csv

    print("1. Loading calibration matrix...")
    src_pts, dst_pts = load_calibration_data(csv_file)
    H_matrix = calculate_homography(src_pts, dst_pts)
    print(f"   Using: {os.path.basename(csv_file)}")

    try:
        connect(CONFIG["epson_ip"], CONFIG["epson_port"])
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("\n3. System Ready! Loading Image UI...")

    default_img = resolve("data/calibration/graph_paper.jpg")
    img_path = resolve(CONFIG["teleop_image"]) if CONFIG.get("teleop_image") else default_img
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not load image at {img_path}")
        sys.exit(1)

    img = cv2.resize(img, (1280, 720))

    for i, (px, py) in enumerate(src_pts):
        px, py = int(px), int(py)
        cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, str(i + 1), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    img_display = img.copy()
    window_name = "Interactive PickerBot (Click to Move, 'q' to quit)"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click, param=(H_matrix, CONFIG["robot_z"], img_display))

    cv2.imshow(window_name, img_display)
    print("Click anywhere on the image to send the robot to that location.")
    print("Press 'q' on your keyboard to quit.")

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nClosing connection.")
    disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
