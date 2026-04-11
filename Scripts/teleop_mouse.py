import numpy as np
import cv2
import csv
import os
import sys
from pickerbot_sender import connect, disconnect, epsonMove

# --- 1. EPSON TCP/IP SETTINGS ---
ip_adddress = "127.0.0.1" # simulator robot
port = 2001
robot_z = 360 # default z height for picking

# --- 2. CALIBRATION FUNCTIONS ---
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Adjust this path based on where your CSV actually is! 
    # (Assuming it's in Utils/Calibration relative to your main project folder)
    csv_file = os.path.join(script_dir, "..", "Utils", "Calibration", "calibration_pixels.csv")
    
    print("1. Loading calibration matrix...")
    src_pts, dst_pts = load_calibration_data(csv_file)
    H_matrix = calculate_homography(src_pts, dst_pts)
    
    # Connect to Robot
    try:
        connect(ip_adddress, port)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("\n3. System Ready! Loading Image UI...")
    
    # Load and prepare the graph paper image
    img_path = os.path.join(script_dir, "..", "Utils", "Calibration", "graph_paper.jpg")
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