import cv2
import csv
import sys
import os

# Global list to store the clicked (x, y) pixel coordinates
points = []

def mouse_click(event, x, y, flags, param):
    """Callback function triggered on mouse events."""
    global points, img_display

    # Listen for Left Mouse Button Down
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # Draw a solid red circle at the clicked point
        cv2.circle(img_display, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        # Add a green text label with the coordinates slightly offset
        label = f"({x}, {y})"
        cv2.putText(img_display, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Update the window with the new drawings
        cv2.imshow("Calibration Grid (Press 's' to save, 'q' to quit)", img_display)
        print(f"Point {len(points)}/77 recorded: {x}, {y}")

def main():
    global img_display

    # Get project root (one level up from tools/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the full paths for the image and the output CSV
    image_path = os.path.join(project_root, "data", "calibration", "graph_paper.jpg")
    csv_filename = os.path.join(project_root, "data", "calibration", "calibration_pixels.csv")

    print(f"Looking for image at: {image_path}")

    # 1. Load the image
    img_original = cv2.imread(image_path)

    if img_original is None:
        print(f"Error: Could not load image. Please ensure 'graph_paper.jpg' is in data/calibration/")
        sys.exit(1)

    # Resize the image to 720p (1280x720)
    img_original = cv2.resize(img_original, (1280, 720))

    # Make a copy for displaying so we don't overwrite the original file data
    img_display = img_original.copy()

    # 2. Setup the UI Window and Mouse Callback
    window_name = "Calibration Grid (Press 's' to save, 'q' to quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    print("Click on the grid points. Press 's' to save and exit, or 'q' to abort.")
    cv2.imshow(window_name, img_display)

    # 3. Wait loop for keyboard commands
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' or 'ESC' to quit without saving
        if key == ord('q') or key == 27:
            print("\nCalibration aborted. No CSV created.")
            break

        # Press 's' to save to CSV and exit
        elif key == ord('s'):
            if len(points) == 0:
                print("\nNo points selected! Exiting without saving.")
                break

            # 1. Sort all points purely by X to broadly order them left-to-right
            points_by_x = sorted(points, key=lambda p: p[0])

            # 2. Group into columns of 7 and sort each column strictly by Y (top-to-bottom)
            sorted_points = []
            points_per_column = 7

            for i in range(0, len(points_by_x), points_per_column):
                # Extract the 7 points that belong to this specific column
                column_chunk = points_by_x[i:i + points_per_column]

                # Sort these 7 points by their Y value
                column_sorted_by_y = sorted(column_chunk, key=lambda p: p[1])

                # Add them to our final sorted list
                sorted_points.extend(column_sorted_by_y)

            # Export to CSV
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Pixel_X', 'Pixel_Y']) # Header row
                writer.writerows(sorted_points)

            print(f"\nSuccess! {len(points)} points sorted and saved to {csv_filename}")
            break

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
