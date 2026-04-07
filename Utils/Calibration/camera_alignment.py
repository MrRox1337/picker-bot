import cv2
import csv
import sys
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full paths for the image and the CSV
    image_path = os.path.join(script_dir, "graph_paper.jpg")
    csv_filename = os.path.join(script_dir, "calibration_pixels.csv")
    
    # 1. Load and resize the image
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image. Please ensure 'graph_paper.jpg' is in {script_dir}")
        sys.exit(1)

    # Resize to 720p (1280x720) just like in the calibration step
    img = cv2.resize(img, (1280, 720))
    
    # 2. Read the points from the CSV
    points = []
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find '{csv_filename}'.")
        sys.exit(1)
        
    print(f"Reading points from: {csv_filename}")
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header
        
        for row in reader:
            if len(row) >= 2:
                try:
                    x, y = int(row[0]), int(row[1])
                    points.append((x, y))
                except ValueError:
                    continue

    if not points:
        print("No valid points found in the CSV.")
        sys.exit(1)

    print(f"Successfully loaded {len(points)} points. Drawing on image...")

    # 3. Draw the points and their order numbers on the image
    for i, (x, y) in enumerate(points):
        # Draw a solid red circle at the coordinate
        cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        
        # Write the order number (1 to 77) in green slightly offset from the point
        # This helps visually verify the chunked sorting is correct
        label = str(i + 1)
        cv2.putText(img, label, (x + 5, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 4. Display the result
    window_name = "Saved Calibration Points (Press any key to close)"
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, img)
    
    print("Image displayed. Press any key while focused on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()