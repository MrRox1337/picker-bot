import cv2
import numpy as np

def main():
    # Attempt to open the external USB Camera. 
    # Usually '0' is your laptop's internal webcam, and '1' is the plug-in USB camera.
    # If this fails or opens the wrong camera, change 1 to 0 or 2.
    cam_id = 1
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_id}.")
        return

    print("Camera loaded. Press 'q' to quit.")

    # Create a window so we can add trackbars to tune the threshold easily!
    cv2.namedWindow('Threshold Tuning')
    
    # We create a dummy function because trackbars require a callback
    def nothing(x):
        pass
        
    # Create slider to adjust the threshold value live
    # If the workbench is light and parts are dark, a threshold of ~100 usually works.
    cv2.createTrackbar('Thresh Value', 'Threshold Tuning', 100, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 1. Grayscale & Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Thresholding
        # Get the current value from the slider
        thresh_val = cv2.getTrackbarPos('Thresh Value', 'Threshold Tuning')
        
        # Here we use THRESH_BINARY_INV assuming: Workbench is LIGHT, Modules are DARK.
        # If your workspace is dark and modules are light, change this to cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # 3. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter out tiny specs of dust or huge shadows
            if 500 < area < 50000:
                
                # --- A. Extacting EPSON Math ---
                rect = cv2.minAreaRect(contour)
                # Draw the angled box for EPSON in RED
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                
                # Get center and angle for robot
                (cx, cy), (w, h), angle = rect
                cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

                # --- B. Extracting AI Crop Box ---
                x, y, crop_w, crop_h = cv2.boundingRect(contour)
                # Draw the upright box for AI cropping in GREEN
                cv2.rectangle(frame, (x, y), (x + crop_w, y + crop_h), (0, 255, 0), 2)
                
                # We will implement the actual 224x224 extraction later once the thresholding works!

        # Show the raw frame with drawings, and the binary threshold view
        cv2.imshow('Live Feed (Red=EPSON Rotated Box, Green=AI Crop Box)', frame)
        cv2.imshow('Threshold Tuning', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
