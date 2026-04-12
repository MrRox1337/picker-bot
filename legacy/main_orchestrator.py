import os
import sys
import json
import socket
import csv
import math
from time import sleep

import cv2
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ultralytics import YOLO

def load_calibration_data(csv_filename):
    src_pixels, dst_world = [], []
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

def pixel_to_world(H, pixel_x, pixel_y):
    point = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, H)
    return round(world_point[0][0][0], 3), round(world_point[0][0][1], 3)

def epsonPick(sock, x, y, z, u):
    """Sends the actual PICK command to EPSON RC+"""
    coordinates = f"PICK {x} {y} {z} {u}\r\n"
    sock.send(coordinates.encode())
    confirmation = sock.recv(1023)
    return confirmation.decode().strip()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.json")
    
    # 1. LOAD CONFIGURATION
    if not os.path.exists(config_path):
        print("Error: config.json not found in project root.")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print(f"\n--- YOLOv8-OBB PIPELINE BOOT SEQUENCE ---")
    
    # 2. LOAD AI MODEL (YOLOv8-OBB)
    model_path = os.path.join(script_dir, "..", "models", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: OBB model not found at {model_path}")
        return
    model = YOLO(model_path)
    print("-> YOLO OBB Core initialized.")

    # 3. LOAD HOMOGRAPHY MATRIX
    csv_file = os.path.join(script_dir, "..", "data", "calibration", "calibration_pixels.csv")
    src_pts, dst_pts = load_calibration_data(csv_file)
    H_matrix, _ = cv2.findHomography(src_pts, dst_pts, 0)
    print("-> Spatial Homography mapped.")
    
    # 4. HANDLE TCP/IP CONNECTION IF ENABLED
    clientSocket = None
    if config.get("enable_epson_tcp", False):
        ip = config.get("epson_ip", "127.0.0.1")
        port = config.get("epson_port", 2001)
        print(f"\n[TCP] Connecting to real EPSON Hardware at {ip}:{port}...")
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientSocket.connect((ip, port))
    else:
        print("\n[DRY RUN] enable_epson_tcp is disabled. Simulating movements.")

    # 5. FETCH FRAME
    input_mode = config.get("input_mode", "image")
    if input_mode == "image":
        img_path = os.path.join(script_dir, "..", config["test_image_path"])
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error loading image: {img_path}")
            return
    else:
        cap = cv2.VideoCapture(config.get("webcam_id", 1))
        # Hardcoding the 720p resolution for CV2 Capture buffer as requested
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture webcam.")
            return

    # 6. RUN YOLO-OBB INFERENCE ENGINE
    min_confidence = config.get("min_confidence", 0.70)
    
    # Single unified engine call. Replaces 60 lines of OpenCV discovery code!
    results = model(frame, conf=min_confidence, verbose=False)
    result = results[0]
    
    # YOLO allows rendering incredibly beautiful angled bounding boxes natively!
    display_frame = result.plot()
    
    print("\n--- INFERENCE RESULTS ---")
    
    # 7. PARSE TARGETS AND TRANSMIT
    if result.obb is not None:
        for idx, box in enumerate(result.obb):
            # Ultralytics natively returns an xywhr tensor for OBB:
            # [center_x, center_y, width, height, rotation_radians]
            xywhr = box.xywhr[0]
            cx, cy, w, h, theta_rads = float(xywhr[0]), float(xywhr[1]), float(xywhr[2]), float(xywhr[3]), float(xywhr[4])
            
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            
            if label.lower() == "noise":
                print(f"=> IGNORED: 'noise' (Background Garbage)")
                continue
                
            print(f"\n=> FOUND: '{label}' at {conf*100:.1f}% confidence!")
            print(f"   Shape: [W: {w:.1f}, H: {h:.1f}, Angle: {math.degrees(theta_rads):.1f} deg]")
            
            # Math Translation natively mapped from YOLO's bounding center
            world_x, world_y = pixel_to_world(H_matrix, cx, cy)
            robot_z = 360 # Default table height
            
            # Convert Native OBB Radians to EPSON Degrees
            robot_u = round(math.degrees(theta_rads), 2)
            
            if clientSocket:
                print(f"--> [TCP] Firing command: PICK {world_x} {world_y} {robot_z} {robot_u}")
                reply = epsonPick(clientSocket, world_x, world_y, robot_z, robot_u)
                print(f"--> [TCP] Robot replied: {reply}")
            else:
                print(f"--> [DRY RUN] Would send: PICK X={world_x} Y={world_y} Z={robot_z} U={robot_u}")
    else:
        print("No oriented objects detected above the confidence threshold.")

    print("\n--- SCAN COMPLETE ---")
    
    # Show user what we saw during dry run
    cv2.imshow("Orchestrator YOLO-OBB Vision Core", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if clientSocket:
        clientSocket.close()

if __name__ == '__main__':
    main()
