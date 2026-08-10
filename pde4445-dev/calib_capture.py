"""
Hand-eye calibration — CAMERA SIDE capture helper.

Run this on the bench with the arm parked at the CAPTURE POSE (camera live).
Click each table marker; it reads the depth at that pixel, converts to a 3D
point in the CAMERA frame (metres), and logs it to calib_pairs.csv.

Then (separately) touch each marker with the tool tip, read the robot X/Y/Z
from Epson RC+ Jog & Teach, and type those into the robot_* columns for the
matching id. Bring calib_pairs.csv home and we solve the camera->robot transform.

Keys:  [q] quit and save
"""
import csv, os
import numpy as np
import cv2
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(__file__), "calib_pairs.csv")

pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

rows = []            # each: [id, u, v, cam_x_m, cam_y_m, cam_z_m]
state = {"depth": None, "units": 0.001}

def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    d = state["depth"]
    patch = d[max(0, y-3):y+4, max(0, x-3):x+4].astype(float)
    patch = patch[patch > 0]
    if patch.size == 0:
        print("  no valid depth at that pixel — click a slightly different spot")
        return
    z = float(np.median(patch)) * state["units"]                 # metres
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [x, y], z)   # camera-frame metres
    i = len(rows)
    rows.append([i, x, y, round(X, 4), round(Y, 4), round(Z, 4)])
    print(f"marker {i}: camera XYZ (m) = ({X:.4f}, {Y:.4f}, {Z:.4f})")

cv2.namedWindow("calib  [click markers]  [q]=quit", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("calib  [click markers]  [q]=quit", on_click)
try:
    while True:
        f = align.process(pipe.wait_for_frames())
        color = np.asanyarray(f.get_color_frame().get_data())
        state["depth"] = np.asanyarray(f.get_depth_frame().get_data())
        state["units"] = f.get_depth_frame().get_units()
        disp = color.copy()
        for r in rows:
            cv2.circle(disp, (r[1], r[2]), 5, (0, 0, 255), -1)
            cv2.putText(disp, str(r[0]), (r[1]+7, r[2]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("calib  [click markers]  [q]=quit", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipe.stop(); cv2.destroyAllWindows()

with open(OUT, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["id", "u", "v", "cam_x_m", "cam_y_m", "cam_z_m",
                "robot_x_mm", "robot_y_mm", "robot_z_mm"])
    for r in rows:
        w.writerow(r + ["", "", ""])
print(f"\nSaved {len(rows)} camera points -> {OUT}")
print("Next: touch each marker with the tool tip, read robot X/Y/Z from Jog & Teach,")
print("and fill robot_x_mm / robot_y_mm / robot_z_mm for each id.")
