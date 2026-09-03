"""
Step through a recording frame-by-frame, run the legacy YOLOv8-OBB model
(models/best.pt) on each frame, print what it detected, and (on your command)
save frames to pde4445-dev/IMG/ for building a test / training set.

Works with TWO sources — flip SOURCE below:
  "db3"  -> RealSense .db3 recording (has depth, but big files)
  "mp4"  -> a plain colour video from the camera (tiny files; great for dataset building)

Run from the repo root:   python pde4445-dev/yolo_test.py
Controls (in the image window):
  [p] keep this frame (save) + advance
  [s] skip this frame (don't save) + advance
  [q] quit
"""
import os
import sys
import numpy as np
import cv2

# --- make the repo's modules importable (this script lives in pde4445-dev/) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from pickerbot_lib.detection import detect_and_annotate  # loads models/best.pt on import

# ============================ settings you can tweak ============================
SOURCE   = "db3"          # "db3"  or  "mp4"   <-- flip this to switch input
DB3_PATH = os.path.join(os.path.dirname(__file__), "independence.db3")
MP4_PATH = os.path.join(os.path.dirname(__file__), "capture.mp4")
STEP     = 30             # frames to advance each keypress (bigger = more variety)
CONF     = 0.25           # detection confidence (low, so you see marginal hits)
OUTDIR   = os.path.join(os.path.dirname(__file__), "IMG")
os.makedirs(OUTDIR, exist_ok=True)

# ============================ open the chosen source ============================
if SOURCE == "db3":
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(DB3_PATH, repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)   # step manually
    align = rs.align(rs.stream.color)

    def next_color():
        """Advance STEP frames; return a BGR colour image, or None at end-of-file."""
        frames = None
        for _ in range(STEP):
            ok, frames = pipe.try_wait_for_frames(1000)
            if not ok:
                return None
        cframe = align.process(frames).get_color_frame()
        if not cframe:
            return None
        img = np.asanyarray(cframe.get_data())
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)            # RealSense=RGB -> OpenCV BGR

    def close():
        pipe.stop()

elif SOURCE == "mp4":
    cap = cv2.VideoCapture(MP4_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {MP4_PATH}")

    def next_color():
        """Advance STEP frames; return a BGR colour image, or None at end-of-file."""
        img = None
        for _ in range(STEP):
            ok, img = cap.read()
            if not ok:
                return None
        return img                                            # cv2 already gives BGR

    def close():
        cap.release()

else:
    raise SystemExit("SOURCE must be 'db3' or 'mp4'")

# ============================ step through the frames ============================
print(f"Source: {SOURCE}")
print("Controls:  [p] keep + next    [s] skip + next    [q] quit")
cv2.namedWindow("YOLO test", cv2.WINDOW_NORMAL)   # resizable, so big frames fit the screen

saved = 0     # counts frames actually saved (keeps filenames contiguous)
idx = 0       # counts frames stepped through (for the console)
color = next_color()

while color is not None:
    annotated, dets = detect_and_annotate(color, CONF)

    print(f"\n--- frame #{idx}  ({len(dets)} detections)   [saved so far: {saved}] ---")
    for (cx, cy, angle, label, conf) in dets:
        print(f"   {label:12s} {conf*100:5.1f}%   centre=({int(cx)},{int(cy)})   angle={angle:6.1f}")
    if not dets:
        print("   (nothing detected)")

    cv2.imshow("YOLO test", annotated)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):                                   # keep this frame
        cv2.imwrite(os.path.join(OUTDIR, f"frame_{saved:04d}.png"), color)
        saved += 1
        idx += 1
        color = next_color()
    elif key == ord('s'):                                   # skip without saving
        idx += 1
        color = next_color()
    # any other key: just re-show the same frame

close()
cv2.destroyAllWindows()
print(f"\nDone. Saved {saved} frames to {OUTDIR}")
