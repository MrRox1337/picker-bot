import pyrealsense2 as rs
import numpy as np
import cv2

BAG = "20260719_123818.db3"

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(BAG, repeat_playback=False)
profile = pipe.start(cfg)

align = rs.align(rs.stream.color)
frames = align.process(pipe.wait_for_frames())

color = np.asanyarray(frames.get_color_frame().get_data())
depth_frame = frames.get_depth_frame()
depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
h, w = depth.shape
uu, vv = np.meshgrid(np.arange(w), np.arange(h))
Z = depth
X = (uu - ppx) / fx * Z
Y = (vv - ppy) / fy * Z
valid = Z > 0

xs, ys, zs = X[valid], Y[valid], Z[valid]
a, b, c = np.linalg.lstsq(np.column_stack([xs, ys, np.ones_like(xs)]), zs, rcond=None)[0]
height = (a * X + b * Y + c) - Z
height[~valid] = 0
height_mm = height * 1000.0

# --- find the parts sitting above the table ---
THRESH_MM = 5      # a pixel is "part" if it's >5 mm above the table
MIN_AREA  = 300    # ignore blobs smaller than this many pixels (noise)

mask = (height_mm > THRESH_MM).astype(np.uint8) * 255
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove speckle
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes

n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

out = color.copy()
count = 0
for i in range(1, n):                       # skip label 0 = background
    if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
        continue
    count += 1
    x, y, bw, bh = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
    cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

cv2.imwrite("parts.png", out)
print(f"Found {count} parts on the table. Saved parts.png")
pipe.stop()