import pyrealsense2 as rs
import numpy as np
import cv2

# ---- settings ----
BAG = "pile_nearby.db3"
THRESH_MM = 5               # "part" if >5 mm above table (lower to 3 if parts missed)
MIN_AREA  = 300             # ignore blobs smaller than this (raise to 800 if noise boxes)

# 1. read one aligned colour + depth frame
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_device_from_file(BAG, repeat_playback=False)
profile = pipe.start(cfg)
frames = rs.align(rs.stream.color).process(pipe.wait_for_frames())
color = np.asanyarray(frames.get_color_frame().get_data())
dframe = frames.get_depth_frame()
depth = np.asanyarray(dframe.get_data()).astype(np.float32) * dframe.get_units()   # metres

# 2. pixels -> real 3D points (metres)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
h, w = depth.shape
uu, vv = np.meshgrid(np.arange(w), np.arange(h))
Z = depth; X = (uu - ppx)/fx*Z; Y = (vv - ppy)/fy*Z
valid = Z > 0

# 3. fit table plane + measure tilt
a, b, c = np.linalg.lstsq(np.column_stack([X[valid], Y[valid], np.ones(valid.sum())]),
                          Z[valid], rcond=None)[0]
tilt_deg = np.degrees(np.arctan(np.hypot(a, b)))

# 4. height above the table
height_mm = ((a*X + b*Y + c) - Z) * 1000.0
height_mm[~valid] = 0

# 5. save pictures
lo, hi = np.percentile(Z[valid], [2, 98])
dnorm = np.clip((Z - lo)/(hi - lo + 1e-6), 0, 1); dnorm[~valid] = 0
cv2.imwrite("color.png", color)
cv2.imwrite("depth.png", cv2.applyColorMap((dnorm*255).astype(np.uint8), cv2.COLORMAP_JET))
hvis = cv2.applyColorMap((np.clip(height_mm,0,30)/30*255).astype(np.uint8), cv2.COLORMAP_JET)
hvis[~valid] = 0
cv2.imwrite("height.png", hvis)

# 6. detect parts above the table
mask = (height_mm > THRESH_MM).astype(np.uint8)*255
k = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
out = color.copy(); count = 0
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < MIN_AREA: continue
    count += 1
    x, y, bw, bh = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
    cv2.rectangle(out, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
cv2.imwrite("parts.png", out)

print(f"Tilt at capture pose : {tilt_deg:.1f} deg")
print(f"Blobs above table    : {count}")
print("Saved: color.png, depth.png, height.png, parts.png")
pipe.stop()