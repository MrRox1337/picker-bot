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

# fit the table plane, then measure each point's height ABOVE it
xs, ys, zs = X[valid], Y[valid], Z[valid]
a, b, c = np.linalg.lstsq(np.column_stack([xs, ys, np.ones_like(xs)]), zs, rcond=None)[0]
height = (a * X + b * Y + c) - Z         # metres; ~0 on the table, positive on raised parts
height[~valid] = 0

# colour by height: table (0) = blue, tall parts = red.  Change the 30 if parts look dim/washed out.
hmm = np.clip(height * 1000.0, 0, 30)    # height in millimetres, clipped to 0..30 mm
vis = cv2.applyColorMap((hmm / 30 * 255).astype(np.uint8), cv2.COLORMAP_JET)
vis[~valid] = 0                           # black where there's no depth reading

cv2.imwrite("color.png", color)
cv2.imwrite("height.png", vis)
print("Saved color.png and height.png — table should be flat blue now, parts standing out.")
pipe.stop()