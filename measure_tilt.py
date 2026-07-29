import pyrealsense2 as rs
import numpy as np

BAG = "20260719_123818.db3"

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(BAG, repeat_playback=False)
profile = pipe.start(cfg)

align = rs.align(rs.stream.color)
frames = align.process(pipe.wait_for_frames())

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

# fit a flat plane  Z = a*X + b*Y + c  (the table dominates the points)
a, b, c = np.linalg.lstsq(np.column_stack([xs, ys, np.ones_like(xs)]), zs, rcond=None)[0]

tilt_deg = np.degrees(np.arctan(np.sqrt(a * a + b * b)))
print(f"Table plane:  Z = {a:.3f}*X + {b:.3f}*Y + {c:.3f}")
print(f"Camera was tilted ~{tilt_deg:.1f} degrees off square to the table")

pipe.stop()