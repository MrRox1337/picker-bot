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
depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()  # metres

# the camera's lens numbers, straight from the recording
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

# turn every pixel into a real 3D point (X, Y, Z) in metres
h, w = depth.shape
uu, vv = np.meshgrid(np.arange(w), np.arange(h))
Z = depth
X = (uu - ppx) / fx * Z
Y = (vv - ppy) / fy * Z

print("3D points ready — shape:", Z.shape)
cx, cy = w // 2, h // 2
print("Centre point (X, Y, Z) m:", round(float(X[cy, cx]), 3),
      round(float(Y[cy, cx]), 3), round(float(Z[cy, cx]), 3))
valid = Z > 0
print("Z range across the scene: {:.3f} m to {:.3f} m".format(Z[valid].min(), Z[valid].max()))

pipe.stop()