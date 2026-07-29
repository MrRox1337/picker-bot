import pyrealsense2 as rs
import numpy as np
import cv2

BAG = "20260719_123818.db3"

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(BAG, repeat_playback=False)
pipe.start(cfg)

align = rs.align(rs.stream.color)
frames = align.process(pipe.wait_for_frames())

color = np.asanyarray(frames.get_color_frame().get_data())
depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)

# rainbow depth image, scaled to this scene so the parts stand out
valid = depth > 0
lo, hi = np.percentile(depth[valid], [2, 98])
norm = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
norm[~valid] = 0
depth_vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

cv2.imwrite("color.png", color)
cv2.imwrite("depth.png", depth_vis)
print("Saved color.png and depth.png — open them and take a look.")
pipe.stop()