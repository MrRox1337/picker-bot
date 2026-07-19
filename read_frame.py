import pyrealsense2 as rs
import numpy as np

BAG = "20260719_123818.db3"

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(BAG, repeat_playback=False)
pipe.start(cfg)

align = rs.align(rs.stream.color)          # line depth up with colour
frames = align.process(pipe.wait_for_frames())

color = np.asanyarray(frames.get_color_frame().get_data())
depth = frames.get_depth_frame()
h, w = np.asanyarray(depth.get_data()).shape
dist_m = depth.get_distance(w // 2, h // 2)   # distance at the centre pixel

print("Colour frame :", color.shape)          # (height, width, 3)
print("Depth frame  :", (h, w))
print(f"Centre pixel is {dist_m:.3f} m away")

pipe.stop()