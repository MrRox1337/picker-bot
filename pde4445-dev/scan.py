"""
SCAN — produce a pick list from the LIVE camera or from a RECORDING.

Why this module exists separately from pose_seg.py:
  * The autonomous loop must re-scan mid-run, and writing a .db3 every time would
    be slow and would litter the disk. It needs a live single-frame grab.
  * The offline vision battery must run the SAME maths over recorded .db3 files.

So the geometry is NOT reimplemented here. `part_pose`, `load_handeye`,
`deproject` and `cam_to_robot` are imported from pose_seg, which stays the single
definition of how a mask becomes a robot coordinate. One implementation, one thing
to describe in the report, no risk of the live and offline paths drifting apart.

USE
    from scan import scan
    picks, vis = scan()                       # live camera, one frame
    picks, vis = scan("pde4445-dev/sept2_1.db3")   # from a recording

    python pde4445-dev/scan.py                     # live, prints the table
    python pde4445-dev/scan.py sept2_1.db3         # from a recording

Returns picks sorted TOPMOST-FIRST (descending base-frame Z), each a dict with
x, y, z, yaw, aspect, label, conf, u, v — identical in shape to pick_list.json.
"""
import os, sys, json

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# single source of truth for the geometry
from pose_seg import (part_pose, load_handeye, HANDEYE, MODEL, NAMES,
                      CONF, FRAME_SKIP, occlusion_pass)      # noqa: E402

# ---------------------------------------------------------------- live capture
# COLOUR IS NOT NEGOTIABLE. The model, the hand-eye calibration and all 35
# recordings assume 1280x720 colour. Lowering it to make a run start would
# silently invalidate every one of them.
COLOR_W, COLOR_H = 1280, 720
FPS              = 30
WARMUP_FRAMES    = 10        # let auto-exposure settle before trusting a frame

# DEPTH IS negotiable, because every frame is aligned to the colour stream before
# anything geometric happens - deprojection uses the COLOUR intrinsics either way.
# It has to be negotiable: on 8 Sep pipe.start() raised "Couldn't resolve requests"
# and killed the run outright.
#
# 848x480 stays FIRST. It is what every one of the 35 eval recordings used, so it
# is the only mode under which live and offline results are directly comparable.
# (The manifest appeared to say those clips used 1280x720 depth; that field was
# being read after align.process() resampled depth onto the colour frame, so it
# recorded the aligned size, not the sensor stream. capture_scene.py now reads the
# stream profile instead. The depth resolution was never the cause of the fault -
# USB 2.x enumeration remains the likeliest explanation.)
#
# The lower rungs exist so a fault DEGRADES instead of aborting. Whatever is
# granted is reported and belongs in the run log: depth resolution changes the
# noise on z, and z is the whole sequencing argument.
DEPTH_LADDER = [(848, 480, 30),      # matches all 35 recordings - keep first
                (640, 480, 30),
                (848, 480, 15),
                (640, 480, 15)]      # last resort: USB 2.x bandwidth
LAST_DEPTH_MODE = None               # set by frame_live(), for logging

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(MODEL)
    return _MODEL


# ------------------------------------------------------------ frame sources
def _start_live(rs):
    """Open the camera, trying each depth mode in turn. Colour is never varied."""
    global LAST_DEPTH_MODE
    errors = []
    for dw, dh, fps in DEPTH_LADDER:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, FPS)
        try:
            prof = pipe.start(cfg)
        except RuntimeError as e:
            errors.append(f"    {dw}x{dh}@{fps}: {e}")
            continue
        LAST_DEPTH_MODE = (dw, dh, fps)
        if (dw, dh, fps) != DEPTH_LADDER[0]:
            print(f"  *** depth fell back to {dw}x{dh}@{fps} (preferred "
                  f"{DEPTH_LADDER[0][0]}x{DEPTH_LADDER[0][1]} was refused).")
            print(f"  *** Usually a USB 2.x link. Note it in the run log: depth noise "
                  f"is not comparable across modes.")
        return pipe, prof
    raise RuntimeError(
        "camera refused every depth mode with 1280x720 colour:\n" + "\n".join(errors) +
        "\n  Reseat the cable into a direct USB 3 port (blue, on the laptop itself, "
        "not a hub) and close any other app holding the camera.\n"
        "  Do NOT drop the colour resolution to work around this.")


def frame_live():
    """Grab ONE aligned colour+depth frame from the camera. Returns (rgb, depth_m, intr)."""
    import pyrealsense2 as rs
    pipe, prof = _start_live(rs)
    try:
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(WARMUP_FRAMES):          # discard early, badly-exposed frames
            ok, fs = pipe.try_wait_for_frames(2000)
            if ok:
                frames = fs
        if frames is None:
            raise RuntimeError("no frames from the camera")
        frames = align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        df    = frames.get_depth_frame()
        depth = np.asanyarray(df.get_data()).astype(np.float32) * df.get_units()
        ci    = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return color, depth, (ci.fx, ci.fy, ci.ppx, ci.ppy)
    finally:
        pipe.stop()


def frame_db3(path, skip=FRAME_SKIP):
    """Read one aligned frame from a recording. Returns (rgb, depth_m, intr)."""
    import pyrealsense2 as rs
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(path, repeat_playback=False)
    prof = pipe.start(cfg)
    prof.get_device().as_playback().set_real_time(False)
    try:
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(max(skip, 1)):
            ok, fs = pipe.try_wait_for_frames(2000)
            if not ok:
                break
            frames = fs
        if frames is None:
            raise RuntimeError(f"no frames in {path}")
        frames = align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        df    = frames.get_depth_frame()
        depth = np.asanyarray(df.get_data()).astype(np.float32) * df.get_units()
        ci    = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return color, depth, (ci.fx, ci.fy, ci.ppx, ci.ppy)
    finally:
        pipe.stop()


# --------------------------------------------------------------- the scan
def picks_from_frame(color_rgb, depth_m, intr, R, t, conf=CONF):
    """One frame -> (picks topmost-first, annotated BGR image)."""
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)     # RealSense is RGB; OpenCV/YOLO want BGR
    H, W = depth_m.shape
    res = _model().predict(bgr, conf=conf, verbose=False)[0]
    vis, picks, masks = bgr.copy(), [], []

    if res.masks is not None:
        for poly, cls, cf in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            poly = np.array(poly, np.int32)
            if len(poly) < 3:
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            pose = part_pose(m, depth_m, intr, R, t)
            if pose is None:
                continue
            pose["label"] = NAMES.get(int(cls), str(int(cls)))
            pose["conf"]  = round(float(cf), 3)
            picks.append(pose)
            masks.append(m)                              # kept: occlusion needs the scene
            cv2.polylines(vis, [poly], True, (0, 255, 0), 2)

    # Whole-scene reasoning, once every part is placed.
    occlusion_pass(picks, masks, depth_m, intr, R, t)

    picks.sort(key=lambda p: -p["z"])                    # topmost-first
    for i, p in enumerate(picks, 1):
        cv2.circle(vis, (p["u"], p["v"]), 16, (0, 0, 255), 2)
        cv2.putText(vis, str(i), (p["u"] - 8, p["v"] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if p.get("ax1") and p.get("ax2"):
            cv2.line(vis, tuple(p["ax1"]), tuple(p["ax2"]), (255, 0, 255), 2)
        # cyan = where the two jaws come down. If one of these sits on a neighbour
        # rather than on the bench, that is the collision the descent has to clear.
        for k in ("land1", "land2"):
            if p.get(k):
                cv2.drawMarker(vis, tuple(p[k]), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        occ = p.get("occlusion")
        if occ is not None:
            cv2.putText(vis, f"occ {occ:.2f}", (p["u"] - 26, p["v"] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255) if occ > 0.35 else (0, 180, 0), 1)
        # Orange arrow: where this part WOULD be nudged to free it. Drawn for the
        # too-occluded parts only, and never executed - the arm does not nudge.
        # Showing the decision is the contribution; performing it is future work.
        n = p.get("nudge")
        if n and occ is not None and occ > 0.35:
            cv2.arrowedLine(vis, (p["u"], p["v"]),
                            (p["u"] + n["px"][0], p["v"] + n["px"][1]),
                            (0, 140, 255), 2, tipLength=0.25)
    return picks, vis


def scan(source=None, conf=CONF, save_as=None):
    """source=None -> live camera; otherwise a path to a .db3."""
    R, t = load_handeye(os.path.join(HERE, "handeye.json") if not os.path.isabs(HANDEYE) else HANDEYE)
    color, depth, intr = frame_live() if source is None else frame_db3(source)
    picks, vis = picks_from_frame(color, depth, intr, R, t, conf)
    if save_as:
        cv2.imwrite(save_as, vis)
    return picks, vis


def print_picks(picks):
    print(f"\n{len(picks)} parts  —  TOPMOST-FIRST\n")
    print(f"  {'#':>2}  {'label':10} {'conf':>5} {'X(mm)':>8} {'Y(mm)':>8} "
          f"{'Z(mm)':>7} {'Zmed':>7} {'Zland':>7} {'wid':>6} {'yaw':>7} {'aspect':>7}")
    for i, p in enumerate(picks, 1):
        flag = "  <- near-square, yaw ambiguous" if p.get("aspect", 9) < 1.25 else ""
        # z_land ABOVE the part top means a jaw would come down on a neighbour,
        # not on the bench. That is the case that broke a finger on 8 Sep.
        zl = p.get("z_land")
        if zl is not None and zl > p["z"] - 2:
            flag += "  <- JAW LANDS ON SOMETHING AT/ABOVE THE PART"
        print(f"  {i:>2}  {p['label']:10} {p['conf']:>5} {p['x']:>8} {p['y']:>8} "
              f"{p['z']:>7} {p.get('z_med','-'):>7} {'-' if zl is None else zl:>7} "
              f"{p.get('width_mm','-'):>6} {p['yaw']:>7} {p.get('aspect',''):>7}{flag}")


def main():
    # Confidence is an EXPERIMENTAL PARAMETER, not a constant. Live runs want it
    # high (don't chase phantoms); the offline battery can sweep it over the same
    # recordings to produce a precision/recall curve. Hence a flag, not a literal.
    argv = [a for a in sys.argv[1:]]
    conf = CONF
    if "--conf" in argv:
        i = argv.index("--conf")
        conf = float(argv[i + 1])
        del argv[i:i + 2]
    src = argv[0] if argv else None
    if src and not os.path.isabs(src) and not os.path.exists(src):
        src = os.path.join(HERE, src)
    out = os.path.join(HERE, "scan_out.png")
    print(f"confidence threshold: {conf}")
    picks, _ = scan(src, conf=conf, save_as=out)
    print_picks(picks)

    # write the pick list so pick_one.py / clear_scene.py can consume it directly
    plist = os.path.join(HERE, "pick_list.json")
    json.dump(picks, open(plist, "w"), indent=2)

    print(f"\nSaved overlay   -> {out}")
    print(f"Saved pick list -> {plist}")
    print("(magenta line = measured long axis; check it lies along the part's long side)")


if __name__ == "__main__":
    main()
