"""
DEBUG / CAPTURE MODE — record tagged scenes for offline evaluation.

THE POINT: the whole vision test battery needs recordings, not the robot.
Scattered / regular / bunched, the adversarial spoon, low light — every one is a
`.db3`, and pose_seg.py runs on `.db3` with no hardware. So capture in bulk on a
lab day and evaluate all week at the desk.

This tool exists so those ~40 recordings arrive ORGANISED and TAGGED rather than
as a pile of unnamed files, and so each one carries a manifest row saying what
condition it represents.

MODES
    probe    report the stream setup of an existing .db3, so new recordings MATCH
    record   record N tagged scenes into captures/<condition>/
    debug    regenerate debug images from an existing .db3 (no camera needed)

EXAMPLES
    python pde4445-dev/capture_scene.py probe  --db3 pde4445-dev/sept2_1.db3
    python pde4445-dev/capture_scene.py record --condition scattered --n 10
    python pde4445-dev/capture_scene.py record --condition adversarial --n 5 \
           --note "kitchen spoon, pliers, unseen objects"
    python pde4445-dev/capture_scene.py record --condition lowlight --n 10 --note "room lights off"
    python pde4445-dev/capture_scene.py debug  --db3 captures/scattered/xxx.db3

OUTPUT
    captures/<condition>/<stamp>_<condition>_<i>.db3      the recording
    captures/<condition>/<stamp>_<condition>_<i>_color.png
    captures/<condition>/<stamp>_<condition>_<i>_depth.png   depth flattened to grayscale
    captures/manifest.csv                                   one row per recording

TWO THINGS THAT MATTER

1. COLOUR FORMAT. Recordings are made as rgb8, matching what the RealSense Viewer
   produced for the existing clips, because pose_seg.py does an RGB->BGR convert on
   read. Recording bgr8 here would silently swap the channels and wreck detection.

2. FIXED DEPTH RANGE for the grayscale images. Auto-normalising each frame to its
   own min/max makes scenes LOOK different only because their content differs, so
   they cannot be compared side by side in the report. A fixed range keeps the
   grayscale physically meaningful: the same shade means the same distance in
   every figure.
"""
import os, csv, sys, time, argparse
import gc
from datetime import datetime

import numpy as np
import cv2

HERE         = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(HERE, "captures")
MANIFEST     = os.path.join(CAPTURES_DIR, "manifest.csv")

# Fixed grayscale mapping so every depth figure is comparable (see note 2 above).
# Workspace sits ~0.33 m from the camera at the capture pose.
DEPTH_MIN_M, DEPTH_MAX_M = 0.25, 0.55

CONDITIONS = ["scattered", "regular", "bunched", "adversarial", "lowlight",
              "pnp_easy", "pnp_hard",
              # 'static' is the ONLY condition where the bench must not be touched
              # between clips. It measures pose repeatability - the perception
              # noise floor - which needs the same object observed twice. The other
              # conditions were re-arranged every clip, which makes them a better
              # detection benchmark and useless for precision.
              "static"]

MANIFEST_FIELDS = ["timestamp", "condition", "index", "db3", "color_png", "depth_png",
                   "color_res", "depth_res", "seconds", "note"]


# ------------------------------------------------------------------ helpers
def depth_to_gray(depth_m, lo=DEPTH_MIN_M, hi=DEPTH_MAX_M):
    """Depth in METRES -> 8-bit grayscale over a FIXED range. Invalid depth = black."""
    d = np.asarray(depth_m, dtype=np.float32)
    g = (d - lo) / max(hi - lo, 1e-6)
    g = np.clip(g, 0.0, 1.0)
    g = ((1.0 - g) * 255).astype(np.uint8)     # near = bright, far = dark
    g[d <= 0] = 0                              # no reading at all
    return g


def append_manifest(row):
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def save_debug_images(color_rgb, depth_m, base):
    """color_rgb is RealSense-native RGB; convert once for OpenCV."""
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(base + "_color.png", bgr)
    cv2.imwrite(base + "_depth.png", depth_to_gray(depth_m))
    return base + "_color.png", base + "_depth.png"


# -------------------------------------------------------------------- probe
def cmd_probe(args):
    import pyrealsense2 as rs
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(args.db3, repeat_playback=False)
    prof = pipe.start(cfg)
    prof.get_device().as_playback().set_real_time(False)
    try:
        c = prof.get_stream(rs.stream.color).as_video_stream_profile()
        d = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        print(f"\n{os.path.basename(args.db3)}")
        print(f"  colour : {c.width()}x{c.height()} @ {c.fps()}fps  format {c.format()}")
        print(f"  depth  : {d.width()}x{d.height()} @ {d.fps()}fps  format {d.format()}")
        i = c.get_intrinsics()
        print(f"  colour intrinsics: fx={i.fx:.1f} fy={i.fy:.1f} ppx={i.ppx:.1f} ppy={i.ppy:.1f}")
        print("\nRecord new clips at THESE settings so detection behaves identically:")
        print(f"  --width {c.width()} --height {c.height()} --fps {c.fps()}")
    finally:
        pipe.stop()


# ------------------------------------------------------------------- record
# How long to wait after tearing a pipeline down before opening the next one.
# Empirically the recorder needs a moment to let go of the device on Windows.
RELEASE_S = 1.0


def _record_clip(rs, db3, args, attempts=3):
    """Record ONE clip and tear the pipeline completely down afterwards.

    WHY THE EXPLICIT TEARDOWN, AND WHY IT IS NOT OPTIONAL.
    `pipe.stop()` ends streaming, but the pipeline, config and profile objects
    keep the RECORDER DEVICE alive until Python collects them. The next
    `pipe.start()` then succeeds while the previous recorder still owns the
    camera, and no frames ever arrive. Observed 12 Sep: clip 1 recorded, clips
    2-10 all reported "no frames captured" on a healthy USB 3.2 link. Dropping
    the references and forcing a collection between clips is the fix.

    Also note the .copy() on the colour frame. numpy wraps the frame's own
    buffer rather than owning it, so reading it after the pipeline is destroyed
    is a use-after-free - it usually returns plausible-looking garbage rather
    than crashing, which is worse.
    """
    last = ""
    for attempt in range(1, attempts + 1):
        pipe, cfg, prof = rs.pipeline(), rs.config(), None
        cfg.enable_stream(rs.stream.depth, args.dwidth, args.dheight, rs.format.z16, args.fps)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
        cfg.enable_record_to_file(db3)
        try:
            prof = pipe.start(cfg)
            align = rs.align(rs.stream.color)
            t0, frames = time.time(), None
            while time.time() - t0 < args.seconds:
                ok, fs = pipe.try_wait_for_frames(2000)
                if ok:
                    frames = fs
            if frames is None:
                last = "no frames arrived"
                continue
            dp = prof.get_stream(rs.stream.depth).as_video_stream_profile()
            dres = f"{dp.width()}x{dp.height()}"
            frames = align.process(frames)
            color = np.asanyarray(frames.get_color_frame().get_data()).copy()
            df = frames.get_depth_frame()
            depth = np.asanyarray(df.get_data()).astype(np.float32) * df.get_units()
            return color, depth, f"{color.shape[1]}x{color.shape[0]}", dres
        except RuntimeError as e:
            last = str(e)
        finally:
            try:
                pipe.stop()
            except Exception:
                pass
            del prof, cfg, pipe
            gc.collect()
            time.sleep(RELEASE_S)

        # A start that produced nothing still created a stub recording.
        if os.path.exists(db3) and os.path.getsize(db3) < 4096:
            try:
                os.remove(db3)
            except OSError:
                pass
        if attempt < attempts:
            print(f"   attempt {attempt} failed ({last}) - retrying")
    print(f"   !! {attempts} attempts failed: {last}")
    return None


def cmd_record(args):
    import pyrealsense2 as rs

    outdir = os.path.join(CAPTURES_DIR, args.condition)
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nCondition '{args.condition}' — {args.n} scenes, {args.seconds}s each")
    print(f"Saving to {outdir}")
    print(f"Colour {args.width}x{args.height}@{args.fps} (rgb8), depth "
          f"{args.dwidth}x{args.dheight}@{args.fps}\n")
    print("The ARM DOES NOT MOVE in this mode. Keep it parked at the capture pose\n"
          "so the recordings match the hand-eye calibration.\n")

    for i in range(1, args.n + 1):
        name = f"{stamp}_{args.condition}_{i:02d}"
        db3  = os.path.join(outdir, name + ".db3")

        input(f"[{i}/{args.n}] Arrange the scene, then press Enter to record {args.seconds}s...")

        got = _record_clip(rs, db3, args)
        if got is None:
            print("   !! giving up on this clip")
            continue
        color, depth, cres, dres = got

        cpng, dpng = save_debug_images(color, depth, os.path.join(outdir, name))
        append_manifest({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "condition": args.condition, "index": i, "db3": db3,
            "color_png": cpng, "depth_png": dpng,
            "color_res": cres, "depth_res": dres,
            "seconds": args.seconds, "note": args.note,
        })
        print(f"   saved {name}.db3  (+ colour and depth PNG)")

    print(f"\nDone. {args.n} scenes under '{args.condition}'. Manifest: {MANIFEST}")
    print("These can now be evaluated offline with pose_seg.py - no lab needed.")


# -------------------------------------------------------------------- debug
def cmd_debug(args):
    """Regenerate debug images from an existing recording. No camera required."""
    import pyrealsense2 as rs
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(args.db3, repeat_playback=False)
    prof = pipe.start(cfg)
    prof.get_device().as_playback().set_real_time(False)
    try:
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(max(args.frame, 1)):
            ok, fs = pipe.try_wait_for_frames(2000)
            if not ok:
                break
            frames = fs
        if frames is None:
            raise SystemExit("could not read a frame from that recording")
        frames = align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        df    = frames.get_depth_frame()
        depth = np.asanyarray(df.get_data()).astype(np.float32) * df.get_units()
    finally:
        pipe.stop()

    base = os.path.splitext(args.db3)[0] + f"_f{args.frame}"
    cpng, dpng = save_debug_images(color, depth, base)
    mid = depth[depth.shape[0] // 2, depth.shape[1] // 2]
    valid = float((depth > 0).mean() * 100)
    print(f"  {cpng}\n  {dpng}")
    print(f"  centre depth {mid:.3f} m · valid depth pixels {valid:.1f}%")
    print(f"  grayscale range fixed at {DEPTH_MIN_M}-{DEPTH_MAX_M} m "
          f"(near = bright), so figures are comparable across scenes")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Tagged scene capture + debug imagery.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="report an existing .db3's stream setup")
    p.add_argument("--db3", required=True)
    p.set_defaults(func=cmd_probe)

    r = sub.add_parser("record", help="record N tagged scenes")
    r.add_argument("--condition", required=True, choices=CONDITIONS)
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--seconds", type=float, default=2.0)
    r.add_argument("--note", default="")
    r.add_argument("--width", type=int, default=1280)
    r.add_argument("--height", type=int, default=720)
    r.add_argument("--dwidth", type=int, default=848)
    r.add_argument("--dheight", type=int, default=480)
    r.add_argument("--fps", type=int, default=30)
    r.set_defaults(func=cmd_record)

    d = sub.add_parser("debug", help="regenerate debug images from a recording")
    d.add_argument("--db3", required=True)
    d.add_argument("--frame", type=int, default=30)
    d.set_defaults(func=cmd_debug)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
