"""
PERCEPTION NOISE FLOOR — how repeatable is a pose when nothing moves?

Camera only. No arm, no gripper. About three minutes.

WHY THIS EXISTS SEPARATELY FROM capture_scene.py
capture_scene records a .db3 per scene, which means opening and closing the
camera once per clip. On 12 Sep that broke down after four clips with Windows
Media Foundation errors - 0x800701b1 "a device which does not exist", then
MFCreateDeviceSource "cannot find the path" - i.e. the OS had lost the device.
Ten open/close cycles in a minute is more than the UVC stack will take.

Nothing about a repeatability measurement needs recordings. It needs N looks at
one unchanging scene. So this opens the camera ONCE, holds it open, and grabs N
frames through the same perception pipeline the robot uses.

WHAT IT MEASURES
The parts do not move, so any variation in a part's reported pose is the
perception system's own noise. That is the precision figure every pick-accuracy
number has to be read against: a 3 mm placement error means nothing if the
sensing is +/- 5 mm, and means a great deal if the sensing is +/- 0.5 mm.

    python pde4445-dev/noise_floor.py                # 15 frames, 0.6s apart
    python pde4445-dev/noise_floor.py --n 25
    python pde4445-dev/noise_floor.py --self-test    # no camera needed

SET THE SCENE AND DO NOT TOUCH IT - not the parts, not the bench, not the
camera, not the arm. One nudge and the measurement becomes meaningless.

Writes noise_floor.json and noise_floor_overlay.png.
"""
import os, sys, json, time, argparse, statistics as stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT_JSON = os.path.join(HERE, "noise_floor.json")
OUT_PNG  = os.path.join(HERE, "noise_floor_overlay.png")
MATCH_MM = 25.0          # a detection this close to a track's mean is the same part
SETTLE_S = 0.6           # between grabs; long enough that frames are independent
MOVED_FLOOR_MM = 3.0     # below this nothing has physically moved, however small
                         # the measurement noise happens to be


def yaw_spread(vals):
    """Spread of yaw readings, modulo 180 degrees.

    PCA returns an UNDIRECTED long axis, so the same board reads +85 in one frame
    and -95 in the next. Those are the same orientation. A plain standard
    deviation counts every sign flip as a ~180 degree error.
    """
    a0 = vals[0]
    return stats.pstdev([((v - a0 + 90) % 180) - 90 for v in vals])


def track(frames, match_mm=MATCH_MM):
    """Group detections across frames into per-part tracks.

    Matching is class-aware, one-to-one and cheapest-pair-first, anchored on each
    track's running mean. Without those three rules two same-class parts sitting
    close together trade detections between frames, and the "noise" that comes
    out is the distance between two different objects.
    """
    if not frames:
        return []
    tracks = [{"label": p["label"], "pts": [p]} for p in frames[0]]
    for f in frames[1:]:
        cand = []
        for ti, tr in enumerate(tracks):
            cx = stats.mean(q["x"] for q in tr["pts"])
            cy = stats.mean(q["y"] for q in tr["pts"])
            for di, p in enumerate(f):
                if p["label"] != tr["label"]:
                    continue
                d2 = (cx - p["x"]) ** 2 + (cy - p["y"]) ** 2
                if d2 <= match_mm ** 2:
                    cand.append((d2, ti, di))
        cand.sort()
        ut, ud = set(), set()
        for _d2, ti, di in cand:
            if ti in ut or di in ud:
                continue
            tracks[ti]["pts"].append(f[di])
            ut.add(ti); ud.add(di)
        # Anything unmatched starts its own track. Seeding only from frame 0 would
        # make a part that happens to be missed in the FIRST frame invisible for
        # the whole run - reported as nothing at all rather than as intermittent,
        # which is the opposite of what a repeatability check should do.
        for di, p in enumerate(f):
            if di not in ud:
                tracks.append({"label": p["label"], "pts": [p]})
    return tracks


def summarise(tracks, n_frames):
    """Per-part spread, and the pooled figure to quote."""
    need = max(3, int(0.8 * n_frames))
    rows = []
    for tr in tracks:
        pts = tr["pts"]
        if len(pts) < need:
            rows.append({"label": tr["label"], "seen": len(pts), "of": n_frames,
                         "dropped": True})
            continue
        jump = max((((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** .5)
                   for a, b in zip(pts, pts[1:])) if len(pts) > 1 else 0.0
        rows.append({
            "label": tr["label"], "seen": len(pts), "of": n_frames, "dropped": False,
            "sd_x_mm": round(stats.pstdev([p["x"] for p in pts]), 2),
            "sd_y_mm": round(stats.pstdev([p["y"] for p in pts]), 2),
            "sd_z_mm": round(stats.pstdev([p["z"] for p in pts]), 2),
            "sd_yaw_deg": round(yaw_spread([p["yaw"] for p in pts]), 2),
            "sd_width_mm": round(stats.pstdev([p["width_mm"] for p in pts]), 2)
            if all("width_mm" in p for p in pts) else None,
            "max_step_mm": round(jump, 2),
        })
    kept = [r for r in rows if not r["dropped"]]
    pooled = {}
    if kept:
        for k in ("sd_x_mm", "sd_y_mm", "sd_z_mm", "sd_yaw_deg"):
            pooled[k] = round(stats.mean(r[k] for r in kept), 2)
        pooled["worst_step_mm"] = round(max(r["max_step_mm"] for r in kept), 2)
    return rows, pooled, kept


def report(rows, pooled, kept, n_frames):
    print(f"\n---- pose repeatability over {n_frames} frames ----")
    print(f"  {'part':11} {'seen':>6} {'sd x':>7} {'sd y':>7} {'sd z':>7} "
          f"{'sd yaw':>8} {'sd width':>9} {'max step':>9}")
    for r in rows:
        if r["dropped"]:
            print(f"  {r['label']:11} {r['seen']:>3}/{r['of']:<3}  "
                  f"DROPPED - not detected in enough frames")
            continue
        w = "-" if r["sd_width_mm"] is None else f"{r['sd_width_mm']:.2f}"
        print(f"  {r['label']:11} {r['seen']:>3}/{r['of']:<3} {r['sd_x_mm']:>7.2f} "
              f"{r['sd_y_mm']:>7.2f} {r['sd_z_mm']:>7.2f} {r['sd_yaw_deg']:>8.2f} "
              f"{w:>9} {r['max_step_mm']:>9.2f}")
    if not kept:
        print("\n  No part survived enough frames — detection is the problem here,")
        print("  not precision. Check the overlay before reading anything into this.")
        return
    print(f"\n  POOLED: sd_x {pooled['sd_x_mm']} mm   sd_y {pooled['sd_y_mm']} mm   "
          f"sd_z {pooled['sd_z_mm']} mm   sd_yaw {pooled['sd_yaw_deg']} deg")
    print(f"  Largest frame-to-frame step: {pooled['worst_step_mm']} mm")
    # A step only means something if it is BOTH large relative to the noise AND
    # large in absolute terms. With sd_x at 0.07 mm, five times the spread is
    # 0.35 mm - so a 0.6 mm step tripped this and cried "something moved" about a
    # bench nobody had touched. Nothing physical shifts by half a millimetre.
    planar = max(pooled["sd_x_mm"], pooled["sd_y_mm"], 0.05)
    if pooled["worst_step_mm"] > max(5 * planar, MOVED_FLOOR_MM):
        print(f"  *** A {pooled['worst_step_mm']} mm step is far larger than the "
              f"{planar:.2f} mm spread — something MOVED mid-run, or two parts")
        print("  *** swapped tracks. Re-run without touching anything.")
    print(f"\n  Read every pick-accuracy figure against this. Hand-eye RMS is "
          f"2.67 mm;\n  if sd here is well under that, placement error is dominated by "
          f"calibration\n  and the arm, not by perception.")


def main():
    ap = argparse.ArgumentParser(description="Perception noise floor, camera only.")
    ap.add_argument("--n", type=int, default=15, help="frames to grab (default 15)")
    ap.add_argument("--settle", type=float, default=SETTLE_S)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--match-mm", type=float, default=MATCH_MM)
    args = ap.parse_args()

    import numpy as np
    import cv2
    import pyrealsense2 as rs
    import scan
    from pose_seg import load_handeye, HANDEYE

    R, t = load_handeye(HANDEYE if os.path.isabs(HANDEYE)
                        else os.path.join(HERE, "handeye.json"))

    print(f"\nGrabbing {args.n} frames, {args.settle}s apart, at conf {args.conf}.")
    print("DO NOT TOUCH the parts, the bench, the camera or the arm.\n")
    input("Scene set? Press Enter to start...")

    # ONE pipeline for the whole run. The camera is opened once and held open —
    # the whole point of this script.
    pipe, prof = scan._start_live(rs)
    frames_out, vis_last = [], None
    try:
        align = rs.align(rs.stream.color)
        ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intr = (ci.fx, ci.fy, ci.ppx, ci.ppy)
        for _ in range(10):                       # let auto-exposure settle
            pipe.try_wait_for_frames(2000)

        # KEEP DRAINING THE STREAM. The first version slept for `settle` and then
        # ran ~200 ms of inference, so the pipeline went ~0.8 s between polls.
        # librealsense hands frames out of a fixed pool and a consumer that stops
        # collecting can stall the stream. Polling continuously and simply
        # discarding frames we do not want costs nothing and keeps the queue moving.
        #
        # The explicit `del` matters for the same reason: a frameset held in a
        # Python local is a frame not returned to the pool.
        i, next_t, misses = 0, time.time(), 0
        while i < args.n:
            ok, fs = pipe.try_wait_for_frames(4000)
            if not ok:
                misses += 1
                if misses in (5, 25) or misses % 100 == 0:
                    print(f"  ...{misses} empty polls — the camera has stopped "
                          f"delivering frames")
                if misses > 300:
                    print("  !! giving up: the device is no longer streaming.")
                    break
                continue
            misses = 0
            if time.time() < next_t:
                del fs                          # drained, not wanted
                continue
            aligned = align.process(fs)
            color = np.asanyarray(aligned.get_color_frame().get_data()).copy()
            df = aligned.get_depth_frame()
            depth = np.asanyarray(df.get_data()).astype(np.float32) * df.get_units()
            del df, aligned, fs                 # back to the pool before inference
            picks, vis = scan.picks_from_frame(color, depth, intr, R, t, conf=args.conf)
            frames_out.append(picks)
            vis_last = vis
            i += 1
            print(f"  [{i}/{args.n}] {len(picks)} parts")
            next_t = time.time() + args.settle
    finally:
        pipe.stop()

    if not frames_out:
        raise SystemExit("no frames were processed")
    if vis_last is not None:
        cv2.imwrite(OUT_PNG, vis_last)

    tracks = track(frames_out, args.match_mm)
    rows, pooled, kept = summarise(tracks, len(frames_out))
    report(rows, pooled, kept, len(frames_out))

    json.dump({"n_frames": len(frames_out), "conf": args.conf,
               "per_part": rows, "pooled": pooled,
               "when": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(OUT_JSON, "w"), indent=2)
    print(f"\n  wrote {OUT_JSON} and {OUT_PNG}")


# ----------------------------------------------------------------- self-test
def self_test():
    import random
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    ck("a PCA sign flip is not treated as rotation",
       yaw_spread([85.0, -95.0, 85.0, -95.0]) < 0.01)
    ck("real rotation still registers", yaw_spread([0.0, 20.0, 40.0]) > 10)

    rnd = random.Random(0)
    truth = [("arduino", 0.0, 600.0), ("arduino", 11.0, 605.0),   # 12mm apart!
             ("esp", 90.0, 590.0), ("lcd", 150.0, 610.0)]
    frames = []
    for k in range(12):
        f = [{"label": l, "x": x + rnd.gauss(0, .4), "y": y + rnd.gauss(0, .4),
              "z": 250 + rnd.gauss(0, .3), "yaw": 30.0 + 180.0 * (k % 2),
              "width_mm": 53.0}
             for l, x, y in truth]
        rnd.shuffle(f)                       # detection order is not stable
        frames.append(f)
    tracks = track(frames)
    rows, pooled, kept = summarise(tracks, len(frames))
    ck("every part is tracked", len(kept) == 4, str([r["label"] for r in rows]))
    ck("two same-class parts 12mm apart do not swap tracks",
       all(r["sd_x_mm"] < 1.5 for r in kept), str([r["sd_x_mm"] for r in kept]))
    ck("shuffled detection order does not matter",
       all(r["seen"] == 12 for r in kept), str([r["seen"] for r in kept]))
    ck("yaw flips do not inflate the spread",
       all(r["sd_yaw_deg"] < 1.0 for r in kept), str([r["sd_yaw_deg"] for r in kept]))
    ck("no false movement reported", pooled["worst_step_mm"] < 4, str(pooled))

    # a part that vanishes half the time must be dropped, not averaged in
    thin = [f if k % 2 else [p for p in f if p["label"] != "lcd"]
            for k, f in enumerate(frames)]
    rows2, _, kept2 = summarise(track(thin), len(thin))
    ck("an intermittently-detected part is dropped, not averaged",
       any(r["dropped"] and r["label"] == "lcd" for r in rows2),
       str([(r["label"], r["seen"]) for r in rows2]))

    # a genuine nudge mid-run must show up as a step
    moved = [[dict(p, x=p["x"] + (40 if k >= 6 else 0)) for p in f]
             for k, f in enumerate(frames)]
    _, pooled3, _ = summarise(track(moved, match_mm=60), len(moved))
    ck("a real 40mm nudge shows up as a large step",
       pooled3["worst_step_mm"] > 30, str(pooled3))

    print("\n" + "=" * 56)
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("noise_floor self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    main()
