"""
Perception -> robot pick list  (offline, runs on a recorded .db3 — no robot needed).

Pipeline:
  seg model -> mask per part
  mask -> area centroid + long-axis (PCA) + top-surface depth
  deproject -> camera 3D (metres)
  handeye.json -> robot pick coordinate (mm) + yaw (deg, object long axis in base frame)
  sort by robot Z (base +Z is up) -> TOPMOST-FIRST order

Outputs a printed pick list, pick_list.json, and an annotated pose_seg_out.png.

Usage:  python pde4445-dev/pose_seg.py
"""
import os, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------- config -------------------------------
DB3        = os.path.join(HERE, "sept2_1.db3")   # a recorded scene with depth
MODEL      = os.path.join(HERE, "best.pt")           # your trained SEG weights (from Colab)
HANDEYE    = os.path.join(HERE, "handeye.json")
FRAME_SKIP = 30            # advance this many frames into the clip
CONF       = 0.35          # detection confidence
TOP_BAND_M = 0.008         # 8 mm: how deep below the nearest point still counts as "top surface"
NAMES      = {0: "arduino", 1: "esp", 2: "lcd", 3: "ultrasonic"}

# ---- finger-landing sampling -------------------------------------------------
# A parallel-jaw grasp does not descend onto the part: it descends onto the two
# strips of scene either side of the part's SHORT axis. Whatever is there sets the
# lowest safe fingertip height, and on 8 Sep it was an esp lying under the jaw -
# the descent was planned against the arduino's top surface, met the esp instead,
# and snapped a PLA finger. So the two landing patches are measured explicitly.
FINGER_OUT    = 1.35       # finger centreline sits this multiple of the half-width out
LAND_PATCH_PX = 7          # half-size of the depth patch sampled at each landing point
LAND_NEAR_PCTL = 10        # percentile of that patch treated as "the top of what is there".
                           # Not the minimum: a single speckle of depth noise would then
                           # veto every descent. Not the median either: that would average
                           # an obstruction away with the bench behind it.


def land_z(depth, intr, R, t, uu, vv, rad=LAND_PATCH_PX):
    """Base-frame Z (mm) of the highest thing inside a small patch. None if unmeasurable."""
    H, W_ = depth.shape
    u0, u1 = max(0, int(uu) - rad), min(W_, int(uu) + rad + 1)
    v0, v1 = max(0, int(vv) - rad), min(H,  int(vv) + rad + 1)
    if u1 <= u0 or v1 <= v0:
        return None
    d = depth[v0:v1, u0:u1]
    d = d[d > 0]
    if d.size < 10:                       # too few valid pixels to trust
        return None
    z_near = float(np.percentile(d, LAND_NEAR_PCTL))
    return float(cam_to_robot(deproject(intr, uu, vv, z_near), R, t)[2])

# --------------------- pure helpers (no hardware deps) ---------------------
def load_handeye(path):
    h = json.load(open(path))
    return np.array(h["R"]), np.array(h["t_mm"])

def cam_to_robot(p_cam_m, R, t):
    """camera point in METRES -> robot base MILLIMETRES."""
    return R @ (np.asarray(p_cam_m, float) * 1000.0) + t

def deproject(intr, u, v, z_m):
    """pixel (u,v) + depth z(m) -> camera XYZ (m).  intr=(fx,fy,ppx,ppy)."""
    fx, fy, ppx, ppy = intr
    return np.array([(u - ppx) / fx * z_m, (v - ppy) / fy * z_m, z_m])

def part_pose(mask, depth, intr, R, t):
    """One mask -> dict with robot pick coords, yaw, and pixel centroid. None if unusable."""
    valid = (mask > 0) & (depth > 0)
    if valid.sum() < 30:
        return None
    ys, xs = np.where(mask > 0)
    cu, cv = xs.mean(), ys.mean()                       # area centroid (pixels)

    # long axis via PCA on the mask pixels
    P = np.stack([xs - cu, ys - cv], 1).astype(float)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    axis = vt[0]
    proj = P @ axis
    L = (proj.max() - proj.min()) / 2.0

    # top-surface depth = median of the nearest band inside the mask
    dvals = depth[valid]
    z_top = float(np.median(dvals[dvals <= dvals.min() + TOP_BAND_M]))

    # centre + long-axis endpoints -> camera 3D -> robot frame
    c  = cam_to_robot(deproject(intr, cu, cv, z_top), R, t)
    e1 = cam_to_robot(deproject(intr, cu - axis[0]*L, cv - axis[1]*L, z_top), R, t)
    e2 = cam_to_robot(deproject(intr, cu + axis[0]*L, cv + axis[1]*L, z_top), R, t)
    yaw = float(np.degrees(np.arctan2(e2[1] - e1[1], e2[0] - e1[0])))

    # pixel endpoints of the long axis, so the overlay can SHOW what yaw was measured
    ax1 = (int(cu - axis[0] * L), int(cv - axis[1] * L))
    ax2 = (int(cu + axis[0] * L), int(cv + axis[1] * L))
    # elongation: how trustworthy is "long axis"? ~1.0 means square => yaw is ambiguous
    perp = np.array([-axis[1], axis[0]])
    proj_perp = P @ perp
    W = (proj_perp.max() - proj_perp.min()) / 2.0
    aspect = float(L / W) if W > 1e-6 else 999.0

    # ---- short-axis width in mm: what the jaws have to span -------------------
    # Feeds the width-adaptive aperture. Descending fully open sweeps the whole
    # jaw span through the scene; opening only as far as the part needs shrinks
    # the swept footprint, which is what the broken finger was paying for.
    w1 = cam_to_robot(deproject(intr, cu - perp[0]*W, cv - perp[1]*W, z_top), R, t)
    w2 = cam_to_robot(deproject(intr, cu + perp[0]*W, cv + perp[1]*W, z_top), R, t)
    width_mm = float(np.linalg.norm(w2 - w1))

    # ---- tilt-robust body height ---------------------------------------------
    # z_top is the NEAREST band, which on a tilted board is its raised corner - the
    # measured cause of the 2/3 failures at grasp_dz=35. The median over the whole
    # mask tracks the body instead. Both are emitted so the report can compare them
    # rather than assert which is better.
    z_med = float(cam_to_robot(deproject(intr, cu, cv, float(np.median(dvals))), R, t)[2])

    # ---- TILT: fit a plane to the part's own surface ---------------------------
    # z_top and z_med are single numbers and cannot express a slope, so a board
    # propped up by a header on its underside looks identical to a flat one. The
    # jaws then contact one edge only and the part pivots out on the lift.
    # Least-squares z = ax + by + c over the mask, in the BASE frame, so the
    # gradient is a physical slope rather than a pixel one.
    try:
        ss = max(1, int(valid.sum() // 800))          # subsample: 800 pts is plenty
        yy, xx = np.where(valid)
        yy, xx = yy[::ss], xx[::ss]
        dd = depth[yy, xx]
        P3 = np.array([cam_to_robot(deproject(intr, u_, v_, d_), R, t)
                       for u_, v_, d_ in zip(xx, yy, dd)])
        A = np.c_[P3[:, 0], P3[:, 1], np.ones(len(P3))]
        coef, *_ = np.linalg.lstsq(A, P3[:, 2], rcond=None)
        grad = float(np.hypot(coef[0], coef[1]))
        tilt_deg = float(np.degrees(np.arctan(grad)))
        # direction of steepest DESCENT, i.e. which way the low edge lies
        low_dir = (-coef[0] / grad, -coef[1] / grad) if grad > 1e-6 else (0.0, 0.0)
        # how far the surface drops across the part's own short axis
        drop_mm = float(grad * width_mm)
    except Exception:
        tilt_deg, low_dir, drop_mm = None, (0.0, 0.0), None

    # ---- what the fingers will actually land on -------------------------------
    off = W * FINGER_OUT
    l1  = (cu - perp[0]*off, cv - perp[1]*off)
    l2  = (cu + perp[0]*off, cv + perp[1]*off)
    zl  = [z for z in (land_z(depth, intr, R, t, *l1),
                       land_z(depth, intr, R, t, *l2)) if z is not None]

    return {"x": round(float(c[0]), 1), "y": round(float(c[1]), 1), "z": round(float(c[2]), 1),
            "yaw": round(yaw, 1), "u": int(cu), "v": int(cv),
            "ax1": ax1, "ax2": ax2, "aspect": round(aspect, 2),
            "z_med": round(z_med, 1),
            "width_mm": round(width_mm, 1),
            # highest of the two landing patches: the descent must clear the WORSE side
            "z_land": round(max(zl), 1) if zl else None,
            "land1": (int(l1[0]), int(l1[1])), "land2": (int(l2[0]), int(l2[1])),
            # surface slope, and how much the part drops across the grasp width
            "tilt_deg": None if tilt_deg is None else round(tilt_deg, 1),
            "tilt_drop_mm": None if drop_mm is None else round(drop_mm, 1),
            "low_dir": [round(low_dir[0], 3), round(low_dir[1], 3)]}

# ------------------------- occlusion and nudging -------------------------
# A single mask cannot say whether its object is buried - that needs the whole
# scene. So this runs as a pass over ALL detections once part_pose has placed
# each one, and answers three questions the sequencing policy actually needs:
#
#   crowding   how boxed-in is this part? (the 'arduino and lcd in a T' case)
#   occlusion  how much of it is shadowed by parts sitting HIGHER than it?
#   nudge      if it is unpickable, which way is there room to push it?
#
# Crowding and occlusion are deliberately separate. A part can be surrounded yet
# perfectly graspable because everything around it is lower; and a part can touch
# only one neighbour yet be pinned under it. Collapsing them into one number
# would hide exactly the distinction the pick decision turns on.
OCC_DILATE_PX = 9      # how close another mask must come to count as touching
OCC_HIGHER_MM = 4.0    # a neighbour must be this much higher to shadow this part
NUDGE_STEP_PX = 12     # ray-march step when looking for free space
NUDGE_MAX_PX  = 260    # stop looking this far out


def _free_run_px(occupied, u, v, dux, duy, shape):
    """How far a ray from (u,v) travels before it meets another mask or the edge."""
    H, W = shape
    for k in range(1, NUDGE_MAX_PX // NUDGE_STEP_PX + 1):
        uu = int(u + dux * k * NUDGE_STEP_PX)
        vv = int(v + duy * k * NUDGE_STEP_PX)
        if not (0 <= uu < W and 0 <= vv < H):
            return k * NUDGE_STEP_PX, True          # ran off the frame
        if occupied[vv, uu]:
            return k * NUDGE_STEP_PX, False
    return NUDGE_MAX_PX, False


def occlusion_pass(picks, masks, depth, intr, R, t):
    """Annotate every pick with crowding, occlusion and a nudge direction.

    Scored against the part's OUTLINE, not its area. An area-based fraction does
    not scale with how buried a part is: a board sharing one whole edge with a
    neighbour still only has about dilate_px x edge_length of its area within
    reach of that neighbour, which for a 100 x 100 px part is ~0.09 whether it is
    barely touching or half covered. Perimeter contact says what it should - one
    side of four against something is ~0.25, pinned on two sides is ~0.5, ringed
    is ~1.0 - so a threshold means something across part sizes.
    """
    if not picks:
        return picks
    H, W = depth.shape
    k = np.ones((OCC_DILATE_PX, OCC_DILATE_PX), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    grown = [cv2.dilate(m, k) > 0 for m in masks]
    binm  = [m > 0 for m in masks]
    # outline = mask minus its own erosion
    rims  = [(m > 0) & ~(cv2.erode(m, k3) > 0) for m in masks]
    rimn  = [max(int(r.sum()), 1) for r in rims]
    union = np.zeros((H, W), bool)
    for b in binm:
        union |= b

    def height(pk):
        # z_med tracks the body; z is the nearest band and jumps to a neighbour's
        # height wherever masks abut. Comparing heights is exactly where that
        # matters, so prefer the median.
        return pk.get("z_med", pk["z"])

    for i, p in enumerate(picks):
        crowd = occl = 0.0
        push = np.zeros(2)
        blockers = []
        for j, q in enumerate(picks):
            if i == j:
                continue
            frac = float((rims[i] & grown[j]).sum()) / rimn[i]
            if frac <= 0.0:
                continue
            crowd += frac
            # Push AWAY from every neighbour, weighted by how much of this part
            # that neighbour crowds. A part hemmed in on two opposite sides gets
            # near-cancelling vectors, which is the correct answer: there is no
            # good direction, and the magnitude says so.
            d = np.array([p["u"] - q["u"], p["v"] - q["v"]], float)
            n = np.linalg.norm(d)
            if n > 1e-6:
                push += frac * d / n
            if height(q) > height(p) + OCC_HIGHER_MM:
                occl += frac
                # Identity, not just class. To CLEAR a blocker the arm has to be
                # told which one to pick, and "an lcd" is not an instruction when
                # there are two of them on the bench.
                blockers.append({"label": q.get("label", "?"),
                                 "x": q["x"], "y": q["y"], "frac": round(frac, 3)})

        p["crowding"]  = round(min(crowd, 1.0), 3)
        p["occlusion"] = round(min(occl, 1.0), 3)
        blockers.sort(key=lambda b: -b["frac"])      # worst offender first
        p["under"]     = [b["label"] for b in blockers]
        p["under_xy"]  = [[b["x"], b["y"]] for b in blockers]

        mag = float(np.linalg.norm(push))
        if mag < 1e-3:
            p["nudge"] = None
            continue
        dux, duy = (push / mag).tolist()
        run_px, off_frame = _free_run_px(union, p["u"], p["v"], dux, duy, (H, W))
        # Convert the pixel direction into a base-frame direction at THIS part's
        # depth, so the report is in millimetres the arm could actually use.
        z_m = float(np.median(depth[binm[i] & (depth > 0)])) if (binm[i] & (depth > 0)).any() \
            else None
        if z_m:
            a = cam_to_robot(deproject(intr, p["u"], p["v"], z_m), R, t)
            b = cam_to_robot(deproject(intr, p["u"] + dux * run_px,
                                       p["v"] + duy * run_px, z_m), R, t)
            vec = b - a
            p["nudge"] = {"dx_mm": round(float(vec[0]), 1),
                          "dy_mm": round(float(vec[1]), 1),
                          "free_mm": round(float(np.linalg.norm(vec[:2])), 1),
                          "to_open_space": bool(off_frame),
                          "px": [int(dux * run_px), int(duy * run_px)]}
        else:
            p["nudge"] = None
    return picks


def pickability(p, max_occlusion=0.35):
    """Should this part be attempted at all? Returns (ok, reason).

    The supervisor's framing, and the right one for a 12-week project: rather
    than trying to grasp everything, score what is reachable, attempt that, and
    declare the rest out of scope WITH A REASON. An honest deferral is a result;
    a silent failure is not.
    """
    occ = p.get("occlusion")
    if occ is None:
        return True, ""
    if occ > max_occlusion:
        under = ", ".join(p.get("under") or []) or "a higher neighbour"
        return False, f"occluded {occ:.2f} by {under}"
    return True, ""


# --------------------------------- main ---------------------------------
def main():
    from ultralytics import YOLO
    import pyrealsense2 as rs

    R, t = load_handeye(HANDEYE)
    model = YOLO(MODEL)

    # grab one frame from the recording
    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_device_from_file(DB3, repeat_playback=False)
    prof = pipe.start(cfg); prof.get_device().as_playback().set_real_time(False)
    align = rs.align(rs.stream.color)
    frames = None
    for _ in range(FRAME_SKIP):
        ok, frames = pipe.try_wait_for_frames(1000)
        if not ok:
            break
    frames = align.process(frames)
    color = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()), cv2.COLOR_RGB2BGR)
    df = frames.get_depth_frame()
    depth = np.asanyarray(df.get_data()).astype(np.float32) * df.get_units()   # metres
    ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = (ci.fx, ci.fy, ci.ppx, ci.ppy)
    pipe.stop()

    H, W = depth.shape
    res = model.predict(color, conf=CONF, verbose=False)[0]
    vis = color.copy()
    picks = []

    if res.masks is not None:
        for poly, cls, cf in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            poly = np.array(poly, np.int32)
            if len(poly) < 3:
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            pose = part_pose(m, depth, intr, R, t)
            if pose is None:
                continue
            pose["label"] = NAMES.get(int(cls), str(int(cls)))
            pose["conf"] = round(float(cf), 3)
            picks.append(pose)
            cv2.polylines(vis, [poly], True, (0, 255, 0), 2)

    picks.sort(key=lambda p: -p["z"])                 # topmost-first: highest base Z first

    print(f"\n{len(picks)} parts detected  —  TOPMOST-FIRST pick order:\n")
    print(f"  {'#':>2}  {'label':10} {'conf':>5}   {'X(mm)':>8} {'Y(mm)':>8} {'Z(mm)':>7}  "
          f"{'yaw':>6} {'aspect':>7}")
    for i, p in enumerate(picks, 1):
        flag = "  <- near-square, yaw ambiguous" if p["aspect"] < 1.25 else ""
        print(f"  {i:>2}  {p['label']:10} {p['conf']:>5}   {p['x']:>8} {p['y']:>8} {p['z']:>7}  "
              f"{p['yaw']:>6} {p['aspect']:>7}{flag}")
        # draw the measured long axis so you can SEE whether yaw is sensible
        cv2.line(vis, tuple(p["ax1"]), tuple(p["ax2"]), (255, 0, 255), 2)
        cv2.circle(vis, (p["u"], p["v"]), 16, (0, 0, 255), 2)
        cv2.putText(vis, f"{i}:{p['yaw']:.0f}", (p["u"] - 14, p["v"] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(HERE, "pose_seg_out.png"), vis)
    json.dump(picks, open(os.path.join(HERE, "pick_list.json"), "w"), indent=2)
    print("\nSaved  ->  pose_seg_out.png   (numbered pick order)")
    print("Saved  ->  pick_list.json     (x, y, z, yaw per part)")
    print("\nNOTE: 'yaw' is the object's long-axis heading in the robot base frame.")
    print("      For a parallel-jaw grip across the short side, the gripper command is")
    print("      likely yaw ± 90 deg — confirm the exact offset with Aman's end-effector.")

if __name__ == "__main__":
    main()
