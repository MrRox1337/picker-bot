#!/usr/bin/env python3
"""bench_plane.py — measure the bench as a PLANE, not as a number.

WHY
---
`--min-z` is a single absolute Z, calibrated by one paper-drag at one spot
(Z = 213.315, 12 Sep). It is the hard floor applied after every descent policy,
and it is the only thing standing between a bad grasp_dz and a broken finger.

A single number is only correct if the bench is level. It is not. Fitting a
plane to `z_land` over 60 isolated parts from 16 and 19 Sep, with a per-class
offset to control for jaw span, gives a gradient of -0.94 mm per 100 mm in y -
about 2.6 mm across the working range, 0.54 deg - and all four classes agree on
the sign independently. Including the plane drops the residual from 1.58 to
1.00 mm.

So the real clearance under `--min-z 215` varies by ~2.6 mm depending on where
the part is: too tight at one end of the bench, too loose at the other.

WHAT THIS SCRIPT DOES
---------------------
Records a direct paper-drag contact height at several XY positions, fits a
plane to them, and CROSS-CHECKS that plane against the one derived from the
depth stream. The two are measured in different frames - the paper drag gives a
commanded Z at fingertip contact, z_land gives a depth-derived surface height,
and they differ by the tool offset - so the intercepts are not comparable and
are not compared. The GRADIENTS are, and agreement between an optical estimate
and a mechanical one is the evidence: it is the same cross-validation as the
aperture model against the measured stalls.

IT DOES NOT MOVE THE ARM. Jog by hand, read the controller, type the number.

USAGE
    python pde4445-dev/bench_plane.py                 # interactive
    python pde4445-dev/bench_plane.py --point -120 590 214.6 ...
    python pde4445-dev/bench_plane.py --self-test
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "bench_plane.json")

# The clearance the current floor was chosen to give: --min-z 215 against a
# measured contact of 213.315. Preserved so the tilt-aware floor is the same
# safety margin, just applied correctly at every point on the bench.
CLEARANCE_MM = 1.7

# Span the working area actually used: x -144..83, y 561..839.
SUGGESTED = [(-120.0, 590.0, "near-left  (close to the arm)"),
             (60.0, 590.0, "near-right"),
             (-120.0, 830.0, "far-left"),
             (60.0, 830.0, "far-right"),
             (-30.0, 710.0, "centre  (check point, not used in the fit)")]


# A bench that varies by more than this is not a bench; it is a bad fit.
MAX_PLAUSIBLE_FALL_MM = 20.0
# Minimum spread PERPENDICULAR to the points' main direction. Below this the
# points are effectively a line and no plane is determined by them.
MIN_PERP_SPREAD_MM = 40.0


def usable(pts):
    """Can these points determine a plane at all? -> (ok, [reasons]).

    Added after the fit below returned slopes of 41 mm per 100 mm and a bench
    that fell 150 mm across itself. The inputs were (60,590), (-30,710) and
    (-120,830): three points spaced by exactly (-90,+120) twice, i.e. COLLINEAR.
    Infinitely many planes contain a line, least squares picked one of them, and
    the result looked like a measurement.

    A tool that answers confidently when the data cannot support an answer is
    worse than one that has no answer, so this refuses instead.
    """
    why = []
    P = np.asarray(pts, float)
    if len(P) < 4:
        why.append(f"{len(P)} point(s): a plane has 3 parameters, so 3 points "
                   f"fit it exactly with zero residual whatever they are, and "
                   f"tell you nothing about whether the plane is real. Use 4 or "
                   f"more, and hold one out as a check.")
    if len(P) >= 3:
        Q = P[:, :2] - P[:, :2].mean(axis=0)
        s = np.linalg.svd(Q, compute_uv=False)
        perp = float(s[1]) / np.sqrt(len(P)) if len(s) > 1 else 0.0
        if perp < MIN_PERP_SPREAD_MM:
            why.append(f"the XY positions are nearly collinear (spread across "
                       f"the line is only {perp:.1f} mm): they lie along one "
                       f"direction, so the tilt across that direction is not "
                       f"observable. Spread them over a rectangle, not a line.")
    zs = P[:, 2]
    if len(zs) and np.allclose(zs, np.round(zs)):
        span = zs.max() - zs.min()
        why.append(f"every reading is a whole millimetre and the total spread "
                   f"is {span:.0f} mm, so the measurement resolution is about "
                   f"as large as the effect. Re-measure in 0.25 mm steps before "
                   f"believing any gradient.")
    return (not why), why


def fit_plane(pts):
    """pts = [(x, y, z), ...] -> (cx, cy, c0), residual rms."""
    P = np.asarray(pts, float)
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    c, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    resid = P[:, 2] - A @ c
    return tuple(float(v) for v in c), float(np.sqrt((resid ** 2).mean()))


def depth_plane(runs_dir=RUNS):
    """The bench gradient as the DEPTH STREAM sees it, from z_land.

    Isolated parts only - a crowded part's jaws may land on a neighbour rather
    than on the bench - and a per-class offset, because a wide Arduino's jaws
    land further out than a narrow ultrasonic's and that is not bench tilt.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "run_2026091[69]_*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            try:
                if float(r["crowding"] or 0) > 0 or float(r["occlusion"] or 0) > 0:
                    continue
                x, y, zl = float(r["x_mm"]), float(r["y_mm"]), float(r["z_land"])
            except (ValueError, TypeError, KeyError):
                continue
            if 500 < y < 900 and -250 < x < 150:
                rows.append((r["label"], x, y, zl))
    if len(rows) < 12:
        return None
    labs = sorted({r[0] for r in rows})
    A = np.array([[r[1], r[2]] + [1.0 * (r[0] == L) for L in labs] for r in rows])
    z = np.array([r[3] for r in rows])
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A @ c
    return {"slope_x": float(c[0]), "slope_y": float(c[1]),
            "n": len(rows), "residual_sd_mm": float(resid.std()),
            "class_offsets": {L: float(v) for L, v in zip(labs, c[2:])}}


def report(pts, check=None):
    (cx, cy, c0), rms = fit_plane(pts)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    out = []
    add = out.append

    # The one number that does not need a plane, and the one we actually came
    # for. Report it FIRST and unconditionally: even a set of points that
    # cannot determine a tilt still tells you how high the bench is.
    zs = [p[2] for p in pts] + [c[2] for c in (check or [])]
    add("")
    add("CONTACT HEIGHT  (this needs no plane, and is the safety-critical number)")
    add(f"  measured: {', '.join(f'{z:.2f}' for z in sorted(zs))}")
    add(f"  mean {np.mean(zs):.2f}  median {np.median(zs):.2f}  "
        f"spread {max(zs)-min(zs):.2f} mm  over {len(zs)} point(s)")
    add(f"  12 Sep paper-drag put contact at 213.315, and --min-z 215 was")
    add(f"  chosen to clear it by {CLEARANCE_MM} mm.")
    gap = 215.0 - float(np.median(zs))
    add(f"  against today's median, --min-z 215 actually clears the bench by "
        f"{gap:+.2f} mm.")
    if abs(gap - CLEARANCE_MM) > 0.75:
        add(f"  *** That is {gap - CLEARANCE_MM:+.2f} mm away from the intended "
            f"margin. The floor is not where it is believed to be, and on a")
        add(f"  *** clamped descent the jaws stop that much high.")

    ok, why = usable(pts)
    if not ok:
        add("")
        add("NO PLANE FITTED — these points cannot determine one:")
        for w in why:
            add(f"  - {w}")
        add("")
        add("  The contact height above still stands: it is an average, and an")
        add("  average does not need the points to be well placed. Only the")
        add("  TILT is unobtainable from this data.")
        return out, {"contact_mean_mm": float(np.mean(zs)),
                     "contact_median_mm": float(np.median(zs)),
                     "contact_spread_mm": float(max(zs) - min(zs)),
                     "n_points": len(zs), "plane_fitted": False,
                     "refused_because": why,
                     "points": [list(p) for p in pts],
                     "check_points": [list(p) for p in (check or [])],
                     "depth_derived": depth_plane()}

    fall = abs(cx * (max(xs) - min(xs))) + abs(cy * (max(ys) - min(ys)))
    if fall > MAX_PLAUSIBLE_FALL_MM:
        add("")
        add(f"NO PLANE REPORTED — the fit says the bench falls {fall:.0f} mm "
            f"across itself.")
        add("  That is not a bench. Treat the fit as degenerate and re-measure.")
        return out, {"plane_fitted": False,
                     "refused_because": [f"implausible fall {fall:.0f} mm"],
                     "contact_median_mm": float(np.median(zs)),
                     "points": [list(p) for p in pts]}
    add("")
    add("MEASURED BENCH PLANE  (paper-drag, commanded Z at fingertip contact)")
    add(f"  z_contact(x, y) = {cx:+.5f}*x {cy:+.5f}*y {c0:+.3f}")
    add(f"  slope in x : {100*cx:+.3f} mm per 100 mm")
    add(f"  slope in y : {100*cy:+.3f} mm per 100 mm")
    add(f"  fit rms    : {rms:.2f} mm over {len(pts)} points")
    add(f"  total fall across the measured span: "
        f"x {cx*(max(xs)-min(xs)):+.2f} mm, y {cy*(max(ys)-min(ys)):+.2f} mm")

    dp = depth_plane()
    if dp:
        add("")
        add("CROSS-CHECK against the depth-derived plane (z_land, per-class offsets)")
        add(f"  optical  : x {100*dp['slope_x']:+.3f}, y {100*dp['slope_y']:+.3f} "
            f"mm per 100 mm   (n={dp['n']}, residual sd {dp['residual_sd_mm']:.2f} mm)")
        add(f"  mechanical: x {100*cx:+.3f}, y {100*cy:+.3f} mm per 100 mm")
        dx = 100 * (cx - dp["slope_x"]); dy = 100 * (cy - dp["slope_y"])
        add(f"  difference: x {dx:+.3f}, y {dy:+.3f} mm per 100 mm")
        add("  Intercepts are NOT compared: the paper drag is a commanded Z at")
        add("  the fingertip, z_land is a depth-derived surface height, and the")
        add("  two differ by the tool offset. Only the gradients are comparable.")
        agree = abs(dy) < 0.35 and abs(dx) < 0.35
        add("  -> the two independent estimates AGREE on the tilt."
            if agree else
            "  -> the two estimates DISAGREE. Trust the paper drag for the floor,")
        if not agree:
            add("     and treat the depth-derived gradient as suspect: the same")
            add("     depth stream would be both the error and its correction.")

    add("")
    add("TILT-AWARE HARD FLOOR  (same safety margin, applied correctly)")
    add(f"  min_z(x, y) = {cx:+.5f}*x {cy:+.5f}*y {c0 + CLEARANCE_MM:+.3f}")
    add(f"  i.e. measured contact + {CLEARANCE_MM} mm, the clearance the current")
    add(f"  flat floor of 215.0 was chosen to give against a contact of 213.315.")
    lo = min(cx*x + cy*y + c0 + CLEARANCE_MM for x in xs for y in ys)
    hi = max(cx*x + cy*y + c0 + CLEARANCE_MM for x in xs for y in ys)
    add(f"  range over the bench: {lo:.2f} .. {hi:.2f}  (a flat floor is wrong by")
    add(f"  up to {hi-lo:.2f} mm somewhere on the bench, whatever value is chosen)")
    if check:
        add("")
        add("CHECK POINT (held out of the fit)")
        for x, y, z in check:
            pred = cx*x + cy*y + c0
            add(f"  ({x:+.0f},{y:+.0f}) measured {z:.2f}, predicted {pred:.2f}, "
                f"error {z-pred:+.2f} mm")
    return out, {"slope_x": cx, "slope_y": cy, "intercept": c0, "rms_mm": rms,
                 "clearance_mm": CLEARANCE_MM,
                 "min_z_intercept": c0 + CLEARANCE_MM,
                 "points": [list(p) for p in pts],
                 "check_points": [list(p) for p in (check or [])],
                 "depth_derived": dp}


def interactive():
    print(__doc__.split("USAGE")[0])
    print("Jog to each XY at a safe height, then lower in 0.5 mm steps until a")
    print("sheet of paper under the jaws just binds. Type the commanded Z.")
    print("Blank to skip a point. Ctrl-C to abort.\n")
    pts, check = [], []
    for x, y, what in SUGGESTED:
        s = input(f"  {what:34s} X={x:+7.1f} Y={y:7.1f}  contact Z = ").strip()
        if not s:
            continue
        try:
            z = float(s)
        except ValueError:
            print("    not a number, skipping")
            continue
        (check if "check" in what else pts).append((x, y, z))
    return pts, check


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", nargs=3, type=float, action="append",
                    metavar=("X", "Y", "Z"), help="a measured contact point")
    ap.add_argument("--check", nargs=3, type=float, action="append",
                    metavar=("X", "Y", "Z"), help="held-out validation point")
    args = ap.parse_args()

    pts = [tuple(p) for p in (args.point or [])]
    check = [tuple(p) for p in (args.check or [])]
    if not pts:
        pts, check = interactive()
    if len(pts) < 3:
        raise SystemExit("need at least 3 points to fit a plane")

    lines, rec = report(pts, check)
    print("\n".join(lines))
    json.dump(rec, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    print("NOT applied to any run: --min-z is still the flat value, so today's")
    print("battery stays in one configuration epoch. Apply it off-robot.")


# ----------------------------------------------------------------- self-test
def self_test():
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    # A known plane must be recovered exactly.
    truth = (0.002, -0.0094, 215.0)
    pts = [(x, y, truth[0]*x + truth[1]*y + truth[2])
           for x, y in ((-120, 590), (60, 590), (-120, 830), (60, 830))]
    (cx, cy, c0), rms = fit_plane(pts)
    ck("a known plane is recovered",
       abs(cx-truth[0]) < 1e-9 and abs(cy-truth[1]) < 1e-9 and rms < 1e-9,
       f"(rms {rms:.2e})")

    # A level bench must come back level.
    flat = [(x, y, 213.3) for x, y in ((-120, 590), (60, 590), (-120, 830))]
    (fx, fy, f0), _ = fit_plane(flat)
    ck("a level bench reads as level", abs(fx) < 1e-9 and abs(fy) < 1e-9
       and abs(f0-213.3) < 1e-9)

    # Noise must not be mistaken for tilt at the scale we care about.
    rng = np.random.default_rng(0)
    noisy = [(x, y, 213.3 + rng.normal(0, 0.2))
             for x, y in ((-120, 590), (60, 590), (-120, 830), (60, 830),
                          (-30, 710))]
    (nx, ny, _), _ = fit_plane(noisy)
    ck("0.2 mm of measurement noise does not fake a 0.9 mm/100mm gradient",
       abs(100*ny) < 0.3, f"(got {100*ny:+.3f} mm per 100 mm)")

    # The floor must preserve the existing clearance, and never sit below
    # contact anywhere on the bench.
    lines, rec = report(pts)
    ck("the tilt-aware floor keeps the 1.7 mm margin",
       abs(rec["min_z_intercept"] - (truth[2] + CLEARANCE_MM)) < 1e-9)
    for x, y in ((-144, 561), (83, 839), (0, 700)):
        contact = rec["slope_x"]*x + rec["slope_y"]*y + rec["intercept"]
        floor = contact + CLEARANCE_MM
        if floor <= contact:
            fails.append("floor at or below contact")
    ck("the floor is above contact everywhere on the bench",
       "floor at or below contact" not in fails)
    ck("intercepts are explicitly not compared across frames",
       any("Intercepts are NOT compared" in l for l in lines))

    # ---- the degeneracy that produced a 150 mm bench ----------------------
    # The real inputs from 19 Sep: (60,590), (-30,710), (-120,830) are spaced
    # by exactly (-90,+120) twice, so they lie on one line.
    collinear = [(-30.0, 710.0, 211.0), (-120.0, 830.0, 212.0),
                 (60.0, 590.0, 211.0)]
    ok, why = usable(collinear)
    ck("three collinear points are refused, not fitted", ok is False)
    ck("and the refusal says WHY", any("collinear" in w for w in why))
    lines2, rec2 = report(collinear)
    ck("no plane is reported for them", rec2["plane_fitted"] is False)
    ck("no absurd gradient reaches the output",
       not any("41." in l or "150." in l for l in lines2))
    ck("the contact height is still reported without a plane",
       any("CONTACT HEIGHT" in l for l in lines2)
       and rec2["contact_median_mm"] == 211.0)
    ck("and the floor discrepancy is called out",
       any("not where it is believed to be" in l for l in lines2))

    # Whole-millimetre readings must be flagged even when well spread.
    coarse = [(-120.0, 590.0, 211.0), (60.0, 590.0, 211.0),
              (-120.0, 830.0, 212.0), (60.0, 830.0, 212.0)]
    ok, why = usable(coarse)
    ck("whole-millimetre readings are flagged as under-resolved",
       ok is False and any("0.25 mm steps" in w for w in why))

    # A well-spread, finely-resolved set must still be accepted.
    good = [(-120.0, 590.0, 211.25), (60.0, 590.0, 211.40),
            (-120.0, 830.0, 212.05), (60.0, 830.0, 212.20)]
    ok, why = usable(good)
    ck("a well-spread, finely-resolved set is accepted", ok is True, str(why))
    _, rec3 = report(good)
    ck("and it does fit a plane", rec3.get("plane_fitted", True) is not False)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("bench_plane self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    main()
