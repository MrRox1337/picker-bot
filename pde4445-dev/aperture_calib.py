"""
APERTURE CALIBRATION — how many servo ticks is a millimetre of jaw gap?

Gripper only: NO ARM CODE RUNS HERE. About ten minutes with a ruler or calipers.

WHY MEASURE INSTEAD OF CLOSING ON KNOWN OBJECTS
The obvious method - close onto parts of known width and record where the fingers
stall - is contaminated by exactly the effect that makes thin parts invisible:
under this rig's linkage friction the fingers stall at ~1674 whether or not
anything is there, and compliant padding lets a thin board through. Opening to a
commanded position and MEASURING the gap avoids contact altogether, so friction
and padding never enter the number.

RUN
    python pde4445-dev/aperture_calib.py
    python pde4445-dev/aperture_calib.py --points 6

RE-RUN AFTER ANY FINGER OR CALIBRATION CHANGE. The mapping is a property of the
finger geometry, so the glued finger and the spare finger need separate runs, and
recalibrating the travel limits voids it outright.

Writes aperture.json, which pick_one.Gripper loads to decide how far to open
before descending on a part of known width.
"""
import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gripsense_path import load_settings                          # noqa: E402

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT      = os.path.join(HERE, "aperture.json")
OPEN_CURRENT     = 120
VELOCITY         = 40
SETTLE_TIMEOUT_S = 8.0


def settle(g, timeout=SETTLE_TIMEOUT_S):
    t0 = time.time()
    time.sleep(0.20)
    last, stable = None, 0
    while time.time() - t0 < timeout:
        p = g.read_present_position()
        if last is not None and abs(p - last) <= 2:
            stable += 1
            if stable >= 3:
                return p
        else:
            stable = 0
        last = p
        time.sleep(0.06)
    return g.read_present_position()


def fit(pairs):
    """Least-squares ticks = a*mm + b. Returns (a, b, rms_mm, residuals_mm)."""
    n = len(pairs)
    sx = sum(mm for mm, _ in pairs)
    sy = sum(tk for _, tk in pairs)
    sxx = sum(mm * mm for mm, _ in pairs)
    sxy = sum(mm * tk for mm, tk in pairs)
    den = n * sxx - sx * sx
    if abs(den) < 1e-9:
        raise SystemExit("All measurements were at the same gap — spread them out.")
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    # Residuals converted BACK to millimetres: a tick residual is meaningless to
    # judge by eye, but "this fit is good to 1.2 mm" is directly comparable to the
    # 8 mm clearance the aperture leaves either side.
    res = [(tk - (a * mm + b)) / a for mm, tk in pairs]
    rms = (sum(r * r for r in res) / n) ** 0.5
    return a, b, rms, res


def main():
    ap = argparse.ArgumentParser(description="Calibrate jaw gap (mm) against servo ticks.")
    ap.add_argument("--points", type=int, default=5, help="measurements to take (>=3)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.points < 3:
        raise SystemExit("Use at least 3 points — 2 cannot show a bad fit.")

    settings, repo = load_settings()
    mx, mn, calibrated = settings.travel_limits()
    if not calibrated:
        raise SystemExit("Gripper not calibrated — run the travel-limit wizard first.")
    print(f"GripSense: {repo}")
    print(f"Travel limits: max_open={mx}  min_open={mn}")
    print("\nThe jaws will open to a series of positions. At each one, measure the")
    print("GAP BETWEEN THE INNER FACES with a ruler or calipers and type it in mm.")
    print("Measure at the fingertips — that is where the parts are.\n")

    g = settings.connect()
    pairs = []
    try:
        g.set_profile_velocity(VELOCITY)
        g.set_goal_position(mx)
        g.set_goal_current(OPEN_CURRENT)
        g.enable_torque()
        settle(g)

        span = mx - mn
        # Spread the samples over the USEFUL range. Below about 25% of travel the
        # jaws are narrower than any module, so a point there adds nothing to a fit
        # that will only ever be evaluated between ~25 mm and ~70 mm.
        targets = [int(mn + span * f) for f in
                   [0.35 + 0.65 * i / (args.points - 1) for i in range(args.points)]]
        for i, tgt in enumerate(targets, 1):
            g.set_goal_position(tgt)
            g.set_goal_current(OPEN_CURRENT)
            pos = settle(g)
            if abs(pos - tgt) > 60:
                print(f"  [{i}/{len(targets)}] asked {tgt}, fingers reached {pos} "
                      f"— using the ACTUAL position")
            raw = input(f"  [{i}/{len(targets)}] position {pos} — measured gap in mm "
                        f"(or 's' to skip): ").strip()
            if raw.lower().startswith("s"):
                continue
            try:
                pairs.append((float(raw), int(pos)))
            except ValueError:
                print("     not a number, skipped")

        if len(pairs) < 3:
            raise SystemExit("Fewer than 3 usable measurements — nothing written.")

        a, b, rms, res = fit(pairs)
        print("\n---- fit ----")
        print(f"  ticks = {a:.2f} * mm + {b:.1f}")
        print(f"  rms residual: {rms:.2f} mm")
        for (mm, tk), r in zip(pairs, res):
            flag = "   <-- OUTLIER" if abs(r) > 3 * max(rms, 0.1) else ""
            print(f"    {mm:6.1f} mm -> {tk:5d} ticks   residual {r:+.2f} mm{flag}")

        if a <= 0:
            raise SystemExit("Slope is negative — more ticks should mean a WIDER gap. "
                             "Check the measurements were of the gap, not the finger.")
        if rms > 3.0:
            print("\n  *** rms > 3 mm. The aperture leaves 8 mm of clearance either")
            print("  *** side, so a fit this loose eats most of that margin. Re-measure")
            print("  *** before trusting it, or widen JAW_CLEARANCE_MM in pick_one.py.")

        print("\n  predicted apertures (part width + 2 x 8 mm clearance):")
        for name, w in (("esp", 28), ("lcd", 36), ("arduino", 53)):
            t = int(min(max(a * (w + 16) + b, mn + 50), mx))
            print(f"    {name:9} {w} mm -> {t:5d} ticks   "
                  f"({100 * (mx - t) / (mx - mn):.0f}% less travel than full open)")

        json.dump({"slope_ticks_per_mm": round(a, 3),
                   "intercept_ticks": round(b, 1),
                   "rms_mm": round(rms, 2),
                   "calib_max_open": mx,
                   "calib_min_open": mn,
                   "measured": [{"gap_mm": mm, "ticks": tk} for mm, tk in pairs],
                   "when": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")
        print("  pick_one.Gripper picks this up automatically on connect().")
    finally:
        try:
            g.set_goal_position(mx)
            g.set_goal_current(OPEN_CURRENT)
            settle(g)
            g.disable_torque()
        finally:
            g.close()
        print("\nTorque released, port closed.")


def self_test():
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    # a perfectly linear rig
    truth_a, truth_b = 38.0, 700.0
    pairs = [(mm, truth_a * mm + truth_b) for mm in (20, 35, 50, 65, 80)]
    a, b, rms, res = fit(pairs)
    ck("recovers the slope", abs(a - truth_a) < 1e-6, f"({a})")
    ck("recovers the intercept", abs(b - truth_b) < 1e-6, f"({b})")
    ck("reports zero residual on exact data", rms < 1e-9, f"({rms})")

    # one fat-fingered measurement
    bad = list(pairs); bad[2] = (50, truth_a * 50 + truth_b + 400)   # ~10mm out
    a2, b2, rms2, res2 = fit(bad)
    ck("an outlier shows up in the rms", rms2 > 2.0, f"(rms {rms2:.2f} mm)")
    ck("and is the largest residual", abs(res2[2]) == max(abs(r) for r in res2),
       f"({[round(r,1) for r in res2]})")

    ck("residuals are in mm, not ticks", abs(res2[2]) < 20,
       f"(largest residual {res2[2]:.1f})")

    print("\n" + "=" * 56)
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("aperture_calib self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    main()
