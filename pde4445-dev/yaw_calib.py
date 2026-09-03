"""
YAW CALIBRATION — solve how the part's long axis maps to the wrist U (J6).

Model:      U = SIGN * yaw + OFFSET      (mod 180: a parallel jaw is symmetric)

Two unknowns. SIGN can be -1 because with the gripper pointing DOWN the tool Z is
anti-parallel to base +Z, which can reverse the sense of rotation. Last attempt
fitted OFFSET=54 from effectively ONE good point, which is why alignment is off.
This records several pairs and solves properly, with residuals so you can SEE
whether the fit is trustworthy.

USE ONLY ELONGATED PARTS. pose_seg now prints an `aspect` column: anything below
~1.25 is near-square and its long axis is ambiguous - such parts CANNOT calibrate
yaw and are excluded automatically.

PRE-REQ
  * pick_list.json fresh, from the CURRENT scene (pose_seg.py)
  * LOOK AT pose_seg_out.png FIRST: the magenta line is the measured long axis.
    If that line does not lie along the part's long side, the VISION yaw is wrong
    and no offset will fix it - tell me instead of calibrating.
  * Main.prg running; arm jogged to the GRIPPER-DOWN ready pose.

Commands:  <number> set U     +N / -N nudge U     r record this pair
           p<i> switch part   s solve now         q quit
Run:  python pde4445-dev/yaw_calib.py
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from pickerbot_lib import sender

PICK_LIST        = os.path.join(HERE, "pick_list.json")
OUT              = os.path.join(HERE, "yaw_calib.json")
APPROACH_MM      = 80.0
TOOL_Z_OFFSET_MM = 70.0      # keep in sync with the pick script
MIN_ASPECT       = 1.25      # below this the long axis is ambiguous


def wrap180(a):
    """Fold an angle into (-90, 90] — the symmetry of a parallel jaw."""
    return ((a + 90.0) % 180.0) - 90.0


def solve(pairs):
    """Fit OFFSET for BOTH signs. Returns both, best first.

    Comparing the two RMS values is how we detect a degenerate scene: if the
    recorded yaws are all ~0 and ~90 then +90 == -90 (mod 180), both signs fit
    equally well, and the sign is simply not determined by this data.
    """
    fits = []
    for sign in (+1, -1):
        offs = [wrap180(u - sign * y) for y, u in pairs]
        # circular mean, doubled so that 180-periodic angles average correctly
        sx = sum(math.sin(math.radians(2 * o)) for o in offs)
        cx = sum(math.cos(math.radians(2 * o)) for o in offs)
        mean = math.degrees(math.atan2(sx, cx)) / 2.0
        resid = [wrap180(o - mean) for o in offs]
        rms = math.sqrt(sum(r * r for r in resid) / len(resid))
        fits.append({"sign": sign, "offset": mean, "rms": rms, "resid": resid})
    fits.sort(key=lambda f: f["rms"])
    return fits


def main():
    picks = json.load(open(PICK_LIST))
    usable = [(i, p) for i, p in enumerate(picks) if p.get("aspect", 99) >= MIN_ASPECT]
    if not usable:
        raise SystemExit("No sufficiently elongated parts in the scene. Lay out some "
                         "clearly rectangular modules (lcd / esp) and re-run pose_seg.")

    print("\nElongated parts usable for yaw calibration:")
    for i, p in usable:
        print(f"  [{i}] {p['label']:10s} yaw={p['yaw']:7.1f}  aspect={p.get('aspect')}  "
              f"@({p['x']:.0f}, {p['y']:.0f}, {p['z']:.0f})")
    excluded = len(picks) - len(usable)
    if excluded:
        print(f"  ({excluded} near-square part(s) excluded — ambiguous long axis)")

    print("\nAim for at least 3 pairs with WIDELY DIFFERENT yaws.")
    print("If you have only one elongated part, align it, record, then physically")
    print("rotate it, re-run pose_seg, and run this again to add more pairs.\n")

    sender.connect()
    pairs = []

    def hover(idx, u):
        p = picks[idx]
        z = p["z"] + APPROACH_MM - TOOL_Z_OFFSET_MM
        reply = sender.epsonJump(p["x"], p["y"], z, u)
        if "OK" not in str(reply).upper():
            print(f"  !! arm non-OK: {reply!r}  (out of range? try U -/+ 180)")
        else:
            print(f"  part[{idx}] {p['label']}  vision yaw={p['yaw']:.1f}  ->  U={u:.1f}")

    idx = usable[0][0]
    u = picks[idx]["yaw"]
    hover(idx, u)
    print("\nNudge U until the OPEN JAWS STRADDLE THE SHORT WIDTH, then press 'r'.\n")

    try:
        while True:
            cmd = input(f"[part {idx}  U={u:.1f}  pairs={len(pairs)}] > ").strip()
            if not cmd:
                continue
            if cmd == "q":
                break
            if cmd == "r":
                y = picks[idx]["yaw"]
                if any(abs(y - py) < 1 and abs(u - pu) < 1 for py, pu in pairs):
                    print("  ALREADY RECORDED this exact pair - it adds no information.")
                    print("  Switch part with p<i>, or physically rotate a part, then record.")
                    continue
                pairs.append((y, u))
                print(f"  recorded (yaw={y:.1f}, U={u:.1f})   [{len(pairs)} pairs]")
                spread = sorted(wrap180(py) for py, _ in pairs)
                if len(pairs) >= 2 and (max(spread) - min(spread)) < 20:
                    print("  NOTE: all recorded yaws are close together - the fit will be weak.")
                continue
            if cmd == "s":
                break
            if cmd.startswith("p"):
                try:
                    idx = int(cmd[1:]); u = picks[idx]["yaw"]; hover(idx, u)
                except (ValueError, IndexError):
                    print("  usage: p<index>")
                continue
            try:
                u = (u + float(cmd)) if cmd[0] in "+-" else float(cmd)
            except ValueError:
                print("  enter a number, +N, -N, r, p<i>, s, or q")
                continue
            hover(idx, u)
    finally:
        sender.disconnect()

    if len(pairs) < 2:
        print("\nNeed at least 2 pairs to solve. Nothing saved.")
        return

    fits = solve(pairs)
    r, other = fits[0], fits[1]
    print("\n---- fit ----")
    for f in fits:
        print(f"  sign {f['sign']:+d}:  U = {f['sign']:+d}*yaw + {f['offset']:7.1f}   "
              f"RMS {f['rms']:5.1f} deg")
    print(f"\n  best: U = {r['sign']:+d} * yaw + {r['offset']:.1f}   (mod 180)"
          f"   over {len(pairs)} pairs")
    for (y, u_), res in zip(pairs, r["resid"]):
        mark = "   <- OUTLIER" if abs(res) > 15 else ""
        print(f"    yaw={y:7.1f}  U={u_:7.1f}   residual {res:+6.1f}{mark}")

    if abs(r["rms"] - other["rms"]) < 3:
        print("\n  SIGN NOT DETERMINED: both signs fit equally well. Your recorded yaws")
        print("  are probably only ~0 and ~90 apart, where +90 == -90 (mod 180).")
        print("  Rotate 2-3 parts to INTERMEDIATE angles (~30, 45, 60 deg),")
        print("  re-run pose_seg.py, and record those too. Not saving.")
        return

    if r["rms"] > 15:
        print("\n  POOR FIT. The vision yaw is probably unreliable (check the magenta")
        print("  long-axis lines in pose_seg_out.png) rather than the offset being wrong.")
        print("  Send me these numbers instead of trusting this.")
    else:
        json.dump({"sign": r["sign"], "offset_deg": round(r["offset"], 1),
                   "rms_deg": round(r["rms"], 1), "pairs": pairs},
                  open(OUT, "w"), indent=2)
        print(f"\n  Saved -> {OUT}")
        print("  The pick script will read this instead of the hardcoded 54.")


if __name__ == "__main__":
    main()
