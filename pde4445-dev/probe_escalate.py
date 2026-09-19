"""
ESCALATION PROBE — can MORE current tell a thin part from air after all?

Gripper only. NO ARM CODE RUNS HERE. About 10 minutes.

THE QUESTION
Verification currently fails for esp and lcd because their stall position is
indistinguishable from closing on air (measured 8 Sep at 90 raw: air 1674,
esp 1678, lcd 1680). I argued that raising the grip current cannot help, on the
grounds that more force squashes more padding in BOTH cases so the two stalls
move together.

That argument is incomplete, and possibly wrong. The two stalls are set by
DIFFERENT things:

    EMPTY stall   = where motor torque can no longer beat linkage friction.
                    Raise the current ceiling and this moves FURTHER CLOSED.
    LOADED stall  = where the object physically stops the fingers.
                    Set by the part's width. Raising the current only moves it
                    by however much the padding compresses.

If the padding is stiff relative to the part, the loaded stall barely moves while
the empty stall keeps closing — so separation GROWS with current, which is the
opposite of what I claimed. If the padding is very soft, they do move together
and separation stays flat. Which one this rig is, is an empirical question that
has never been measured, because every previous test held the current fixed.

THE MEASUREMENT
Close at a low ceiling, then raise the ceiling in steps WITHOUT re-opening, and
record where the fingers sit at each step. Do it on air, then on the part.

    air:  should keep creeping closed as the ceiling rises (friction is beaten)
    part: should stop moving once the fingers are on the object

The DIFFERENCE between those two curves is the discriminator. Note this works
even if the absolute positions are useless: what matters is that one yields and
the other does not.

    python pde4445-dev/probe_escalate.py            # default ladder
    python pde4445-dev/probe_escalate.py --steps 80,90,100,110,120

SAFETY. The ceiling tops out at 120 because the servo's Current Limit register
refuses more, and because OPENING must beat GRIPPING — a part gripped at 120
cannot be released by a 120 open. The script therefore always drops back to the
release current and opens fully between runs. Do not carry a part gripped at 120.
"""
import os, sys, time, argparse, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gripsense_path import load_settings                          # noqa: E402

HERE          = os.path.dirname(os.path.abspath(__file__))
OUT           = os.path.join(HERE, "probe_escalate.json")
DEFAULT_STEPS = "90,100,110,120"
RELAX_CURRENT = 50
OPEN_CURRENT  = 120
VELOCITY      = 40
DWELL_S       = 1.2      # let each step settle before reading
SETTLE_S      = 3.0
OPEN_S        = 10.0


def settle(g, timeout=SETTLE_S):
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


def set_current(g, want):
    for c in [int(want)] + [v for v in (120, 110, 100, 90, 80) if v < want]:
        try:
            g.set_goal_current(int(c))
            if c != want:
                print(f"      (servo capped {want} -> {c})")
            return int(c)
        except OSError:
            continue
    raise OSError(f"servo rejected every current up to {want}")


def open_fully(g, mx):
    try:
        g.set_goal_position(g.read_present_position())
        set_current(g, RELAX_CURRENT)
        time.sleep(0.3)
    except OSError:
        pass
    g.set_goal_position(mx)
    set_current(g, OPEN_CURRENT)
    p = settle(g, OPEN_S)
    if p < mx - 200:
        print(f"    WARNING: did not fully open (at {p}, expected ~{mx})")
    return p


def escalate(g, mn, steps):
    """Close at the lowest step, then raise the ceiling without re-opening."""
    curve = []
    g.set_goal_position(mn)
    for c in steps:
        set_current(g, c)
        time.sleep(DWELL_S)
        pos = settle(g)
        curve.append({"current": c, "position": int(pos)})
        print(f"      {c:>4} raw -> {pos}")
    return curve


def main():
    ap = argparse.ArgumentParser(description="Does more current separate a thin part from air?")
    ap.add_argument("--steps", default=DEFAULT_STEPS)
    ap.add_argument("--label", default="esp", help="what you are putting in the jaws")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]

    settings, repo = load_settings()
    mx, mn, calibrated = settings.travel_limits()
    if not calibrated:
        raise SystemExit("Gripper not calibrated — run the travel-limit wizard first.")
    print(f"GripSense: {repo}")
    print(f"Limits: max_open={mx} min_open={mn}   ladder: {steps}\n")

    g = settings.connect()
    result = {"label": args.label, "steps": steps, "max_open": mx, "min_open": mn,
              "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        g.set_profile_velocity(VELOCITY)
        g.set_goal_position(mx)
        set_current(g, OPEN_CURRENT)
        g.enable_torque()
        settle(g, OPEN_S)

        print("=== A. EMPTY (fingers clear) ===")
        open_fully(g, mx)
        input("  Fingers CLEAR — confirm, then Enter...")
        result["empty"] = escalate(g, mn, steps)

        print(f"\n=== B. LOADED ({args.label}) ===")
        open_fully(g, mx)
        input(f"  Place the {args.label} between the fingers, then Enter...")
        result["loaded"] = escalate(g, mn, steps)
        input("  Tug it to check it is actually held, then Enter to release...")
        open_fully(g, mx)

        print("\n---- result ----")
        print(f"  {'current':>8} {'empty':>7} {'loaded':>7} {'gap':>6}")
        gaps = []
        for e, l in zip(result["empty"], result["loaded"]):
            gap = l["position"] - e["position"]
            gaps.append(gap)
            print(f"  {e['current']:>8} {e['position']:>7} {l['position']:>7} {gap:>6}")
        result["gaps"] = gaps

        first, last = gaps[0], gaps[-1]
        drift_empty = result["empty"][0]["position"] - result["empty"][-1]["position"]
        drift_load  = result["loaded"][0]["position"] - result["loaded"][-1]["position"]
        print(f"\n  empty fingers moved {drift_empty} ticks closed across the ladder")
        print(f"  loaded fingers moved {drift_load} ticks closed across the ladder")

        # Order matters: an arduino is separated from the first step AND its gap
        # still grows as the empty floor drops, so "already" has to be tested
        # before "opens" or every thick part is reported as a new discovery.
        if first > 25:
            print(f"\n  Already separated at the lowest step ({first} ticks) — the "
                  f"ladder was not needed for '{args.label}'.")
        elif last > 25:
            print(f"\n  SEPARATION OPENS UP: gap {first} -> {last} ticks.")
            print(f"  Proprioceptive verification IS possible for '{args.label}' at "
                  f"{result['loaded'][-1]['current']} raw.")
            print(f"  Threshold would be ~{(result['loaded'][-1]['position'] + result['empty'][-1]['position']) // 2}.")
            print("  CAUTION before using it: gripping near the 120 ceiling leaves no")
            print("  headroom to open again. Confirm release works before trusting it.")
        else:
            print(f"\n  FLAT: gap stays around {min(gaps)}-{max(gaps)} ticks across "
                  f"the whole ladder.")
            print(f"  Current cannot buy separation for '{args.label}' on this rig.")
            print("  The padding compresses as fast as the friction floor drops.")
            print("  Vision verification is the answer, and this is the evidence.")

        json.dump(result, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}  — send me this file")
    finally:
        try:
            open_fully(g, mx)
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

    def verdict(empty, loaded):
        gaps = [l - e for e, l in zip(empty, loaded)]
        first, last = gaps[0], gaps[-1]
        if first > 25:
            return "already"
        if last > 25:
            return "opens"
        return "flat"

    # the outcome I predicted: padding compresses as fast as the floor drops
    ck("flat curves are called flat",
       verdict([1674, 1650, 1630, 1612], [1678, 1654, 1633, 1615]) == "flat")
    # the outcome the user is suggesting: floor drops, object holds station
    ck("a separating curve is detected",
       verdict([1674, 1620, 1570, 1520], [1678, 1676, 1675, 1674]) == "opens")
    # an arduino: separated from the very first step
    ck("an already-separated part is not misread as 'opens'",
       verdict([1674, 1650, 1630, 1612], [1860, 1858, 1857, 1856]) == "already")

    print("\n" + "=" * 56)
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("probe_escalate self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    main()
