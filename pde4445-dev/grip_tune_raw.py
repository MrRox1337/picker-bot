"""
RAW-CURRENT GRIP TUNING — runs on LAPTOP 2. No robot, no network.

Why this exists: GripperAPI's normalised grip strength maps 0.0 -> current.min
and 1.0 -> current.max, and Aman's config sets those to the benchmark band
(100..120 raw = 269..323 mA). Even strength 0.0 therefore squeezes ~20 N, far
more than a 25 g PCB module needs. This script commands GOAL CURRENT DIRECTLY,
so we can explore BELOW that floor and find the real minimum holding current.

It uses the driver path documented in Aman's README (gripper_settings.connect)
and CHANGES NONE OF HIS CONFIG FILES.

COPY THIS FILE into Aman's repo (next to Scripts/) and run it there:
    python grip_tune_raw.py

Optional: keep the kitchen scale handy. Entering the reading at each level
builds a grip-force-vs-current table for PLA fingers - useful thesis data and
directly comparable with Aman's benchmark.

SAFETY
  * The ARM NEVER MOVES here - no arm code runs.
  * Torque is released in the finally block, including on Ctrl-C.
"""
import sys, time
from pathlib import Path


def find_repo():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (base, *base.parents):
            if (d / "Config" / "gripper_config.yaml").exists() and (d / "Lib").is_dir():
                return d
    raise SystemExit("Could not find Aman's repo (needs Config/ and Lib/). "
                     "Copy this script into the GripSense folder and run it there.")


REPO = find_repo()
sys.path.insert(0, str(REPO / "Lib"))
import gripper_settings as settings              # noqa: E402

MAX_OPEN = None      # filled in from the calibrated limits at startup

# ---- sweep, gentlest first. 1 raw unit ~ 2.69 mA ----
# Below ~25 raw the fingers do not move at all: the linkage has a friction floor
# that must be overcome before any grip force exists. So start above it and go finer.
CURRENTS            = [30, 40, 50, 60, 70, 80, 90, 100]
OPEN_CURRENT        = 100     # only for OPENING (needs to overcome friction); not the grip force
MISS_TOLERANCE_TICKS = 25     # ended this close to min_open => nothing was between the fingers
NOMOVE_TICKS        = 15      # moved less than this from fully open => never left the stop
SETTLE_TIMEOUT_S    = 3.0
VELOCITY            = 40      # slow-ish, so a stall is gentle


def open_fingers(g):
    """Re-open safely: RE-TARGET FIRST, then raise the current ceiling.

    Order matters. Raising the ceiling while the goal is still min_open would
    give the servo full force with a CLOSED target - an unintended hard slam.
    """
    g.set_goal_position(MAX_OPEN)
    g.set_goal_current(OPEN_CURRENT)
    return settle(g)


def settle(g):
    """Wait until the fingers stop moving; return final (position, current)."""
    last, t0 = None, time.time()
    while time.time() - t0 < SETTLE_TIMEOUT_S:
        p = g.read_present_position()
        if last is not None and abs(p - last) <= 2:
            break
        last = p
        time.sleep(0.06)
    return g.read_present_position(), g.read_present_current()


def main():
    global MAX_OPEN
    max_open, min_open, calibrated = settings.travel_limits()
    if not calibrated:
        raise SystemExit("Not calibrated - run the calibration wizard first.")
    MAX_OPEN = max_open
    print(f"Repo: {REPO}")
    print(f"Limits: max_open={max_open}  min_open={min_open}\n")

    g = settings.connect()                 # opens port, sets Operating Mode 5
    rows = []
    try:
        g.set_profile_velocity(VELOCITY)
        g.set_goal_position(max_open)      # target FIRST, then the ceiling (see open_fingers)
        g.set_goal_current(OPEN_CURRENT)
        g.enable_torque()
        settle(g)
        print("Torque ON, fingers open. The ARM WILL NOT MOVE.\n")

        chosen = None
        for c in CURRENTS:
            print(f"\n=== goal current {c} raw  (~{c*2.69:.0f} mA) ===")
            start, _ = open_fingers(g)

            input("  Place the module between the fingers, then Enter...")
            g.set_goal_current(c)                 # the grip-force ceiling under test
            g.set_goal_position(min_open)         # drive closed; stalls on the object
            pos, cur = settle(g)

            moved   = abs(pos - start) > NOMOVE_TICKS
            missed  = (pos - min_open) <= MISS_TOLERANCE_TICKS
            print(f"    from {start} -> {pos} ticks   current={cur} raw (~{abs(cur)*2.69:.0f} mA)")

            if not moved:
                print("    -> DID NOT MOVE: below the friction floor, no grip force at all.")
                open_fingers(g)
                rows.append((c, pos, cur, "-", "nomove"))
                continue
            if missed:
                print("    -> MISS: closed all the way, nothing between the fingers.")
                open_fingers(g)
                rows.append((c, pos, cur, "-", "miss"))
                continue
            print("    -> HOLDING: stopped short, stalled against the module.")

            scale = input("  Kitchen scale reading in g (Enter to skip): ").strip()
            ans = input("  Gently TUG. Held, without marking the board? [y/n/s=stop] ").strip().lower()
            rows.append((c, pos, cur, scale or "-", ans[:1]))

            open_fingers(g)

            if ans.startswith("s"):
                break
            if ans.startswith("y"):
                chosen = c
                print(f"    -> holds at {c} raw")
                break
            print("    -> slipped; trying firmer.")

        print("\n---- summary ----")
        print(f"  {'raw':>4} {'mA':>6} {'pos':>6} {'scale_g':>8}  held")
        for c, pos, cur, scale, held in rows:
            print(f"  {c:>4} {c*2.69:>6.0f} {pos:>6} {scale:>8}  {held}")

        if chosen:
            print(f"\nRESULT: minimum holding current = {chosen} raw (~{chosen*2.69:.0f} mA).")
            print("Tell me this number and the scale readings.")
        else:
            print("\nNothing in the sweep held. Tell me the table above and we will go firmer,")
            print("or add padding (rubber/sponge) so a lower force still grips.")
    finally:
        try:
            open_fingers(g)
            g.disable_torque()
        finally:
            g.close()                      # closes the SERIAL PORT
        print("\nTorque released, port closed.")


if __name__ == "__main__":
    main()
