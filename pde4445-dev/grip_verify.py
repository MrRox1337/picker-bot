"""
GRASP-VERDICT VERIFICATION — measure the holding threshold for the CURRENT
finger calibration. Gripper only: NO ARM CODE RUNS HERE.

Why it matters: under this rig's linkage friction the fingers stall part-way even
with NOTHING between them, so "stopped short of min_open" does not prove a grasp.
We measure both cases and see whether they separate:

    EMPTY closes   -> where do the fingers stall on air?
    LOADED closes  -> where do they stall on the module?

Run this AFTER EVERY RECALIBRATION. Positions are only meaningful relative to the
calibration they were measured under - the old threshold of 1230 was taken when
min_open was -53 and is meaningless now that min_open is 888.

Run from picker-bot (GripSense is found automatically as a sibling repo):
    python pde4445-dev/grip_verify.py            # uses TEST_CURRENT
    python pde4445-dev/grip_verify.py 110        # or pass a current

SAFETY: fingers must be CLEAR during the empty measurements.
Torque is released in the finally block.
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gripsense_path import load_settings          # noqa: E402

settings, REPO = load_settings()

TEST_CURRENT     = 110    # the grip current we actually pick with
OPEN_CURRENT     = 120    # opening must exceed the grip; 120 is the servo's cap
RELAX_CURRENT    = 50     # relax before opening so padding decompresses
VELOCITY         = 40
REPEATS          = 3
SETTLE_TIMEOUT_S = 3.0
OPEN_TIMEOUT_S   = 10.0
MAX_OPEN         = None
_G               = None


def set_current(want):
    """Set goal current, stepping down if the servo's Current Limit rejects it."""
    want = int(want)
    ladder = [want] + [v for v in (120, 110, 100, 90, 80) if v < want]
    last = None
    for c in ladder:
        try:
            _G.set_goal_current(int(c))
            if c != want:
                print(f"    (servo capped goal current {want} -> {c})")
            return int(c)
        except OSError as e:
            last = e
    raise OSError(f"servo rejected every goal current up to {want}: {last}")


def settle(timeout=SETTLE_TIMEOUT_S):
    """Wait for the fingers to actually stop.

    Comparing two reads 60 ms apart is not enough: right after a command the
    servo has not started moving, so those reads match and we return instantly,
    reporting the STARTING position as if it were the stall. Let the move begin,
    then require several consecutive stable reads.
    """
    t0 = time.time()
    time.sleep(0.20)
    last, stable = None, 0
    while time.time() - t0 < timeout:
        p = _G.read_present_position()
        if last is not None and abs(p - last) <= 2:
            stable += 1
            if stable >= 3:
                return p, _G.read_present_current()
        else:
            stable = 0
        last = p
        time.sleep(0.06)
    return _G.read_present_position(), _G.read_present_current()


def open_fingers():
    # Relax first: muscling out of a firm grip does not work when only ~10 raw
    # of headroom remains under the servo's current cap.
    try:
        _G.set_goal_position(_G.read_present_position())
        set_current(RELAX_CURRENT)
        time.sleep(0.3)
    except OSError:
        pass
    _G.set_goal_position(MAX_OPEN)          # re-target BEFORE raising the ceiling
    set_current(OPEN_CURRENT)
    pos, cur = settle(OPEN_TIMEOUT_S)
    if pos < MAX_OPEN - 200:
        print(f"    WARNING: fingers did not fully open (at {pos}, expected ~{MAX_OPEN})")
    return pos, cur


def close_on(current, min_open):
    set_current(current)
    _G.set_goal_position(min_open)
    return settle()


def main():
    global MAX_OPEN, _G
    current = int(sys.argv[1]) if len(sys.argv) > 1 else TEST_CURRENT
    max_open, min_open, calibrated = settings.travel_limits()
    if not calibrated:
        raise SystemExit("Not calibrated - run the calibration wizard first.")
    MAX_OPEN = max_open
    print(f"GripSense: {REPO}")
    print(f"Limits: max_open={max_open}  min_open={min_open}")
    print(f"Testing at {current} raw (~{current*2.69:.0f} mA)\n")

    _G = settings.connect()
    empty, loaded, scales = [], [], []
    try:
        _G.set_profile_velocity(VELOCITY)
        _G.set_goal_position(max_open)
        set_current(OPEN_CURRENT)
        _G.enable_torque()
        settle(OPEN_TIMEOUT_S)
        print("Torque ON. The ARM WILL NOT MOVE.\n")

        print("=== A. EMPTY closes (nothing between the fingers) ===")
        for i in range(1, REPEATS + 1):
            open_fingers()
            input(f"  [{i}/{REPEATS}] Fingers CLEAR - confirm, then Enter...")
            pos, cur = close_on(current, min_open)
            empty.append(pos)
            print(f"      stalled at {pos} ticks   current={cur} raw")

        print("\n=== B. LOADED closes (module between the fingers) ===")
        for i in range(1, REPEATS + 1):
            open_fingers()
            input(f"  [{i}/{REPEATS}] Place the module the SAME way each time, then Enter...")
            pos, cur = close_on(current, min_open)
            loaded.append(pos)
            print(f"      stalled at {pos} ticks   current={cur} raw")
            input("      Tug it, then Enter to reopen...")

        open_fingers()

        print("\n---- result ----")
        print(f"  EMPTY  stalls: {empty}   (max {max(empty)})")
        print(f"  LOADED stalls: {loaded}  (min {min(loaded)})")

        # An object stops the fingers EARLIER, so loaded sits ABOVE empty.
        gap = min(loaded) - max(empty)
        if gap > 25:
            thresh = (min(loaded) + max(empty)) // 2
            print(f"\n  SEPARATED by {gap} ticks.")
            print(f"  HOLDING  <=>  stall position ABOVE ~{thresh} ticks.")
            print(f"  Put HOLD_THRESHOLD = {thresh} in pick_one.py.")
        else:
            print(f"\n  OVERLAP (gap {gap} ticks). Position cannot tell a grasp from air")
            print("  at this current - the grasp needs visual confirmation instead.")
        print("\nSend me these numbers.")
    finally:
        try:
            open_fingers()
            _G.disable_torque()
        finally:
            _G.close()
        print("\nTorque released, port closed.")


if __name__ == "__main__":
    main()
