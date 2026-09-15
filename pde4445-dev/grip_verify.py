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
    python pde4445-dev/grip_verify.py 100        # or pass a current
    python pde4445-dev/grip_verify.py 100 esp    # ...and record it as a CLASS PROFILE

The third form appends to grip_profiles.json. Because each class arrests the
fingers at its own characteristic position, storing that position per class turns
verification from "something is between the fingers" into "an object of roughly
THIS class's width is between the fingers" - which also catches an edge-grasp or
two parts taken at once. Vision supplies the class; the servo checks the width.

SAFETY: fingers must be CLEAR during the empty measurements.
Torque is released in the finally block.
"""
import sys, os, time, json, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gripsense_path import load_settings          # noqa: E402

PROFILES = os.path.join(HERE, "grip_profiles.json")
PROFILE_TOL = 90       # +/- ticks a stall may sit from its class mean and still
                       # count as that class. Wide enough for placement variation,
                       # tight enough that an edge-grasp falls outside.

settings, REPO = load_settings()

# OPERATING RANGE IS 100-120 RAW (Aman's gripper_config.yaml: current.min 100,
# current.max 120). Everything measured before 12 Sep used 80-90, below the floor,
# where the fingers stall at ~1646 on friction and never touch a thin module.
TEST_CURRENT     = 100    # the grip current we actually pick with
OPEN_CURRENT     = 120    # opening must exceed the grip; 120 is the servo's cap
RELAX_CURRENT    = 100    # never command below the configured minimum
VELOCITY         = 60     # closing speed; the calibration probes at 60
OPEN_VELOCITY    = 480    # the calibration REOPENS at 480 - at 40 the open crawls
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
    # Match what the teleop GUI does, which opens these fingers cleanly: retarget,
    # go fast, 120 raw. The old relax-at-50 step belonged to the sub-100 regime and
    # left this function ending MORE CLOSED than it started.
    try:
        _G.set_profile_velocity(OPEN_VELOCITY)
    except OSError:
        pass
    _G.set_goal_position(MAX_OPEN)          # re-target BEFORE raising the ceiling
    set_current(OPEN_CURRENT)
    pos, cur = settle(OPEN_TIMEOUT_S)
    if pos < MAX_OPEN - 200:
        print(f"    WARNING: fingers did not fully open (at {pos}, expected ~{MAX_OPEN})")
    return pos, cur


def close_on(current, min_open):
    try:
        _G.set_profile_velocity(VELOCITY)   # slow onto the object
    except OSError:
        pass
    set_current(current)
    _G.set_goal_position(min_open)
    return settle()


def save_profile(label, current, empty, loaded, max_open, min_open):
    """Record this class's stall band so a later grasp can be checked against it."""
    data = {}
    if os.path.exists(PROFILES):
        try:
            data = json.load(open(PROFILES))
        except ValueError:
            print("  (grip_profiles.json was unreadable - starting a new one)")
    # A profile is only comparable within one calibration and one grip current.
    # If either changed, everything already in the file is void.
    key = {"calib_max_open": max_open, "calib_min_open": min_open,
           "grip_current": current}
    if data.get("key") and data["key"] != key:
        print(f"  *** calibration or current changed {data['key']} -> {key}")
        print(f"  *** discarding {len(data.get('classes', {}))} stale class profile(s)")
        data = {}
    data["key"] = key
    data["empty_stall"] = int(statistics.mean(empty))
    data.setdefault("classes", {})[label] = {
        "stall": int(statistics.mean(loaded)),
        "spread": int(max(loaded) - min(loaded)),
        "tol": PROFILE_TOL,
        "samples": list(map(int, loaded)),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(data, open(PROFILES, "w"), indent=2)
    print(f"\n  profile saved: {label} stalls at "
          f"{data['classes'][label]['stall']} +/- {PROFILE_TOL}  -> {PROFILES}")
    known = data["classes"]
    if len(known) > 1:
        ordered = sorted(known.items(), key=lambda kv: kv[1]["stall"])
        print("  classes recorded so far (empty = %d):" % data["empty_stall"])
        for name, c in ordered:
            print(f"    {name:11} {c['stall']:>5}")
        gaps = [b[1]["stall"] - a[1]["stall"] for a, b in zip(ordered, ordered[1:])]
        if min(gaps) < 2 * PROFILE_TOL:
            print(f"  *** two classes are only {min(gaps)} ticks apart, closer than")
            print(f"  *** the +/-{PROFILE_TOL} tolerance - they cannot be told apart "
                  f"by stall alone.")


def main():
    global MAX_OPEN, _G
    current = int(sys.argv[1]) if len(sys.argv) > 1 else TEST_CURRENT
    label = sys.argv[2] if len(sys.argv) > 2 else None
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
        if label:
            save_profile(label, current, empty, loaded, max_open, min_open)
        else:
            print("\n  (pass a class name as the 2nd argument to record a profile,")
            print("   e.g.  python pde4445-dev/grip_verify.py 100 esp)")
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
