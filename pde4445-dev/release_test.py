"""
RELEASE TEST — can the gripper let go? Gripper only, NO ARM CODE.

The problem: the servo's Current Limit caps Goal Current at 120 raw, so OPENING
can never pull harder than GRIPPING did. With sponge padding the fingers grip a
board at 90 and then cannot back off it - they move ~40 ticks and stall. A
gripper that cannot release cannot complete a pick-place cycle.

Three release strategies, timed back to back on the same grip:

  A  relax to 50 raw for 0.3 s, then open at 120      (what we do now - fails)
  B  relax to 50 raw for 2.0 s, then open at 120      (sponge is visco-elastic:
                                                       given time at low load it
                                                       creeps and sheds normal force)
  C  DISABLE TORQUE for 1 s, then re-enable and open  (back-drivable: the
                                                       compressed sponge springs
                                                       the fingers apart itself,
                                                       needing no current at all,
                                                       so the 120 ceiling is
                                                       irrelevant)

Whichever reaches ~max_open is the release strategy to adopt.

Run:  python pde4445-dev/release_test.py          # default grip 90
      python pde4445-dev/release_test.py 110
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gripsense_path import load_settings          # noqa: E402

settings, REPO = load_settings()

GRIP_CURRENT  = int(sys.argv[1]) if len(sys.argv) > 1 else 90
OPEN_CURRENT  = 120        # the servo's hard ceiling
RELAX_CURRENT = 50
VELOCITY      = 40
_G = None
MAX_OPEN = MIN_OPEN = None


def set_current(want):
    for c in [int(want)] + [v for v in (120, 110, 100, 90, 80) if v < want]:
        try:
            _G.set_goal_current(int(c))
            return int(c)
        except OSError:
            continue
    raise OSError("servo rejected every goal current")


def settle(timeout=4.0):
    t0 = time.time()
    time.sleep(0.20)
    last, stable = None, 0
    while time.time() - t0 < timeout:
        p = _G.read_present_position()
        if last is not None and abs(p - last) <= 2:
            stable += 1
            if stable >= 3:
                return p
        else:
            stable = 0
        last = p
        time.sleep(0.06)
    return _G.read_present_position()


def grip():
    set_current(GRIP_CURRENT)
    _G.set_goal_position(MIN_OPEN)
    return settle()


def try_open(strategy):
    if strategy == "A":
        _G.set_goal_position(_G.read_present_position())
        set_current(RELAX_CURRENT)
        time.sleep(0.3)
    elif strategy == "B":
        _G.set_goal_position(_G.read_present_position())
        set_current(RELAX_CURRENT)
        time.sleep(2.0)
    elif strategy == "C":
        _G.disable_torque()                 # back-drivable: sponge pushes fingers apart
        time.sleep(1.0)
        freed = _G.read_present_position()
        print(f"      (torque off -> fingers sprang to {freed})")
        _G.set_goal_position(MAX_OPEN)      # target BEFORE re-enabling torque
        _G.enable_torque()
    _G.set_goal_position(MAX_OPEN)
    set_current(OPEN_CURRENT)
    return settle(10.0)


def main():
    global _G, MAX_OPEN, MIN_OPEN
    MAX_OPEN, MIN_OPEN, calibrated = settings.travel_limits()
    if not calibrated:
        raise SystemExit("Not calibrated.")
    print(f"max_open={MAX_OPEN}  min_open={MIN_OPEN}  grip={GRIP_CURRENT} raw")
    print("Gripper only — THE ARM WILL NOT MOVE.\n")

    _G = settings.connect()
    results = {}
    try:
        _G.set_profile_velocity(VELOCITY)
        _G.set_goal_position(MAX_OPEN)
        set_current(OPEN_CURRENT)
        _G.enable_torque()
        settle(10.0)

        for s in ("A", "B", "C"):
            input(f"\n[{s}] Place the module between the fingers, then Enter...")
            held = grip()
            print(f"      gripped at {held}")
            end = try_open(s)
            freed = end >= MAX_OPEN - 200
            results[s] = (held, end, freed)
            print(f"      after open: {end}   -> {'RELEASED' if freed else 'STILL STUCK'}")
            if not freed:
                input("      free it by hand if needed, then Enter...")
                _G.set_goal_position(MAX_OPEN)
                set_current(OPEN_CURRENT)
                settle(10.0)

        print("\n---- result ----")
        names = {"A": "relax 0.3s", "B": "relax 2.0s", "C": "torque-off"}
        for s, (held, end, freed) in results.items():
            print(f"  {s} {names[s]:<12} grip {held} -> {end}   "
                  f"{'RELEASED' if freed else 'stuck'}")
        winners = [s for s, r in results.items() if r[2]]
        print(f"\n  Use strategy: {winners[0] if winners else 'NONE — revert to eraser padding'}")
    finally:
        try:
            _G.set_goal_position(MAX_OPEN)
            set_current(OPEN_CURRENT)
            settle(10.0)
            _G.disable_torque()
        finally:
            _G.close()
        print("\nTorque released, port closed.")


if __name__ == "__main__":
    main()
