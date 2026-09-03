"""
PER-CLASS GRIP — runs on LAPTOP 2 only. No arm code, no network, no CV.

Idea: each module class has a THICKNESS SIGNATURE. Closing on it under a fixed
current ceiling always stalls the fingers at about the same position, and that
position is measurably different from closing on air. So the gripper can answer
"am I holding an arduino?" from position alone, per class.

  learn : measure a class - empty stalls vs loaded stalls - and save the profile
  grip  : open, close on the class, VERIFY the stall matches the profile, then
          HOLD while you jog the arm by hand, then let go on your command

Profiles live in grip_profiles.json next to this script, so each class you teach
is remembered.

USAGE (copy into Aman's repo, next to Scripts/, and run there):
    python grip_class.py learn arduino            # teach a new class (uses 80 raw)
    python grip_class.py learn esp --current 80   # another class
    python grip_class.py grip  arduino            # grasp + hold + release
    python grip_class.py list                     # show what has been taught

SAFETY
  * NO ARM CODE RUNS HERE. The arm only moves when YOU jog it by hand.
  * Fingers must be clear during the EMPTY measurements.
  * Torque is released in the finally block, including on Ctrl-C.
"""
import sys, json, time, threading, argparse
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
import gripper_settings as settings                   # noqa: E402

PROFILES         = Path(__file__).resolve().parent / "grip_profiles.json"
DEFAULT_CURRENT  = 80      # measured: reaches and holds an Arduino on this rig
OPEN_CURRENT     = 100     # for OPENING only - never a grip force
VELOCITY         = 40
SETTLE_TIMEOUT_S = 3.0
REPEATS          = 3
BAND_MARGIN      = 40      # ticks of slack allowed around a learned loaded band
SLIP_MARGIN      = 60      # drop this far below the loaded band while holding => slip


# ------------------------------- servo helpers -------------------------------
class Grip:
    def __init__(self):
        self.g = None
        self.max_open = self.min_open = None

    def __enter__(self):
        mx, mn, calibrated = settings.travel_limits()
        if not calibrated:
            raise SystemExit("Not calibrated - run Aman's calibration wizard first.")
        self.max_open, self.min_open = mx, mn
        self.g = settings.connect()                   # opens port, Operating Mode 5
        self.g.set_profile_velocity(VELOCITY)
        self.g.set_goal_position(mx)                  # target FIRST, then the ceiling
        self.g.set_goal_current(OPEN_CURRENT)
        self.g.enable_torque()
        self.settle()
        print(f"Gripper ready.  max_open={mx}  min_open={mn}\n")
        return self

    def __exit__(self, *exc):
        try:
            self.open()
            self.g.disable_torque()
        finally:
            self.g.close()                            # closes the SERIAL PORT
        print("\nTorque released, port closed.")

    def settle(self):
        last, t0 = None, time.time()
        while time.time() - t0 < SETTLE_TIMEOUT_S:
            p = self.g.read_present_position()
            if last is not None and abs(p - last) <= 2:
                break
            last = p
            time.sleep(0.06)
        return self.g.read_present_position()

    def open(self):
        # Re-target BEFORE raising the ceiling, or a high ceiling with a closed
        # goal slams the fingers shut.
        self.g.set_goal_position(self.max_open)
        self.g.set_goal_current(OPEN_CURRENT)
        return self.settle()

    def close(self, current):
        self.g.set_goal_current(current)
        self.g.set_goal_position(self.min_open)
        return self.settle()

    def position(self):
        return self.g.read_present_position()

    def current(self):
        return self.g.read_present_current()


# --------------------------------- profiles ---------------------------------
def load_profiles():
    return json.loads(PROFILES.read_text()) if PROFILES.exists() else {}


def save_profile(name, prof):
    p = load_profiles()
    p[name] = prof
    PROFILES.write_text(json.dumps(p, indent=2))
    print(f"\nSaved profile '{name}' -> {PROFILES}")


# ----------------------------------- learn -----------------------------------
def cmd_learn(args):
    current = args.current
    with Grip() as gr:
        print(f"=== LEARNING '{args.cls}' at {current} raw (~{current*2.69:.0f} mA) ===\n")

        print("A. EMPTY closes - where do the fingers stall on air?")
        empty = []
        for i in range(1, REPEATS + 1):
            gr.open()
            input(f"  [{i}/{REPEATS}] Fingers CLEAR - confirm, then Enter...")
            p = gr.close(current)
            empty.append(p)
            print(f"      stalled {p}")

        print(f"\nB. LOADED closes - place the {args.cls} THE SAME WAY each time.")
        print("   (grip across the short width, on bare board if you can)")
        loaded = []
        for i in range(1, REPEATS + 1):
            gr.open()
            input(f"  [{i}/{REPEATS}] Place the {args.cls}, then Enter...")
            p = gr.close(current)
            loaded.append(p)
            print(f"      stalled {p}")
            input("      Tug it, then Enter to reopen...")
        gr.open()

        e_hi, l_lo = max(empty), min(loaded)
        gap = l_lo - e_hi
        print(f"\n  empty  {empty}   (max {e_hi})")
        print(f"  loaded {loaded}  (min {l_lo})")
        print(f"  separation: {gap} ticks")

        if gap < 25:
            print("\n  TOO CLOSE to tell apart. This class is too thin to verify by position")
            print("  at this current. Try a firmer current, or accept that this class needs")
            print("  visual confirmation instead. Not saving.")
            return

        prof = {
            "current": current,
            "empty_max": e_hi,
            "loaded_min": l_lo,
            "loaded_max": max(loaded),
            "threshold": (e_hi + l_lo) // 2,
            "learned_at": time.strftime("%Y-%m-%d %H:%M"),
        }
        print(f"\n  -> holding when stall position is above {prof['threshold']} ticks")
        save_profile(args.cls, prof)


# ----------------------------------- grip -----------------------------------
def monitor(gr, prof, stop):
    """Watch the grip while the operator jogs the arm. Reports any slip."""
    lo = prof["loaded_min"] - SLIP_MARGIN
    worst = None
    while not stop.is_set():
        try:
            p = gr.position()
        except Exception:
            break
        worst = p if worst is None else min(worst, p)
        if p < lo:
            print(f"\n  !! SLIP: position fell to {p} (expected >= {lo}). The module is moving.")
            return
        time.sleep(0.3)
    if worst is not None:
        print(f"  (lowest position seen while holding: {worst})")


def cmd_grip(args):
    prof = load_profiles().get(args.cls)
    if not prof:
        raise SystemExit(f"No profile for '{args.cls}'. Run:  python grip_class.py learn {args.cls}")

    with Grip() as gr:
        print(f"=== GRIP '{args.cls}'  (current {prof['current']} raw, "
              f"expect stall {prof['loaded_min']}-{prof['loaded_max']}) ===\n")
        gr.open()
        input(f"  Place the {args.cls} between the fingers, then Enter to close...")
        pos = gr.close(prof["current"])
        cur = gr.current()
        print(f"\n  stalled at {pos} ticks   current={cur} raw (~{abs(cur)*2.69:.0f} mA)")

        if pos <= prof["threshold"]:
            print(f"  -> NOTHING GRASPED (below threshold {prof['threshold']}). Not holding.")
            gr.open()
            return
        if pos > prof["loaded_max"] + BAND_MARGIN:
            print(f"  -> GRASPED SOMETHING THICKER than a {args.cls} "
                  f"(expected up to {prof['loaded_max']}).")
            print("     Wrong class, bad placement, or it caught on a component.")
            if not input("     Hold it anyway? [y/N] ").strip().lower().startswith("y"):
                gr.open()
                return
        else:
            print(f"  -> HOLDING a {args.cls}. Signature matches.")

        print("\n  The gripper will KEEP HOLDING now.")
        print("  Go to laptop 1 and JOG THE ARM BY HAND to check the grip is real.")
        print("  Watching for slip in the meantime...\n")

        stop = threading.Event()
        t = threading.Thread(target=monitor, args=(gr, prof, stop), daemon=True)
        t.start()
        try:
            input("  Press Enter when you are done jogging, to LET GO...")
        finally:
            stop.set(); t.join(timeout=1.0)

        print(f"  final position before release: {gr.position()}")
        gr.open()
        print("  released.")


def cmd_list(args):
    p = load_profiles()
    if not p:
        print("No profiles taught yet.")
        return
    print(f"{'class':12} {'current':>7} {'threshold':>9} {'loaded band':>16}   learned")
    for k, v in p.items():
        band = f"{v['loaded_min']}-{v['loaded_max']}"
        print(f"{k:12} {v['current']:>7} {v['threshold']:>9} {band:>16}   {v.get('learned_at','')}")


def main():
    ap = argparse.ArgumentParser(description="Per-class gripper grasping (no arm, no CV).")
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("learn", help="measure and save a class's grip signature")
    a.add_argument("cls")
    a.add_argument("--current", type=int, default=DEFAULT_CURRENT)
    a.set_defaults(func=cmd_learn)

    b = sub.add_parser("grip", help="grasp a known class, hold while you jog, release")
    b.add_argument("cls")
    b.set_defaults(func=cmd_grip)

    c = sub.add_parser("list", help="show taught classes")
    c.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
