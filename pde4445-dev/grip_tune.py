"""
GRIP-STRENGTH TUNING — runs on LAPTOP 2 (the one wired to the U2D2).
No robot, no network. Finds the WEAKEST grip strength that reliably holds a
module, so we never squeeze harder than we must.

Method: the gripper is in current-based position control, so close() drives
toward min-open under a current ceiling and STALLS on whatever is between the
fingers. Aman's API reports "ok" (stopped short = holding something) or "miss"
(reached min open = caught nothing). We sweep the strength ceiling upward and
stop at the first level that both reports "ok" AND survives a gentle tug.

COPY THIS FILE into Aman's repo (next to Scripts/) and run it there:
    python grip_tune.py
It finds Config/ automatically by walking up from the script / working dir.

SAFETY
  * The gripper is on the arm, but the ARM NEVER MOVES here - no arm code runs.
  * Keep fingers clear except when the prompt asks you to feed a module.
  * Ctrl-C at any time: torque is released in the finally block.
"""
import sys, time
from pathlib import Path


# ---------- locate Aman's repo (Config/ + Lib/) ----------
def find_repo():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (base, *base.parents):
            if (d / "Config" / "gripper_config.yaml").exists() and (d / "Lib").is_dir():
                return d
    raise SystemExit(
        "Could not find Aman's repo (needs Config/gripper_config.yaml and Lib/).\n"
        "Copy this script into the GripSense repo folder and run it from there."
    )


REPO = find_repo()
sys.path.insert(0, str(REPO / "Lib"))
from dynamixel_gripper import GripperAPI          # noqa: E402

CFG    = REPO / "Config" / "gripper_config.yaml"
TABLE  = REPO / "Config" / "xm430_control_table.yaml"
LIMITS = REPO / "Config" / "gripper_limits.yaml"

# sweep from gentle upward; stop at the first strength that holds
STRENGTHS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
CONFIRM_REPEATS = 3            # re-tests at the chosen strength


def report(api, tag):
    s = api.state()
    try:
        print(f"    {tag}: status={api.status}  pos={s.position_ticks} ticks "
              f"({s.position_normalised:.2f})  current={s.current_ma:.0f} mA")
    except AttributeError:                      # be forgiving about field names
        print(f"    {tag}: status={api.status}  state={s}")


def main():
    print(f"Using repo: {REPO}")
    api = GripperAPI.from_config(str(CFG), str(TABLE), str(LIMITS))
    with api:
        if not api.calibrated:
            raise SystemExit("NOT CALIBRATED — run Aman's calibration wizard first:\n"
                             "    python Scripts\\gripper_benchmark.py   -> 'Calibrate travel limits...'")
        api.enable(True)
        print("Torque ON. The ARM WILL NOT MOVE — this is gripper-only.\n")

        chosen = None
        try:
            for s in STRENGTHS:
                print(f"\n=== grip strength {s:.2f} ===")
                api.set_grip_strength(s)
                api.open()
                input("  Place a module between the open fingers (hold it lightly), then Enter...")
                verdict = api.close()
                report(api, "closed")
                print(f"    verdict = {verdict!r}")

                if verdict != "ok":
                    print("    -> 'miss': fingers closed all the way, nothing held. Try firmer.")
                    api.open()
                    continue

                ans = input("  Gently TUG the module. Did it HOLD without marking it? [y/n/s=stop] ").strip().lower()
                api.open()
                if ans.startswith("s"):
                    break
                if ans.startswith("y"):
                    chosen = s
                    print(f"    -> holds at {s:.2f}")
                    break
                print("    -> slipped or too weak; increasing.")

            if chosen is None:
                print("\nNo strength in the sweep held the module.")
                print("Either widen STRENGTHS, or check the fingers/padding with Aman.")
                return

            # --- confirm repeatability at the chosen strength ---
            print(f"\n=== confirming {chosen:.2f} over {CONFIRM_REPEATS} repeats ===")
            api.set_grip_strength(chosen)
            held = 0
            for i in range(1, CONFIRM_REPEATS + 1):
                api.open()
                input(f"  [{i}/{CONFIRM_REPEATS}] Place the module, then Enter...")
                verdict = api.close()
                report(api, f"repeat {i}")
                ok = verdict == "ok" and input("    held on a tug? [y/n] ").strip().lower().startswith("y")
                held += bool(ok)
                api.open()

            print(f"\nRESULT: grip strength {chosen:.2f} held {held}/{CONFIRM_REPEATS} times.")
            if held == CONFIRM_REPEATS:
                print(f"Use GRIP_STRENGTH = {chosen:.2f} for the pick. Tell me this number.")
            else:
                print("Not repeatable yet — try the next strength up, or add padding.")
        finally:
            api.open()
            api.enable(False)          # release torque; fingers back-drivable
            print("\nTorque released.")


if __name__ == "__main__":
    main()
