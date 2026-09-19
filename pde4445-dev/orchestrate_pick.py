"""
Pick orchestrator — the conductor that runs a full pick sequence.

Same PC drives both systems:
  * the arm over the SPEL+ TCP link   (pickerbot_lib.sender)
  * the gripper over Aman's Dynamixel (gripper_api)

Per part (topmost-first, from pose_seg's pick_list.json):
  open -> JUMP above the part -> GO down to straddle it -> close under a
  force ceiling -> holding? -> lift + GO to drop + open ;
  if nothing grasped, open and skip (logged).

Set DRY_RUN = True to rehearse the whole sequence with mocks (no hardware) —
run it now to check the logic; flip to False at the bench.
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # for pickerbot_lib
sys.path.insert(0, HERE)                                        # for gripper_api

# ---- config / tunables (finalise the *-marked ones at the lab) ----
PICK_LIST    = os.path.join(HERE, "pick_list.json")
DRY_RUN      = True
YAW_OFFSET   = 0.0                       # * gripper-mount + grip-across-short offset (find at lab)
APPROACH_MM  = 60.0                      # * hover this far above the pick Z before descending
GRIP_DZ_MM   = 8.0                       # * descend this far below the top surface to straddle the part
GRIP_CURRENT = 110                       # grip-force ceiling
DROP         = (150.0, 470.0, 400.0, 0.0)   # * a safe (x,y,z,u) drop pose


class ArmFault(Exception):
    """Raised when the EPSON reply to a move is not OK — we stop for safety."""


def _ok(reply):
    """Raise if the arm didn't confirm the move (matches epsonPickAll's OK check)."""
    if "OK" not in str(reply).upper():
        raise ArmFault(f"arm returned non-OK: {reply!r}")


def make_arm():
    if DRY_RUN:
        class Arm:
            def connect(self):        print("[dry] arm connected")
            def jump(self, x,y,z,u):  print(f"[dry] JUMP -> ({x:.1f}, {y:.1f}, {z:.1f})  u={u:.1f}"); return "OK"
            def go(self, x,y,z,u):    print(f"[dry] GO   -> ({x:.1f}, {y:.1f}, {z:.1f})  u={u:.1f}"); return "OK"
            def standby(self):        print("[dry] STANDBY")
            def disconnect(self):     print("[dry] arm disconnected")
        return Arm()
    from pickerbot_lib import sender
    class Arm:
        def connect(self):        sender.connect()
        def jump(self, x,y,z,u):  sender.epsonJump(x, y, z, u)
        def go(self, x,y,z,u):    sender.epsonGo(x, y, z, u)
        def standby(self):        sender.epsonStandby()
        def disconnect(self):     sender.disconnect()
    return Arm()


def make_gripper():
    if DRY_RUN:
        from gripper_api import MockGripper
        return MockGripper()
    if os.environ.get("GRIPPER_HOST"):          # gripper wired to another laptop
        from net_gripper import NetGripper
        return NetGripper()
    from gripper_api import Gripper             # gripper wired to this laptop
    return Gripper()


def main():
    picks = json.load(open(PICK_LIST))
    print(f"{'DRY RUN — ' if DRY_RUN else ''}{len(picks)} parts to clear (topmost-first).")

    arm, grip = make_arm(), make_gripper()
    arm.connect(); grip.connect()

    cleared = skipped = 0
    try:
        for i, p in enumerate(picks, 1):
            x, y, z, u = p["x"], p["y"], p["z"], p["yaw"] + YAW_OFFSET
            print(f"\n--- pick {i}: {p['label']} @ ({x:.1f}, {y:.1f}, {z:.1f})  yaw {u:.1f} ---")
            grip.open()
            _ok(arm.jump(x, y, z + APPROACH_MM, u))   # hover above, fingers open
            _ok(arm.go(x, y, z - GRIP_DZ_MM, u))      # descend to straddle the part
            grip.close(GRIP_CURRENT)
            if grip.holding():
                _ok(arm.go(x, y, z + APPROACH_MM, u)) # lift clear
                _ok(arm.go(*DROP))                    # move to drop zone
                grip.open()                           # release
                cleared += 1
                print("  -> picked & dropped")
            else:
                grip.open()
                skipped += 1
                print("  -> nothing grasped — skip & log")
    except ArmFault as e:
        print(f"\n  !! {e}\n  !! ABORTING run for safety.")
        grip.open()

    arm.standby()
    grip.release(); arm.disconnect()
    print(f"\nDone.  cleared={cleared}  skipped={skipped}")


if __name__ == "__main__":
    main()
