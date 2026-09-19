"""
NO-GRIPPER pick-place test — GRIPPER-DOWN orientation.

The camera and gripper sit ~90 deg apart on the J6 flange. Scanning happens at the
CAPTURE pose (camera looking down); PICKING happens with the GRIPPER pointing down.
Tool 1 (the +Z 162.6 mm TCP you calibrated with the hanging string) is the
gripper's working point, so bringing Tool 1 to a part's (x,y,z) with the gripper
pointing down puts the gripper on the part — no separate tool, just the right wrist
orientation.

The receiver inherits the wrist V,W from wherever the arm currently is, and every
move here is a GO (which keeps orientation). So if you START the arm in the
gripper-down pose, the gripper stays pointing down for the whole run.

WORKFLOW at the bench:
  1. SCAN first, arm at the CAPTURE pose: record .db3 -> run pose_seg -> pick_list.json
  2. STOP Main.prg. Jog the arm to a GRIPPER-DOWN ready pose: gripper pointing
     straight down, a safe height above the workspace, roughly over the parts.
  3. START Main.prg (Home is disabled, so it stays put and listens).
  4. Run this script.

Per part (topmost-first): GO hover above -> GO down to a safe clearance (no
contact) -> pause while you remove the part BY HAND -> GO lift.
Run:  python pde4445-dev/pickplace_nogripper.py
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # for pickerbot_lib
from pickerbot_lib import sender

PICK_LIST   = os.path.join(HERE, "pick_list.json")
APPROACH_MM      = 80.0   # gripper-tip hover above the part (also lift height between parts)
REACH_MM         = 40.0   # gripper-tip clearance above the part at the bottom (NO contact)
TOOL_Z_OFFSET_MM = 70.0   # the calibrated Tool-1 (string) point sits this far BELOW the gripper
                          # fingertip, so the tip was hovering ~10 cm too high. Subtract it to
                          # lower the tip. Still too high? increase it. Dips too low? decrease it.
# Yaw mapping comes from yaw_calib.py:  U = SIGN*yaw + OFFSET  (mod 180).
# Measured 2 Sep 2026: sign +1, offset -88.8 deg, RMS 1.5 deg over 5 orientations
# (i.e. essentially the 90 deg between the long axis and the grip axis).
# Near-square parts still carry an inherent +/-90 long-axis ambiguity.
_CALIB = os.path.join(HERE, "yaw_calib.json")
if os.path.exists(_CALIB):
    _c = json.load(open(_CALIB))
    YAW_SIGN, YAW_OFFSET_DEG = _c["sign"], _c["offset_deg"]
else:
    YAW_SIGN, YAW_OFFSET_DEG = +1, -88.8


class ArmFault(Exception):
    pass


def _ok(reply):
    if "OK" not in str(reply).upper():
        raise ArmFault(f"arm returned non-OK: {reply!r}")


def main():
    picks = json.load(open(PICK_LIST))
    print(f"{len(picks)} parts to visit (topmost-first), GRIPPER-DOWN.")
    print("PRE-REQ: arm already jogged to the gripper-down ready pose (gripper")
    print("pointing straight down, safe height above the workspace).")
    print("SAFETY: hand on the e-stop. Every move keeps the gripper pointing down.\n")
    input("Press Enter to connect and begin...")

    sender.connect()
    done = 0
    try:
        for i, p in enumerate(picks, 1):
            x, y, z = p["x"], p["y"], p["z"]
            u = YAW_SIGN * p["yaw"] + YAW_OFFSET_DEG
            u = ((u + 180) % 360) - 180           # wrap into (-180, 180]; gripper is symmetric
            print(f"\n--- part {i}/{len(picks)}: {p['label']} @ ({x:.1f}, {y:.1f}, {z:.1f})  yaw={p['yaw']:.1f} -> U={u:.1f} ---")
            hover_z = z + APPROACH_MM - TOOL_Z_OFFSET_MM
            reach_z = z + REACH_MM    - TOOL_Z_OFFSET_MM
            _ok(sender.epsonJump(x, y, hover_z, u))   # arch over + set yaw (gripper down)
            _ok(sender.epsonGo(x, y, reach_z))        # straight down, keep orientation
            input("   over the part, jaws aligned. Remove it BY HAND, then press Enter...")
            _ok(sender.epsonGo(x, y, hover_z))        # lift straight up
            done += 1
            print(f"   done ({done}/{len(picks)}).")
    except ArmFault as e:
        print(f"\n  !! {e}\n  !! ABORTING for safety.")
    finally:
        sender.disconnect()
    print(f"\nFinished. Completed {done} of {len(picks)}. Arm left hovering gripper-down.")


if __name__ == "__main__":
    main()
