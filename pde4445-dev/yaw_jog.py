"""
Interactive YAW calibration — find how the gripper's jaw angle relates to the
wrist U (your J6). Model:  U = SIGN * yaw + YAW_OFFSET  (mod 180, gripper is
symmetric). We find SIGN and YAW_OFFSET by eye:

  * hover the gripper (pointing down) over a chosen part
  * rotate U until the OPEN jaws sit ACROSS the part's SHORT width (ready to grasp)
  * record that part's pick-list yaw and the aligned U
  * repeat for 2-3 parts with clearly different yaws (pick rectangular ones:
    lcd / esp / arduino — not the round ultrasonic)

PRE-REQ: arm already at the gripper-down ready pose; Main.prg running; pick_list.json
current. Uses the same tool-Z offset as pickplace.

Commands:  <number> = set U (deg) and re-hover    +N / -N = nudge U by N deg
           p<i>     = switch to part index i       q = quit
Run:  python pde4445-dev/yaw_jog.py
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from pickerbot_lib import sender

PICK_LIST        = os.path.join(HERE, "pick_list.json")
APPROACH_MM      = 80.0
TOOL_Z_OFFSET_MM = 70.0    # keep in sync with pickplace_nogripper.py

picks = json.load(open(PICK_LIST))
print("\nParts in pick_list:")
for i, p in enumerate(picks):
    print(f"  [{i}] {p['label']:10s} yaw={p['yaw']:7.1f}   @({p['x']:.0f}, {p['y']:.0f}, {p['z']:.0f})")

sender.connect()


def hover(idx, u):
    p = picks[idx]
    z = p["z"] + APPROACH_MM - TOOL_Z_OFFSET_MM
    reply = sender.epsonJump(p["x"], p["y"], z, u)
    if "OK" not in str(reply).upper():
        print(f"  !! arm non-OK: {reply!r}   (out of range? try U +/- 180)")
    else:
        print(f"  part[{idx}] {p['label']}   pick-list yaw={p['yaw']:.1f}   ->  U = {u:.1f}")


idx = int(input("\nStart on which part index? "))
u = picks[idx]["yaw"]          # first guess: SIGN=+1, OFFSET=0
hover(idx, u)
print("\nRotate U until the OPEN jaws straddle the part's SHORT width.")
print("When it looks right to grasp, note the part's yaw and this U.\n")

while True:
    cmd = input(f"[part {idx}  U={u:.1f}] > ").strip()
    if not cmd:
        continue
    if cmd == "q":
        break
    if cmd.startswith("p"):
        try:
            idx = int(cmd[1:]); u = picks[idx]["yaw"]; hover(idx, u)
        except (ValueError, IndexError):
            print("  usage: p<index>, e.g. p3")
        continue
    try:
        u = (u + float(cmd)) if cmd[0] in "+-" else float(cmd)
    except ValueError:
        print("  enter a number, +N, -N, p<i>, or q")
        continue
    hover(idx, u)

sender.disconnect()
print("\nGive me your aligned (pick-list yaw, U) pairs for 2-3 parts and I'll solve SIGN + YAW_OFFSET.")
