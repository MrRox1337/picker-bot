"""
Gripper-only bench check — NO arm motion. Validates open/close and holding()
on a scrap object, and confirms the signed-position fix.

Run:  python pde4445-dev/gripper_check.py    (from the repo root)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for gripper_api
from gripper_api import Gripper

g = Gripper()
g.connect()

input("\n[1] Clear the fingers, then press Enter to OPEN...")
g.open()
print(f"    opened -> pos={g.present_position()}  (should be near max_open 3103)")

input("\n[2] Put a SCRAP object between the fingers, then press Enter to CLOSE...")
g.close()
print(f"    closed on object -> pos={g.present_position()}  current={g.present_current()}  holding={g.holding()}")
print("    EXPECT: holding = True")

input("\n[3] Press Enter to OPEN and then test an EMPTY close...")
g.open()
input("    Remove the object (fingers empty), then press Enter to close on nothing...")
g.close()
print(f"    empty close -> pos={g.present_position()}  current={g.present_current()}  holding={g.holding()}")
print("    EXPECT: holding = False")

g.open()
g.release()
print("\nDone. If holding is True-with-object and False-when-empty, the gripper API is good.")
print("If not, tell me the two 'pos=' numbers and I'll tune HOLD_MARGIN_TICKS.")
