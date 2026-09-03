"""
ARM LINK SMOKE TEST — one safe move, hand on the e-stop. Connects to the EPSON,
JUMPs to the capture pose (a small +Z hop, then back to essentially where it
already is) and prints the reply. Confirms the Python->controller link, that
motion happens, and that the wrist orientation (V,W) is preserved.

RUN THIS BEFORE the pick-place loop.

Pre-reqs:
  * SPEL+ receiver running on the controller ("Robot ready, listening to network")
  * config.json:  epson_ip = the controller's IP,  epson_port = 2001
  * arm already jogged to (or near) the capture pose, low speed on the controller
Run:  python pde4445-dev/arm_check.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from pickerbot_lib import sender

PLACE   = (-153.0267, 786.0599, 672.9122)   # capture pose x,y,z
U_FIXED = 90.6427                            # capture pose U (V,W kept from Here)

print("SAFETY: hand on the e-stop. Keep controller speed low.")
input("Press Enter to connect and send ONE move...")
sender.connect()
try:
    print("JUMP to the capture pose (small hop up, returns to where it is)...")
    reply = sender.epsonJump(PLACE[0], PLACE[1], PLACE[2], U_FIXED)
    print("reply:", reply)
finally:
    sender.disconnect()
print("If the arm hopped and returned with 'OK', the link is good.")
