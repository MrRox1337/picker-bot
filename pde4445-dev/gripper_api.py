"""
Gripper API for pick-and-place — a thin wrapper over Aman's GripSense
Dynamixel driver (github.com/MrRox1337/GripSense).

Exposes exactly what the pick orchestrator needs:
    connect() / release()
    open()                 fingers to max-open
    close(current)         close under a current (force) ceiling; stalls on the object
    holding()              True if fingers stalled short of fully-closed (object between them)
    present_current() / present_position()   for slip / verification monitoring

Same PC as the arm, so this just imports Aman's modules. His repo + dynamixel_sdk
are only touched inside connect(), so this file imports cleanly (and MockGripper
works) without any of that installed.

Point GRIPSENSE_LIB at his repo's Lib/ folder (or set the env var of the same name).
"""
import os, sys, time

GRIPSENSE_LIB = os.environ.get(
    "GRIPSENSE_LIB",
    r"C:\Users\10463\MB_faseeh\REPO\GripSense-main\GripSense-main\Lib",
)

# ---- tunables (finalise the *-marked ones at the lab) ----
GRIP_CURRENT      = 110     # raw goal-current units (his benchmark range 100–120) -> grip force
CLOSE_VELOCITY    = 60      # profile velocity for the close move
SETTLE_TIMEOUT_S  = 2.0
HOLD_MARGIN_TICKS = 30      # * fingers stalled this far above fully-closed => object held


def _s32(v):
    """Dynamixel positions read back as raw uint32; make them signed (closed pos is negative)."""
    v = int(v)
    return v - 0x100000000 if v > 0x7FFFFFFF else v


class Gripper:
    """Real gripper, backed by Aman's DynamixelGripper."""

    def __init__(self):
        self.dev = None
        self.max_open = None      # ticks (fingers apart)
        self.min_open = None      # ticks (fully closed)

    def connect(self):
        if GRIPSENSE_LIB not in sys.path:
            sys.path.insert(0, GRIPSENSE_LIB)
        import gripper_settings as gs                     # his config/layout module (no hardware)
        from dynamixel_gripper import DynamixelGripper
        self.dev = DynamixelGripper(gs.DEVICE_PORT, gs.BAUDRATE,
                                    gs.PROTOCOL_VERSION, gs.DXL_ID,
                                    str(gs.CONTROL_TABLE_PATH))
        mx, mn, calibrated = gs.travel_limits()
        if not calibrated:
            print("WARNING: gripper not calibrated for these fingers — run his calibration first.")
        self.max_open, self.min_open = _s32(mx), _s32(mn)   # extended mode: closed is negative
        self.dev.set_operating_mode_current_based_position()   # Mode 5: force-limited close
        self.dev.set_profile_velocity(CLOSE_VELOCITY)
        self.dev.enable_torque()
        print(f"Gripper ready.  max_open={self.max_open}  min_open={self.min_open} (ticks)")

    def _pos(self):
        return _s32(self.dev.read_present_position())

    def _settle(self):
        last, t0 = None, time.time()
        while time.time() - t0 < SETTLE_TIMEOUT_S:
            p = self._pos()
            if last is not None and abs(p - last) <= 2:
                return p
            last = p
            time.sleep(0.05)
        return last

    def open(self):
        self.dev.set_goal_current(GRIP_CURRENT)
        self.dev.set_goal_position(self.max_open)
        self._settle()

    def close(self, current=GRIP_CURRENT):
        self.dev.set_goal_current(current)                 # force ceiling
        self.dev.set_goal_position(self.min_open)          # drive to fully-closed; stalls on object
        self._settle()

    def holding(self):
        """True if the fingers stopped short of fully-closed (something is between them)."""
        return (self._pos() - self.min_open) > HOLD_MARGIN_TICKS

    def present_current(self):
        return self.dev.read_present_current()

    def present_position(self):
        return self._pos()

    def release(self):
        if self.dev:
            self.dev.close()                               # disables torque + closes port
            self.dev = None


class MockGripper:
    """Offline stand-in with the same interface (for dry-run orchestration)."""
    def __init__(self):            self._pos = 0
    def connect(self):             print("[mock] gripper connected")
    def open(self):                print("[mock] open");                 self._pos = 100
    def close(self, current=GRIP_CURRENT): print(f"[mock] close @ current {current}"); self._pos = 40
    def holding(self):             print("[mock] holding? -> True");     return True
    def present_current(self):     return 100
    def present_position(self):    return self._pos
    def release(self):             print("[mock] released")
