"""
GRIPPER NETWORK SERVER — runs on LAPTOP 2 (the one wired to the U2D2).
Exposes the gripper over TCP so the pick orchestrator on LAPTOP 1 (arm + vision)
can drive it. One client at a time, line-delimited JSON.

WHY THE RAW DRIVER RATHER THAN GripperAPI
  * GripperAPI's normalised grip strength maps 0.0 -> current.min = 100 raw, so
    the gentlest grip it can command is ~2 kg of squeeze. We need ~80 raw.
  * Its miss detection asks "did the fingers reach min_open?". Under this rig's
    linkage friction, empty fingers stall around 1197 ticks and never get near
    min_open (-53), so that check reports "ok" even on air.
  * So we command goal current directly and decide holding with a MEASURED
    threshold (see HOLD_THRESHOLD_TICKS), from grip_verify.py.

MEASURED ON THIS RIG (Arduino, 80 raw, PLA fingers, no padding):
    empty  closes stall at 1178-1198 ticks
    loaded closes stall at 1262-1265 ticks
  -> threshold 1230. Above it = something is between the fingers.
  RE-MEASURE with grip_verify.py after any finger swap or recalibration, and
  note that a THINNER module stalls lower and may fall below the threshold.

COPY INTO Aman's repo (next to Scripts/) and run there:
    python gripper_server.py            # listens on 0.0.0.0:5005

Protocol - one JSON object per line:
  {"cmd":"open"}                  -> {"ok":true,"result":{"position":...}}
  {"cmd":"close","current":80}    -> {"ok":true,"result":{"position":...,"holding":true}}
  {"cmd":"holding"}               -> {"ok":true,"result":true/false}
  {"cmd":"state"}                 -> {"ok":true,"result":{"position":...,"current":...,"holding":...}}
  {"cmd":"release"}               -> ends this client (servo stays connected)
  errors                          -> {"ok":false,"error":"..."}
"""
import json, socket, sys, time
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
import gripper_settings as settings                  # noqa: E402

HOST, PORT           = "0.0.0.0", 5005
GRIP_CURRENT         = 80      # measured minimum that reaches and holds the module
OPEN_CURRENT         = 100     # for OPENING only - not a grip force
HOLD_THRESHOLD_TICKS = 1230    # above this after a close => holding (from grip_verify.py)
VELOCITY             = 40
SETTLE_TIMEOUT_S     = 3.0


class Grip:
    def __init__(self):
        self.g = None
        self.max_open = self.min_open = None

    def connect(self):
        mx, mn, calibrated = settings.travel_limits()
        if not calibrated:
            raise SystemExit("Not calibrated - run the calibration wizard first.")
        self.max_open, self.min_open = mx, mn
        self.g = settings.connect()                  # opens port, Operating Mode 5
        self.g.set_profile_velocity(VELOCITY)
        self.g.set_goal_position(mx)                 # target FIRST, then the ceiling
        self.g.set_goal_current(OPEN_CURRENT)
        self.g.enable_torque()
        self._settle()
        print(f"Gripper ready.  max_open={mx}  min_open={mn}  "
              f"grip={GRIP_CURRENT} raw  hold_threshold={HOLD_THRESHOLD_TICKS}")

    def _settle(self):
        last, t0 = None, time.time()
        while time.time() - t0 < SETTLE_TIMEOUT_S:
            p = self.g.read_present_position()
            if last is not None and abs(p - last) <= 2:
                break
            last = p
            time.sleep(0.06)
        return self.g.read_present_position()

    def open(self):
        # Re-target BEFORE raising the ceiling: high current with a closed goal
        # would slam the fingers shut.
        self.g.set_goal_position(self.max_open)
        self.g.set_goal_current(OPEN_CURRENT)
        return {"position": self._settle()}

    def close(self, current=GRIP_CURRENT):
        self.g.set_goal_current(current)
        self.g.set_goal_position(self.min_open)
        pos = self._settle()
        return {"position": pos, "holding": pos > HOLD_THRESHOLD_TICKS}

    def holding(self):
        return self.g.read_present_position() > HOLD_THRESHOLD_TICKS

    def state(self):
        return {"position": self.g.read_present_position(),
                "current": self.g.read_present_current(),
                "holding": self.holding()}

    def shutdown(self):
        if self.g:
            try:
                self.open()
                self.g.disable_torque()
            finally:
                self.g.close()                       # closes the SERIAL PORT
            self.g = None


def handle(grip, msg):
    cmd = msg.get("cmd")
    if   cmd == "open":    return grip.open()
    elif cmd == "close":   return grip.close(int(msg.get("current", GRIP_CURRENT)))
    elif cmd == "holding": return grip.holding()
    elif cmd == "state":   return grip.state()
    raise ValueError(f"unknown cmd: {cmd!r}")


def main():
    grip = Grip()
    grip.connect()
    print(f"serving on port {PORT}. Ctrl-C to stop.")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT)); srv.listen(1)
    try:
        while True:
            conn, addr = srv.accept()
            print("client connected:", addr)
            f = conn.makefile("rwb", buffering=0)
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        if msg.get("cmd") == "release":
                            grip.open()
                            f.write(b'{"ok":true,"result":null}\n')
                            break
                        res = handle(grip, msg)
                        f.write((json.dumps({"ok": True, "result": res}) + "\n").encode())
                    except Exception as e:
                        f.write((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
            except Exception as e:
                print("connection error:", e)
            finally:
                conn.close(); print("client disconnected")
    except KeyboardInterrupt:
        pass
    finally:
        grip.shutdown(); srv.close()
        print("gripper released, server stopped.")


if __name__ == "__main__":
    main()
