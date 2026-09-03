"""
NETWORK GRIPPER CLIENT — runs on LAPTOP 1 (arm + vision + orchestrator).
Forwards every call over TCP to gripper_server.py on LAPTOP 2, which is the
machine wired to the U2D2. Drop-in for the pick orchestrator.

Point it at the gripper laptop:
    set GRIPPER_HOST=172.20.10.9        (Windows)   and optionally GRIPPER_PORT
or pass the host on the command line for the self-test below.

Self-test (no arm motion; exercises open / close / holding over the network):
    python pde4445-dev/net_gripper.py 172.20.10.9
"""
import os, json, socket

GRIP_CURRENT = 80          # keep in sync with gripper_server.py


class NetGripper:
    def __init__(self, host=None, port=None):
        self.host = host or os.environ.get("GRIPPER_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("GRIPPER_PORT", 5005))
        self.sock = None
        self.f = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=15)
        self.f = self.sock.makefile("rwb", buffering=0)
        print(f"NetGripper connected -> {self.host}:{self.port}")

    def _rpc(self, cmd, **kw):
        self.f.write((json.dumps({"cmd": cmd, **kw}) + "\n").encode())
        line = self.f.readline()
        if not line:
            raise RuntimeError("gripper server closed the connection")
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(f"gripper server error: {r.get('error')}")
        return r.get("result")

    def open(self):
        return self._rpc("open")

    def close(self, current=GRIP_CURRENT):
        """Close under a current ceiling. Returns {'position':..., 'holding':bool}."""
        return self._rpc("close", current=current)

    def holding(self):
        return bool(self._rpc("holding"))

    def state(self):
        return self._rpc("state")

    def release(self):
        try:
            if self.f:
                self._rpc("release")
        finally:
            if self.sock:
                self.sock.close()
            self.sock = self.f = None


if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRIPPER_HOST")
    g = NetGripper(host)
    g.connect()

    input("\n[1] Fingers CLEAR. Enter to OPEN...")
    print("    ", g.open())

    input("\n[2] Fingers still CLEAR. Enter to close on NOTHING...")
    r = g.close()
    print("    ", r, "  EXPECT holding=False")

    g.open()
    input("\n[3] Place the module between the fingers, then Enter to CLOSE...")
    r = g.close()
    print("    ", r, "  EXPECT holding=True")

    input("\n    Tug it, then Enter to release...")
    g.open()
    g.release()
    print("\nDone. If [2] was False and [3] was True, the bridge is good.")
