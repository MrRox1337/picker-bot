"""
DESK REGRESSION TEST — no arm, no gripper, no camera, no model.

Standing rule for this project: no first-time code path runs on a lab day. Every
branch added on 9 Sep (bounded freeze, active probe, three descent modes) is
exercised here against a simulated servo and a synthetic scene, so the lab session
only meets code that has already run.

    python pde4445-dev/test_descent.py

Exit status is non-zero if anything fails, so it can be run before every session.
"""
import os, sys, json, time, tempfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pick_one                                                  # noqa: E402
import clear_scene as cs                                         # noqa: E402

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILED.append(name)


def check_true(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILED.append(name)


# ============================================================ fake servo
class FakeServo:
    """Minimal XM430 stand-in.

    Position marches toward the goal at a realistic rate (profile velocity 40 is
    about 600 ticks/s on this servo) and is ARRESTED at `obstacle` when something
    is between the fingers. That single property is what the whole verification
    argument rests on, so the fake models it and nothing else.
    """
    RATE = 600.0                     # ticks per second

    def __init__(self, pos, floor, obstacle=None):
        self.pos, self.goal = float(pos), float(pos)
        self.floor = float(floor)              # where EMPTY fingers stall (friction)
        self.obstacle = obstacle               # where an object stops them, or None
        self.cur = 0
        self.t = time.time()
        self.torque = False

    def _limit(self):
        return self.floor if self.obstacle is None else max(self.floor, float(self.obstacle))

    def read_present_position(self):
        now = time.time()
        dt, self.t = now - self.t, now
        target = self.goal
        if target < self.pos:                  # closing: an object can stop us early
            target = max(target, self._limit())
        travel = self.RATE * dt
        if self.pos < target:
            self.pos = min(target, self.pos + travel)
        elif self.pos > target:
            self.pos = max(target, self.pos - travel)
        return int(round(self.pos))

    def set_goal_position(self, p):  self.goal = float(p)
    def set_goal_current(self, c):   self.cur = int(c)
    def read_present_current(self):  return self.cur
    def enable_torque(self):         self.torque = True
    def disable_torque(self):        self.torque = False
    def set_profile_velocity(self, v): pass
    def close(self):                 pass


def make_gripper(obstacle):
    """A pick_one.Gripper wired to a fake servo, skipping connect()'s hardware.

    Numbers are the 12 Sep rig: calibration 3541/655, empty stall 708 at 100 raw,
    an arduino arresting the fingers at 1859.
    """
    g = pick_one.Gripper(threshold=790)
    g.max_open, g.min_open = 3541, 655
    g.g = FakeServo(pos=3541, floor=708, obstacle=obstacle)
    return g


# ============================================================ 1. hold policy
def test_bounded_freeze():
    print("\n[1] bounded freeze — goal parks INSIDE the stall, not at it, not at min_open")
    g = make_gripper(obstacle=1859)
    pos, holding = g.close(90)
    check_true("close reaches the object", abs(pos - 1859) <= 3, f"(pos={pos})")
    check("holding verdict", holding, True)
    # The old bug: goal left at min_open => ~1500 ticks of error for the whole carry.
    # The regression it caused: goal parked AT pos => zero error, no clamp, no signal.
    check_true("goal is not min_open", g._hold_goal != g.min_open, f"(goal={g._hold_goal})")
    check_true("goal is not the stall itself", g._hold_goal != pos, f"(goal={g._hold_goal})")
    check("freeze bias", pos - g._hold_goal, pick_one.FREEZE_BIAS_TICKS)


# ============================================================ 2. active probe
def test_probe_holding():
    print("\n[2] probe on a HELD part — fingers arrested, travel ~0")
    g = make_gripper(obstacle=1859)
    g.close(90)
    still, travel, note = g.probe("arduino")
    check("verdict", still, True)
    check_true("travel is small", travel <= pick_one.PROBE_MOVE_TICKS, f"(travel={travel})")
    check_true("goal re-parked after the probe",
               g._hold_goal < g.g.read_present_position(), f"(goal={g._hold_goal})")


def test_probe_dropped():
    print("\n[3] probe after the part is GONE — this is the case that used to log 'placed'")
    g = make_gripper(obstacle=1859)
    g.close(90)
    g.g.obstacle = None                        # the part falls out during the lift
    passive = g.g.read_present_position()
    still, travel, note = g.probe("arduino")
    check("verdict", still, False)
    check_true("probe travelled far", travel > pick_one.PROBE_MOVE_TICKS, f"(travel={travel})")
    check_true("passive read alone would NOT have caught it",
               abs(passive - 1859) <= pick_one.FREEZE_BIAS_TICKS + 5,
               f"(passive read was {passive}, i.e. still 'in range')")


def test_probe_thin_class():
    print("\n[4] probe on an UNCHARACTERISED class — must not invent a failure")
    g = make_gripper(obstacle=None)            # esp: fingers reach the friction floor
    g.close(90)
    still, travel, note = g.probe("mystery_part")
    check("verdict", still, True)
    check_true("and says why", "vision" in note, f"(note={note!r})")


# ============================================================ 5. descent modes
class A:                                        # stand-in for parsed args
    def __init__(self, **kw):
        self.descent_mode = "fixed"
        self.grasp_dz = 25.0
        self.grasp_dz_raised = 32.0
        self.raised_mm = 12.0
        self.finger_clear = 6.0
        self.tip_offset = 0.0                   # tests set it explicitly where it matters
        self.dz_by_class = {}
        self.z_ref = "top"
        self.__dict__.update(kw)


FLAT    = {"z": 250.0, "z_land": 244.0, "label": "arduino"}   # jaws land on the bench
STACKED = {"z": 265.0, "z_land": 244.0, "label": "arduino"}   # on top of another board
# An esp lies under a jaw but BELOW the part's own top, so the grasp is still
# possible - just shallower. A z_land at or ABOVE the part top is a different
# case entirely: nothing can be grasped through it, and wanted() refuses outright
# rather than clamping. See jaw_obstructed().
BLOCKED = {"z": 250.0, "z_land": 246.0, "label": "arduino"}
LEGACY  = {"z": 250.0, "label": "arduino"}                    # pick list with no z_land


def test_descent_modes():
    print("\n[5] descent modes")
    h, _ = cs.grasp_height(FLAT, A(descent_mode="fixed"), 244.0)
    check("fixed, flat part", h, 275.0)
    h, _ = cs.grasp_height(STACKED, A(descent_mode="fixed"), 244.0)
    check("fixed, stacked part (same dz - the baseline's blind spot)", h, 290.0)

    h, why = cs.grasp_height(FLAT, A(descent_mode="two-regime"), 244.0)
    check("two-regime, flat -> normal dz", h, 275.0)
    check_true("and calls it flat", "flat" in why, f"({why})")
    h, why = cs.grasp_height(STACKED, A(descent_mode="two-regime"), 244.0)
    check("two-regime, stacked -> raised dz", h, 297.0)
    check_true("and calls it raised", "RAISED" in why, f"({why})")

    # tip_offset=0 means "the grasp-height reference IS the jaw tips". Under that
    # assumption a bench 31 mm below the grasp height never threatens anything.
    h, why = cs.grasp_height(FLAT, A(descent_mode="clearance"), 244.0)
    check("clearance, jaws well clear of the bench -> unchanged", h, 275.0)

    # With the real offset the jaws hang 26.7 mm lower, and the same scene is
    # suddenly marginal - which is exactly what "the fingers rubbed the bench at
    # dz=25" reported from the lab. The rule reproduces the observation.
    real = A(descent_mode="clearance", tip_offset=26.7)
    h, why = cs.grasp_height(FLAT, real, 244.0)
    check_true("with the real tip offset, dz=25 over bare bench is already too deep",
               "CLAMPED" in why, f"({why})")
    check_true("and it lifts the grasp only slightly", 276.0 < h < 278.0, f"(h={h})")

    h, why = cs.grasp_height(BLOCKED, real, 244.0)
    check("clearance, esp under a jaw -> clamped 6mm above it", round(h, 1), 278.7)
    check_true("and says it clamped", "CLAMPED" in why, f"({why})")
    check_true("the clamp forces a SHALLOWER grasp than the fixed baseline",
               h > BLOCKED["z"] + 25.0, f"(h={h})")

    # And the case the clamp CANNOT rescue: something sitting on top of the part.
    over = {"z": 246.0, "z_med": 246.0, "z_land": 264.0, "label": "arduino"}
    a = A(descent_mode="fixed", max_occlusion=0.35, only=None, skip=None,
          min_conf=0.0, want_by_class=None)
    ok, why = cs.wanted(over, a)
    check_true("a part with something ON it is refused, not attempted",
               (not ok) and "jaws would land" in why, f"({why})")

    h, why = cs.grasp_height(LEGACY, real, 244.0)
    check("clearance on a legacy pick list falls back to fixed", h, 275.0)
    check_true("LOUDLY", "UNAVAILABLE" in why, f"({why})")


def test_per_class_dz():
    print("\n[5b] per-class grasp depth")
    a = A(descent_mode="fixed", dz_by_class={"lcd": 32.0})
    h, why = cs.grasp_height({"z": 250.0, "label": "lcd"}, a, 244.0)
    check("an overridden class uses its own dz", h, 282.0)
    h, why = cs.grasp_height({"z": 250.0, "label": "arduino"}, a, 244.0)
    check("every other class keeps the global dz", h, 275.0)


def test_z_reference():
    print("\n[5d] z reference — depth bleed across a stack")
    # The 12 Sep L1 scene: an esp resting on an arduino. Aligned depth bleeds over
    # the boundary, so the arduino's NEAREST band reports the esp's height and the
    # two parts tie at 261.
    scene = [{"label": "ultrasonic", "z": 265.5, "z_med": 262.0},
             {"label": "esp",        "z": 261.3, "z_med": 259.5},
             {"label": "arduino",    "z": 261.0, "z_med": 250.8},
             {"label": "lcd",        "z": 254.2, "z_med": 249.0}]

    top = [p["label"] for p in cs.order_picks(scene, "topmost", z_ref="top")]
    med = [p["label"] for p in cs.order_picks(scene, "topmost", z_ref="med")]
    check_true("with z_ref=top the buried arduino ties with the esp on top of it",
               abs(scene[1]["z"] - scene[2]["z"]) < 1, "")
    check("with z_ref=med the esp is correctly ordered above the arduino",
          med.index("esp") < med.index("arduino"), True)
    # The sequence can come out the same either way — sort stability happens to
    # break the tie correctly here. What matters is the MARGIN: under z_ref=top
    # the two parts are separated by less than the 0.39 mm depth noise, so the
    # order is decided by chance. Under z_ref=med they are 8.7 mm apart.
    gap_top = abs(scene[1]["z"] - scene[2]["z"])
    gap_med = abs(scene[1]["z_med"] - scene[2]["z_med"])
    check_true("z_ref=top separates them by less than the depth noise",
               gap_top < 1.0, f"({gap_top:.1f} mm, sd_z is 0.39 mm)")
    check_true("z_ref=med separates them decisively", gap_med > 5.0,
               f"({gap_med:.1f} mm)")

    a_top = A(descent_mode="fixed", grasp_dz=35.0, z_ref="top")
    a_med = A(descent_mode="fixed", grasp_dz=35.0, z_ref="med")
    h1, _ = cs.grasp_height(scene[2], a_top, 243.6)
    h2, _ = cs.grasp_height(scene[2], a_med, 243.6)
    check_true("and the arduino is aimed ~10mm lower, at its own body",
               9 < h1 - h2 < 11, f"({h1:.1f} -> {h2:.1f})")

    # a part with no z_med (an old pick list) must not crash or silently shift
    check("a pick list without z_med falls back to z",
          cs.part_top({"z": 250.0}, "med"), 250.0)


def test_clearance():
    print("\n[5c] clearance by position — the metric that reported 3/3 remaining")
    init = [{"label": "arduino", "x": -113.6, "y": 805.8},
            {"label": "lcd", "x": 59.1, "y": 707.5},
            {"label": "ultrasonic", "x": 19.0, "y": 701.7}]
    far = (250.0, 600.0)

    # everything picked and stacked at a place pose well away from the bench
    r = [{"label": l, "x": 250.0 + d, "y": 600.0} for l, d in
         (("arduino", -5), ("lcd", 5), ("ultrasonic", 0))]
    kept, dropped = cs.drop_placed(r, far)
    gone, left, moved = cs.clearance(init, kept)
    check("placed parts are excluded from the scan", len(dropped), 3)
    check("and all three count as cleared", len(gone), 3)

    # the lcd was never picked
    r = [{"label": "lcd", "x": 59.0, "y": 708.0},
         {"label": "arduino", "x": 245.0, "y": 600.0},
         {"label": "ultrasonic", "x": 250.0, "y": 600.0}]
    kept, dropped = cs.drop_placed(r, far)
    gone, left, moved = cs.clearance(init, kept)
    check("a part still on the bench is NOT counted as cleared", len(gone), 2)
    check("and is named", [p["label"] for p in left], ["lcd"])

    # the lcd was shoved aside by a neighbour's pick rather than picked
    r = [{"label": "lcd", "x": 139.0, "y": 707.0},
         {"label": "arduino", "x": 245.0, "y": 600.0},
         {"label": "ultrasonic", "x": 250.0, "y": 600.0}]
    kept, _ = cs.drop_placed(r, far)
    gone, left, moved = cs.clearance(init, kept)
    check_true("a disturbed part is flagged, not silently counted",
               [p["label"] for p, _d in moved] == ["lcd"], str(moved))

    # a place pose sitting ON the picking area erases real parts - the reason the
    # fix is to move the place pose, not to widen the radius
    near = (0.0, 750.0)
    r = [{"label": "ultrasonic", "x": 19.0, "y": 701.7}]
    kept, dropped = cs.drop_placed(r, near)
    check_true("a place pose inside the bench area swallows real parts",
               len(dropped) == 1, "(this is why PlacePoint must be moved away)")


def test_scene_floor():
    print("\n[6] scene floor")
    z, src = cs.scene_floor([FLAT, STACKED], None)
    check("prefers the lowest landing measurement", z, 244.0)
    z, src = cs.scene_floor([LEGACY], None)
    check("falls back to the lowest part top", z, 250.0)
    check_true("and admits it is crude", "crude" in src, f"({src})")
    z, src = cs.scene_floor([FLAT], 240.0)
    check("an explicit --table-z always wins", z, 240.0)


# ============================================================ 7. full dry runs
SCENE = [
    {"label": "arduino", "conf": 0.91, "x": -50.0, "y": 600.0, "z": 265.0, "z_med": 263.0,
     "z_land": 244.0, "width_mm": 53.0, "yaw": -60.0, "aspect": 1.6, "u": 600, "v": 350},
    # z_land close under the part, not above it: clearance should CLAMP this one.
    # A z_land above the part top is refused outright by jaw_obstructed and never
    # reaches the clamp - that case is covered separately in [5].
    {"label": "arduino", "conf": 0.88, "x": 20.0, "y": 610.0, "z": 250.0, "z_med": 249.0,
     "z_land": 246.0, "width_mm": 53.0, "yaw": 12.0, "aspect": 1.6, "u": 700, "v": 380},
    {"label": "esp", "conf": 0.80, "x": 60.0, "y": 590.0, "z": 246.0, "z_med": 245.5,
     "z_land": 244.0, "width_mm": 28.0, "yaw": 92.0, "aspect": 2.3, "u": 760, "v": 340},
]


def run_dry(extra):
    scene = os.path.join(tempfile.gettempdir(), "picker_test_scene.json")
    json.dump(SCENE, open(scene, "w"))
    runs = os.path.join(tempfile.gettempdir(), "picker_test_runs")
    env = dict(os.environ, PICKER_PICK_LIST=scene, PICKER_RUNS_DIR=runs)
    cmd = [sys.executable, os.path.join(HERE, "clear_scene.py"),
           "--dry-run", "--grasp-dz", "25", "--rescan", "none"] + extra
    p = subprocess.run(cmd, input="\n", capture_output=True, text=True, env=env, timeout=180)
    return p.stdout + p.stderr


def _aperture_for(out, label):
    """Pull the aperture printed for the first pick of a given class."""
    cur = None
    for line in out.splitlines():
        if line.lstrip().startswith("-- [") and f" {label} " in line:
            cur = label
        elif cur == label and "aperture:" in line:
            return int(line.split("aperture:")[1].split()[0])
    return -1


def test_full_dry_runs():
    print("\n[7] full dry runs through clear_scene.py")
    for mode in ("fixed", "two-regime", "clearance"):
        out = run_dry(["--descent-mode", mode])
        check_true(f"{mode}: completes", "Done. attempts=" in out,
                   "" if "Done. attempts=" in out else out[-700:])
        check_true(f"{mode}: reports its descent decision", "descent:" in out, "")
    out = run_dry(["--descent-mode", "clearance"])
    check_true("clearance clamps the blocked part in a real run", "CLAMPED" in out, "")

    check_true("the adaptive aperture fires by default", "aperture:" in out, "")
    check_true("and a thin part opens narrower than a wide one",
               _aperture_for(out, "esp") < _aperture_for(out, "arduino"),
               f"(esp {_aperture_for(out, 'esp')} vs arduino {_aperture_for(out, 'arduino')})")
    full = run_dry(["--aperture", "full"])
    check_true("--aperture full restores the old full-open behaviour",
               "aperture:" not in full, "")

    # Since the 12 Sep move to 100 raw the esp ARRESTS the fingers (1262 against
    # an empty 721), so both verification policies now succeed on it. That is the
    # finding, not a broken test: the divergence that existed at 90 raw came from
    # fingers that never reached the part. What still has to hold is the plumbing
    # — that the policy is selectable and reported per pick.
    strict = run_dry(["--verify", "proprioceptive"])
    check_true("strict policy now grips the esp too", "no_grasp" not in strict, "")
    check_true("and says which policy settled it", "verify: proprioceptive" in strict, "")
    lenient = run_dry(["--verify", "vision"])
    check_true("vision policy is reachable", "verify: vision" in lenient, "")
    check_true("running vision-verified with no re-scan is called out",
               "unevidenced" in lenient, "")

    # --only arduino: the drop simulation is about the PROBE, which only speaks for
    # classes with a signature. Leaving the esp in would legitimately produce a
    # vision-verified 'placed' and mask the thing being tested.
    for where, outcome in (("lift", "lost_on_lift"), ("place", "lost_in_transit")):
        out = run_dry(["--descent-mode", "fixed", "--only", "arduino", "--sim-drop", where])
        check_true(f"a drop at the {where} is caught", outcome in out, "")
        check_true(f"and {where} drops are NOT counted as placed",
                   "placed=0" in out, out[-400:])


# ============================================================ 8. pose_seg geometry
# The descent modes are only as good as z_land. This exercises part_pose on a
# SYNTHETIC frame - a known rectangle at a known depth over a known bench - so the
# landing measurement is checked against arithmetic rather than against a guess
# about what the camera saw.
def test_part_pose():
    print("\n[8] pose_seg.part_pose — landing measurement on a synthetic frame")
    try:
        import numpy as np
        import pose_seg
    except ImportError as e:
        print(f"  SKIP  numpy/cv2 not available here ({e})")
        return

    H, W = 720, 1280
    intr = (900.0, 900.0, W / 2.0, H / 2.0)
    # camera looks straight down: nearer (smaller metric depth) => higher base Z
    R = np.diag([1.0, 1.0, -1.0])
    t = np.array([0.0, 0.0, 700.0])

    BENCH_M, PART_M = 0.400, 0.380          # bench 300mm, part top 320mm in base frame
    depth = np.full((H, W), BENCH_M, np.float32)
    mask = np.zeros((H, W), np.uint8)
    # a 240 x 100 px board: long axis along +x, so the JAWS close along y
    y0, y1, x0, x1 = 310, 410, 520, 760
    mask[y0:y1, x0:x1] = 255
    depth[y0:y1, x0:x1] = PART_M

    p = pose_seg.part_pose(mask, depth, intr, R, t)
    check_true("returns a pose", p is not None)
    check("part top z", p["z"], 320.0)
    check_true("long axis is the 240px side (yaw ~0)", abs(p["yaw"]) < 2.0, f"(yaw={p['yaw']})")
    check_true("aspect ~2.4", 2.2 < p["aspect"] < 2.6, f"(aspect={p['aspect']})")
    check("jaws land on the bare bench", p["z_land"], 300.0)
    # 100 px across the short side at 0.38 m with fx=900 is 42 mm. Getting the LONG
    # side here would mean the jaws were being sized to the wrong axis entirely.
    check_true("width is the SHORT side (42mm), not the long one (101mm)",
               40.0 < p["width_mm"] < 44.0, f"(width_mm={p['width_mm']})")
    check("flat board: z_med equals z_top", p["z_med"], 320.0)

    # Now slide an esp under one jaw, 10 mm proud of the bench. The long axis runs
    # along x, so the jaws close along Y and land BEYOND THE LONG EDGES - putting
    # the obstruction past the short end (a natural mistake) tests nothing.
    d2 = depth.copy()
    d2[y1 + 5:y1 + 55, x0:x1] = 0.390        # base Z 310, under the +y jaw
    p2 = pose_seg.part_pose(mask, d2, intr, R, t)
    check_true("an obstruction under a jaw raises z_land",
               p2["z_land"] > p["z_land"], f"({p['z_land']} -> {p2['z_land']})")
    check_true("and it is the OBSTRUCTION's height, not the bench's",
               abs(p2["z_land"] - 310.0) < 2.0, f"(z_land={p2['z_land']})")
    check_true("the OTHER jaw still sees bench, and the worse side wins",
               p2["z_land"] > 305.0, f"(z_land={p2['z_land']})")

    # a tilted board: z_top tracks the raised edge, z_med tracks the body
    d3 = depth.copy()
    d3[y0:y1, x0:x1] = np.linspace(0.365, 0.395, x1 - x0, dtype=np.float32)
    p3 = pose_seg.part_pose(mask, d3, intr, R, t)
    check_true("tilted: z_top follows the RAISED edge",
               p3["z"] > p3["z_med"], f"(z_top={p3['z']} z_med={p3['z_med']})")
    check_true("and the gap is the tilt error that broke grasps at dz=35",
               6.0 < p3["z"] - p3["z_med"] < 16.0, f"(gap={p3['z'] - p3['z_med']:.1f}mm)")


# ============================================================ 9. aperture
def test_aperture():
    print("\n[9] width-adaptive aperture")
    g = make_gripper(obstacle=1859)

    g.aperture = None
    check("uncalibrated returns None, so the caller falls back to full open",
          g.ticks_for_width(53.0), None)

    g.aperture = {"slope_ticks_per_mm": 38.0, "intercept_ticks": 700.0, "rms_mm": 1.0}
    wide = g.ticks_for_width(53.0)           # arduino
    narrow = g.ticks_for_width(28.0)         # esp
    check_true("a narrow part gets a narrower aperture", narrow < wide,
               f"(esp {narrow} < arduino {wide})")
    check_true("and both are well inside a full open", wide < g.max_open,
               f"(arduino {wide} vs max_open {g.max_open})")
    # 53 + 16 = 69mm -> 38*69 + 700 = 3322
    check("arduino aperture is part width plus 2x8mm clearance", wide, 3322)
    check_true("an absurdly wide part is clamped to the mechanism",
               g.ticks_for_width(500.0) == g.max_open, f"({g.ticks_for_width(500.0)})")
    check_true("a zero-width detection cannot command a closed jaw",
               g.ticks_for_width(0.0) > g.min_open, f"({g.ticks_for_width(0.0)})")

    # Width check: vision says how wide the part is, the servo confirms it.
    g.aperture = {"slope_ticks_per_mm": 25.983, "intercept_ticks": 758.4, "rms_mm": 0.12}
    check("predicts the arduino stall from its width",
          g.expected_stall(53.4), 1878)
    ok, note = g.classify(1859, 53.4, "arduino")
    check_true("a correct grasp passes quietly", ok and note == "", f"({note})")
    ok, note = g.classify(700, 53.4, "arduino")
    check_true("an empty close is caught by the threshold, not the width",
               (not ok) and "nothing between" in note, f"({note})")
    ok, note = g.classify(2400, 53.4, "arduino")
    check_true("two parts at once: held, but flagged as too wide",
               ok and "WIDTH MISMATCH" in note, f"({note})")
    ok, note = g.classify(1100, 53.4, "arduino")
    check_true("an edge grasp: held, but flagged as too narrow",
               ok and "WIDTH MISMATCH" in note, f"({note})")
    g.aperture = None
    ok, note = g.classify(1859, 53.4, "arduino")
    check_true("with no calibration the width check stays silent", ok and note == "",
               f"({note})")

    # A calibration taken under a different travel range is worse than none:
    # every tick means something else. It must be refused, not scaled.
    import json as _j, tempfile as _t
    p = os.path.join(_t.gettempdir(), "aperture.json")
    _j.dump({"slope_ticks_per_mm": 38.0, "intercept_ticks": 700.0,
             "calib_max_open": 9999}, open(p, "w"))
    g2 = make_gripper(obstacle=1859)
    old = pick_one.APERTURE_FILE
    try:
        pick_one.APERTURE_FILE = p
        g2._load_aperture()
        check("a stale calibration is refused, not used", g2.aperture, None)
    finally:
        pick_one.APERTURE_FILE = old


# ============================================================ 10. occlusion
def test_occlusion():
    print("\n[10] occlusion score, pickability gate and nudge direction")
    try:
        import numpy as np
        import pose_seg
    except ImportError as e:
        print(f"  SKIP  numpy/cv2 not available here ({e})")
        return

    H, W = 720, 1280
    intr = (900.0, 900.0, W / 2.0, H / 2.0)
    R = np.diag([1.0, 1.0, -1.0])
    t = np.array([0.0, 0.0, 700.0])
    depth = np.full((H, W), 0.400, np.float32)

    def rect(x0, y0, x1, y1, z_m):
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = 255
        depth[y0:y1, x0:x1] = z_m
        return m

    # A: alone in open space.  B: half covered by C.  C: resting on top of B.
    # Masks are DISJOINT, as instance segmentation produces them — B's mask is
    # only the part of B still visible, which is why its outline runs right along
    # C's edge. Overlapping masks would be a test artefact, not a real scene.
    mA = rect(150, 150, 290, 250, 0.390)
    mB = rect(600, 300, 680, 400, 0.390)
    mC = rect(680, 300, 820, 400, 0.370)      # 20 mm higher, hard against B
    masks = [mA, mB, mC]
    picks = []
    for m in masks:
        p = pose_seg.part_pose(m, depth, intr, R, t)
        p["label"] = "arduino"
        picks.append(p)
    pose_seg.occlusion_pass(picks, masks, depth, intr, R, t)
    A, B, C = picks

    check("an isolated part has zero crowding", A["crowding"], 0.0)
    check("and zero occlusion", A["occlusion"], 0.0)
    check_true("the part UNDERNEATH is scored as occluded", B["occlusion"] > 0.1,
               f"(B occlusion {B['occlusion']})")
    check("and names what is on top of it", B["under"], ["arduino"])
    check("the part ON TOP is not occluded", C["occlusion"], 0.0)
    check_true("but the top part IS crowded — touching is not the same as buried",
               C["crowding"] > 0.1, f"(C crowding {C['crowding']})")

    ok, why = pose_seg.pickability(A, 0.35)
    check("an isolated part is pickable", ok, True)
    ok, why = pose_seg.pickability(B, 0.05)
    check_true("a buried part is refused with a reason", (not ok) and "occluded" in why,
               f"({why})")
    ok, _ = pose_seg.pickability(B, 1.0)
    check("threshold 1.0 attempts everything (the ablation baseline)", ok, True)
    ok, _ = pose_seg.pickability({"label": "x"}, 0.35)
    check("a pick with no score is not blocked", ok, True)

    check_true("the buried part gets a nudge direction", B["nudge"] is not None)
    # C is to the RIGHT of B (centroid x 770 vs 670), so B is pushed LEFT in the
    # image. Base x = camera x here, so dx_mm must be negative.
    check_true("and it points away from what is on top of it",
               B["nudge"]["dx_mm"] < 0,
               f"(dx {B['nudge']['dx_mm']}, dy {B['nudge']['dy_mm']})")
    check_true("with a free distance attached", B["nudge"]["free_mm"] > 0,
               f"({B['nudge']['free_mm']} mm)")

    # Pinned from both sides: the vectors should largely cancel, which is the
    # honest answer - there is no good direction.
    dep2 = np.full((H, W), 0.400, np.float32)

    def rect2(x0, y0, x1, y1, z_m):
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = 255
        dep2[y0:y1, x0:x1] = z_m
        return m
    mid  = rect2(600, 300, 740, 400, 0.390)
    left = rect2(520, 310, 610, 390, 0.370)
    right= rect2(730, 310, 820, 390, 0.370)
    ms2 = [mid, left, right]
    pk2 = []
    for m in ms2:
        p = pose_seg.part_pose(m, dep2, intr, R, t)
        p["label"] = "arduino"
        pk2.append(p)
    pose_seg.occlusion_pass(pk2, ms2, dep2, intr, R, t)
    pinned = pk2[0]
    check_true("a part pinned on both sides scores high occlusion",
               pinned["occlusion"] > 0.1, f"({pinned['occlusion']})")
    sideways = abs(pinned["nudge"]["dx_mm"]) if pinned["nudge"] else 0
    check_true("and is NOT confidently pushed sideways into a neighbour",
               pinned["nudge"] is None or sideways < 40,
               f"(dx {sideways:.0f} mm)")


def main():
    test_bounded_freeze()
    test_probe_holding()
    test_probe_dropped()
    test_probe_thin_class()
    test_descent_modes()
    test_per_class_dz()
    test_z_reference()
    test_clearance()
    test_scene_floor()
    test_part_pose()
    test_aperture()
    test_occlusion()
    test_full_dry_runs()
    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed — these paths have now run at least once off-robot")


if __name__ == "__main__":
    main()
