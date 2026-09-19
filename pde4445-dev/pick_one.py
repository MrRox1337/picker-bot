"""
FIRST REAL PICK — one part, arm + gripper, on the single laptop.

Deliberately NOT autonomous. The descent is stepped by hand the first time,
because the modules lie flat and the fingers must come down alongside the board
WITHOUT hitting the table. Once you know the depth that works it is printed as
GRASP_DZ_MM and later runs can use it directly.

PRE-REQ
  * Main.prg running; arm at the GRIPPER-DOWN ready pose (V,W correct)
  * pick_list.json fresh from the current scene (pose_seg.py)
  * yaw_calib.json present (yaw_calib.py), or pass --yaw to set U by hand
  * gripper powered, calibrated, fingers clear

RUN
    python pde4445-dev/pick_one.py --dry-run          # rehearse, no hardware
    python pde4445-dev/pick_one.py                    # topmost part
    python pde4445-dev/pick_one.py --part 2           # a specific part
    python pde4445-dev/pick_one.py --part 2 --yaw 45  # override U entirely

DESCENT CONTROLS
    Enter  down one step        u  up one step
    g      grip here            a  abort (lift and stop)

SAFETY: hand on the e-stop. Nothing descends without you pressing a key.
"""
import os, sys, json, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

PICK_LIST   = os.path.join(HERE, "pick_list.json")
YAW_CALIB   = os.path.join(HERE, "yaw_calib.json")
PROFILES    = os.path.join(HERE, "grip_profiles.json")

# ---- geometry (mm) ----
APPROACH_MM      = 80.0    # hover above the part before descending
TOOL_Z_OFFSET_MM = 70.0    # calibrated Tool-1 point sits this far below the fingertips
STEP_MM          = 5.0     # descent increment
MAX_DROP_MM      = 60.0    # refuse to descend further than this below the part top
# PLACE MUST BE REACHABLE IN THE GRIPPER-DOWN ORIENTATION.
# Do NOT use the capture pose: that was taught camera-down, and its height
# (Z~673) is unreachable with the wrist rotated gripper-down - the arm faults
# and the controller stops without replying. Keep the place pose at a height
# comparable to the picks (parts sit near Z~250), inside the proven envelope.
# Override at runtime with:  --place x,y,z
PLACE            = (0.0, 550.0, 330.0)
ARM_TIMEOUT_S    = 30.0    # never block forever waiting for an OK that will not come

# ---- gripper ----
# THE OPERATING RANGE IS 100-120 RAW. Aman's gripper_config.yaml declares
# current.min = 100 and current.max = 120, and the travel-limit calibration
# probes at 110. Every run before 12 Sep used 80-90 - BELOW THE FLOOR - and that
# single mistake produced most of the gripper mysteries in this project.
#
# Measured on the rig, 12 Sep, empty fingers commanded to min_open:
#     90 raw  -> stall 1646      (fingers stop with a gap, nothing touches)
#    100 raw  -> 721   jaws in contact
#    110 raw  -> 690   jaws in contact
#    120 raw  -> 671   jaws in contact
# Below ~100 the servo cannot beat the linkage's static friction and the fingers
# park at ~1646 whatever is in front of them. That gap is WIDER than an esp, an
# lcd or an ultrasonic, so those parts were never gripped at all - the fingers
# closed past nothing and stopped short. The earlier "the sponge conforms around
# a thin board and hides it" explanation was wrong: there was no contact to hide.
GRIP_CURRENT     = 100     # the bottom of the usable range; leaves 20 raw for opening
OPEN_CURRENT     = 120     # the servo's ceiling, and what teleop opens cleanly with
OPEN_RETRY_CURRENT = 120   # nothing above 120 exists; the retry is a second attempt
RELAX_CURRENT    = 100     # never command below the configured minimum
# Velocity matters as much as current here. Aman's calibration closes at 60 and
# REOPENS AT 480; our 40 everywhere made opening a slow crawl against friction.
CLOSE_VELOCITY   = 60
OPEN_VELOCITY    = 480
EMPTY_STALL_TICKS = 708    # where the fingers stop on air at 100 raw. Spread: 1 tick.
HOLD_THRESHOLD   = 830     # 12 Sep, 100 raw, calibration 3541/655, new fingers + sponge.
                           # RAISED back up after the L1 run. I had lowered it to 750 to
                           # let an ultrasonic that stalled at 796 count as a grasp. That
                           # was wrong: 796 is 88 ticks above empty, i.e. 3.4 mm of jaw
                           # travel, and nothing 16-20 mm wide was ever in there. The
                           # next run repeated it at 789, logged "placed", and the
                           # verification scan found the part still on the bench.
                           # A threshold has to sit below the lowest REAL grasp, not
                           # below the highest failed one. The lowest hand-measured
                           # loaded stall is the ultrasonic at 878; 830 clears empty by
                           # 122 ticks and sits 48 below that.
                           # Erring high costs a cycle and logs an honest no_grasp.
                           # Erring low writes a false success into the trial data,
                           # which is far more expensive.
MARGINAL_BAND    = 70      # a stall this close above the threshold is worth flagging:
                           # not rejected, but not to be trusted without the re-scan
                           # Empty 707-708 over six closes. Loaded stalls, per class:
                           #   ultrasonic 901 (spread 59)   lcd 1143 (2)
                           #   esp        1237 (4)          arduino 1859 (9)
                           # The binding case is the ULTRASONIC, whose worst close was
                           # 878: two cylindrical cans give the jaws line contact, so the
                           # part can roll and settle further in under load. 790 sits 82
                           # above empty and 88 below that worst close - deliberately
                           # symmetric, because the empty stall is tight (1 tick) while
                           # the ultrasonic's own spread is 59.
                           # NOTE grip_verify suggests a threshold from whatever class is
                           # in the jaws; for the esp it said 974, which would sit ON TOP
                           # of the ultrasonic's mean. Always re-derive across ALL classes.
                           # Superseded history below - all of it was measured at 80-90
                           # raw, i.e. below the servo's usable floor, so the numbers
                           # describe fingers that never reached the object.
_SUPERSEDED_HOLD = 1816    # above this after a close => holding.
                           # 5 Sep 16:30, calibration max_open 3571 / min_open 601.
                           # M2 run: 3/3 arduinos gripped (stalls 1848, 1856, 1881),
                           # carried and released with zero creep on every lift.
                           # The 1942 below was the SAME rig one calibration earlier -
                           # the wizard shifted the whole tick scale ~155 and every real
                           # grasp was then rejected as no_grasp. Superseded notes:
                           # Measured 5 Sep with grip_verify.py at 90 raw, arduino, SPONGE:
                           #   empty  1830, 1829, 1830   (fingers' own closed position)
                           #   loaded 2073, 2059, 2055   (arrested by the board)
                           # 225-tick separation. Note the empty stall is 1829 under EVERY
                           # padding and calibration tried - it is set by linkage friction,
                           # not by min_open.
                           # Sponge is more compliant than the eraser it replaced, so the
                           # loaded stall is lower and more variable (spread 18 vs 0 ticks):
                           # compliance buys grip on thin parts and costs signal sharpness.
                           # RE-MEASURE after any finger, padding or calibration change.
THRESHOLD_CALIB_MAX_OPEN = 3541   # the calibration HOLD_THRESHOLD was measured under.
                                  # If travel_limits() reports a different max_open the
                                  # gripper has been recalibrated and the threshold is void.
SLIP_DROP_TICKS  = 150     # fingers closing this much further during the lift => it escaped
VELOCITY         = 40
SETTLE_TIMEOUT_S = 3.0     # a close stops as soon as it meets something
OPEN_TIMEOUT_S   = 10.0    # an open travels the full span against friction - much slower

# ---- grasp HOLD policy (bounded freeze) ----
# The first fix for the creep-wedge froze goal_position at exactly the achieved
# stall. That cured the wedge but broke verification: with zero position error the
# fingers CANNOT advance, so a part that fell out produced the same reading as one
# still held, and every run logged "placed". Force and observability were traded
# against each other by accident.
#
# The bounded freeze keeps both. Goal sits a SMALL fixed bias inside the stall, so:
#   * the drive still has an error to work against  -> real clamp force
#   * total travel is capped at the bias            -> creep cannot run away
#     (the runaway case was a ~1300-tick error held for a 40 s carry)
FREEZE_BIAS_TICKS = 25     # how far inside the stall the goal is parked

# ---- ACTIVE grasp probe ----
# A bounded freeze still cannot be READ passively: the fingers are near their goal
# whether or not anything is between them. Verification therefore has to be active -
# briefly re-command a close and see whether the fingers are ARRESTED or run free.
# Empty fingers travel hundreds of ticks to their friction floor (~1674-1829 on this
# rig); a held board stops them dead. That gap is the measurement.
PROBE_CURRENT     = GRIP_CURRENT   # must be enough to move EMPTY fingers. 80 raw was
                                   # measured to reach only the friction floor, so the
                                   # probe current tracks the grip current rather than
                                   # being lowered "for safety" - a probe too weak to
                                   # move free fingers always reports "holding".
PROBE_WINDOW_S    = 0.5            # bounded: 0.5 s cannot creep what 40 s did
PROBE_MOVE_TICKS  = 60             # travelled more than this => nothing is in there
PROBE_BUDGET_TICKS = 80            # cumulative probe travel per grasp before we warn:
                                   # each probe re-parks the goal slightly deeper, so
                                   # repeated probes could ratchet the fingers inward

# ---- WIDTH-ADAPTIVE APERTURE ----
# The jaws currently descend FULLY OPEN, so every pick sweeps the entire jaw span
# through the scene on the way down. On 8 Sep that span met an esp lying beside
# the arduino being picked and snapped a PLA finger. Vision already measures the
# part's short-axis width (pose_seg emits width_mm), so the jaws can open only as
# far as that part needs, and the swept footprint shrinks accordingly.
#
# The mapping from millimetres of jaw gap to servo ticks is a property of Aman's
# linkage and is NOT known a priori - run aperture_calib.py once per finger set.
# Until then this falls back to a full open, which is the previous behaviour.
APERTURE_FILE     = os.path.join(HERE, "aperture.json")
# Sponge compression at grip, measured 12 Sep by comparing each class's stall
# position against the free-gap calibration: 9.6 / 10.2 / 10.5 / 11.0 mm for esp,
# lcd, ultrasonic and arduino. Constant to ~1 mm over a 16-53 mm range of parts,
# which is what makes a stall position predictable from a VISION-measured width:
#     expected_stall = slope * (width_mm - SPONGE_COMPRESSION_MM) + intercept
# Predicts all four classes to within 20 ticks (0.8 mm). This replaces per-class
# stall profiles with something that also covers parts never characterised.
SPONGE_COMPRESSION_MM = 10.3
STALL_TOL_TICKS       = 120  # ~4.6 mm. PROVISIONAL, widened from 60 on 12 Sep.
                             # The 60-tick band was fitted to HAND-PLACED parts in
                             # grip_verify, where the width is the true part width.
                             # In situ there are two systematic offsets: the mask
                             # bleeds past the board edge (vision read 55.5 mm for a
                             # 53.4 mm arduino) and the jaws bite inside that edge
                             # (the grip implied 51.6 mm). ~4 mm of bias, so a 2.3 mm
                             # band cried wolf on a perfectly good grasp.
                             # 120 still catches what matters: a second part in the
                             # jaws stalls hundreds of ticks wide, an edge grasp
                             # hundreds narrow. DO NOT tune this from one sample -
                             # every run row logs width_mm and grip_pos_after_close,
                             # so fit the bias offline once there are trials to fit.
JAW_CLEARANCE_MM  = 8.0    # gap left either side of the part, so a small pose error
                           # does not turn the approach into a collision
APERTURE_FLOOR_MM = 10.0   # never plan an aperture narrower than this

# PLAUSIBLE SHORT-AXIS WIDTH PER CLASS, from the parts themselves.
# In clutter the segmentation mask merges with whatever a part touches, and for a
# near-square outline the PCA long/short assignment flips, so the measured width
# can come back as the LONG axis or larger still. Measured 12 Sep: an arduino read
# 78.6 mm (true short axis 53.4, long axis 68.6) because its mask had merged.
#
# That number is not a wider part, it is a broken measurement - and acting on it
# opens the jaws to a 107 mm span, putting a finger 54 mm from the centre, which
# on that scene was exactly where the neighbouring esp was sitting. Clamping to a
# plausible range keeps the finger 8 mm clear instead.
#
# The clamp is deliberately generous: it only catches readings that cannot be the
# part at all, and never narrows the jaws below the real part.
WIDTH_SANITY_MM = {"arduino": (48.0, 60.0),
                   "lcd":     (32.0, 45.0),
                   "esp":     (24.0, 34.0),
                   "ultrasonic": (16.0, 50.0)}   # cans or board, depending on pose

# Classes whose width actually arrests the fingers, i.e. where the servo's own
# reading is evidence. Measured 12 Sep at 100 raw against an empty stall of 708:
#     ultrasonic  901 (spread 59)    lcd  1143 (2)
#     esp        1237 (spread  4)    arduino 1859 (9)
# EVERY class now separates from empty by at least 193 ticks. The 8 Sep conclusion
# that esp and lcd were proprioceptively invisible was an artefact of running at
# 90 raw, below the servo's configured minimum, where the fingers stalled on
# friction at ~1646 and never touched anything thinner than that gap.
#
# The ultrasonic is the weak member and stays the binding case for HOLD_THRESHOLD:
# two cylindrical cans give the jaws line contact, so the part can roll and settle
# further in under load - hence its 59-tick spread against 2-9 for flat-faced parts.
PROPRIOCEPTIVE_CLASSES = {"arduino", "esp", "lcd", "ultrasonic"}


class ArmFault(Exception):
    pass


def _ok(reply):
    if "OK" not in str(reply).upper():
        raise ArmFault(f"arm returned non-OK: {reply!r}")


# --------------------------------- gripper ---------------------------------
def find_gripsense():
    env = os.environ.get("GRIPSENSE_REPO")
    cands = [env] if env else []
    cands += [r"C:\Users\10463\MB_faseeh\REPO\GripSense",
              r"C:\Users\10463\MB_faseeh\REPO\GripSense-main",
              r"C:\Users\10463\MB_faseeh\REPO\GripSense-main\GripSense-main"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "Config", "gripper_config.yaml")):
            return c
    raise SystemExit("Could not find Aman's GripSense repo. Set GRIPSENSE_REPO to it, e.g.\n"
                     '    set GRIPSENSE_REPO=C:\\Users\\10463\\MB_faseeh\\REPO\\GripSense')


class Gripper:
    """Raw-driver gripper: gentle currents + our own measured holding threshold."""

    def __init__(self, threshold=HOLD_THRESHOLD):
        self.g = None
        self.threshold = threshold
        # Set here as well as in connect(), so the width checks degrade to
        # "no opinion" rather than raising when a Gripper is built without
        # hardware — which is how every off-robot test constructs one.
        self.aperture = None
        self.last_note = ""

    def connect(self):
        repo = find_gripsense()
        sys.path.insert(0, os.path.join(repo, "Lib"))
        import gripper_settings as settings
        self.settings = settings
        mx, mn, calibrated = settings.travel_limits()
        if not calibrated:
            raise SystemExit("Gripper not calibrated - run the calibration wizard first.")
        self.max_open, self.min_open = mx, mn
        # A recalibration shifts every tick value, which silently invalidates the
        # measured hold threshold. On 5 Sep a recalibration 44 s before a run moved
        # the scale ~155 ticks and every real grasp was reported as no_grasp.
        if abs(mx - THRESHOLD_CALIB_MAX_OPEN) > 20:
            print(f"  *** WARNING: max_open is {mx}, but HOLD_THRESHOLD={self.threshold} "
                  f"was measured at max_open={THRESHOLD_CALIB_MAX_OPEN}.")
            print(f"  *** The gripper has been RECALIBRATED. Re-run grip_verify.py "
                  f"and pass the new value, or grasps will be misjudged.")
        self.g = settings.connect()
        self.g.set_profile_velocity(VELOCITY)
        self.g.set_goal_position(mx)          # target FIRST, then the ceiling
        self._set_current(OPEN_CURRENT)
        self.g.enable_torque()
        self._settle()
        print(f"  gripper ready: max_open={mx} min_open={mn} threshold={self.threshold}")
        self._load_aperture()

    def _set_current(self, want):
        """Set goal current, backing off if the servo rejects it.

        The XM430 has a Current Limit register that caps Goal Current, and it is
        set below 160 on this servo - writing more raises
        'The data value exceeds the limit value!'. We do not know the cap, so try
        the value we want and step down until one is accepted.
        """
        want = int(want)
        ladder = [want] + [v for v in (200, 180, 160, 150, 140, 130, 120, 110, 100, 90, 80)
                           if v < want]
        last = None
        for c in ladder:
            try:
                self.g.set_goal_current(int(c))
                if c != want:
                    print(f"  (servo capped goal current {want} -> {c})")
                self._current_cap = c
                return int(c)
            except OSError as e:
                last = e
        raise OSError(f"servo rejected every goal current up to {want}: {last}")

    def _settle(self, timeout=SETTLE_TIMEOUT_S):
        """Wait for the fingers to actually stop.

        The naive version compared two reads 60 ms apart - but right after a
        command the servo has not STARTED moving yet, so those two reads match
        and it returned instantly. Cleanup then cut torque mid-move and the
        fingers never opened. So: let the move begin, then require several
        consecutive stable reads.
        """
        t0 = time.time()
        time.sleep(0.20)                       # let the move actually start
        last, stable = None, 0
        while time.time() - t0 < timeout:
            p = self.g.read_present_position()
            if last is not None and abs(p - last) <= 2:
                stable += 1
                if stable >= 3:
                    return p
            else:
                stable = 0
            last = p
            time.sleep(0.06)
        return self.g.read_present_position()

    def _load_aperture(self):
        """Load the width->ticks mapping, if one has been measured for THIS rig."""
        self.aperture = None
        if not os.path.exists(APERTURE_FILE):
            print("  aperture: no calibration — jaws will descend FULLY OPEN. "
                  "Run aperture_calib.py to narrow the approach.")
            return
        a = json.load(open(APERTURE_FILE))
        # Same trap as HOLD_THRESHOLD: a recalibration shifts the whole tick scale,
        # so a mapping measured under a different max_open is silently wrong.
        if abs(a.get("calib_max_open", -1) - self.max_open) > 20:
            print(f"  *** aperture.json was measured at max_open="
                  f"{a.get('calib_max_open')}, but this rig reports {self.max_open}.")
            print(f"  *** IGNORING it — re-run aperture_calib.py. Full open until then.")
            return
        self.aperture = a
        print(f"  aperture: {a['slope_ticks_per_mm']:.1f} ticks/mm, "
              f"fit rms {a.get('rms_mm', '?')} mm")

    def expected_stall(self, width_mm):
        """Where the fingers SHOULD stop on a part of this width. None if uncalibrated."""
        if not self.aperture or width_mm is None:
            return None
        return int(self.aperture["slope_ticks_per_mm"]
                   * (float(width_mm) - SPONGE_COMPRESSION_MM)
                   + self.aperture["intercept_ticks"])

    def classify(self, pos, width_mm=None, label=None):
        """Turn a stall position into a verdict AND a width check.

        Two different questions, and only the first was ever asked before:
          1. is anything between the fingers?      -> pos above HOLD_THRESHOLD
          2. is it the size vision said it was?    -> pos near expected_stall

        The second catches the failures a threshold cannot: a board caught by its
        edge stalls too wide, two parts taken together stall far too wide, and a
        part that is not what the detector labelled it stalls somewhere else
        entirely. Vision supplies the width; the servo independently confirms it.
        """
        if pos <= self.threshold:
            gap = pos - EMPTY_STALL_TICKS
            return False, (f"nothing between the fingers (stalled {pos}, only {gap:+d} "
                           f"ticks from empty — about "
                           f"{gap / 25.983:.1f} mm of jaw travel)")
        if pos < self.threshold + MARGINAL_BAND:
            return True, (f"MARGINAL: stalled {pos}, only {pos - self.threshold} ticks "
                          f"above the threshold. Treat as unverified until the "
                          f"re-scan confirms the part left the bench.")
        width_mm, _ = self.sane_width(width_mm, label)
        want = self.expected_stall(width_mm)
        if want is None:
            return True, ""
        err = pos - want
        if abs(err) > STALL_TOL_TICKS:
            mm = err / self.aperture["slope_ticks_per_mm"]
            return True, (f"WIDTH MISMATCH: stalled {pos}, expected ~{want} for a "
                          f"{width_mm:.0f}mm {label or 'part'} ({mm:+.1f}mm). "
                          f"Edge grasp, two parts at once, or a mislabelled detection.")
        return True, ""

    def sane_width(self, width_mm, label=None):
        """Clamp a vision width into what this class can physically be."""
        if width_mm is None:
            return None, ""
        lo, hi = WIDTH_SANITY_MM.get(label, (0.0, 1e9))
        w = min(max(float(width_mm), lo), hi)
        if abs(w - float(width_mm)) > 0.5:
            return w, (f"width {width_mm:.0f}mm is not possible for a {label} "
                       f"(mask merged, or the PCA axes flipped) - using {w:.0f}mm")
        return w, ""

    def ticks_for_width(self, width_mm, label=None):
        """Jaw opening in ticks for a part of this width. None if uncalibrated."""
        if not self.aperture or width_mm is None:
            return None
        width_mm, note = self.sane_width(width_mm, label)
        if note:
            print(f"   {note}")
        want = max(float(width_mm) + 2 * JAW_CLEARANCE_MM, APERTURE_FLOOR_MM)
        t = self.aperture["slope_ticks_per_mm"] * want + self.aperture["intercept_ticks"]
        # Clamp into the mechanism. Opening WIDER than max_open is impossible;
        # planning NARROWER than the part plus a margin would clip it on descent.
        return int(min(max(t, self.min_open + 50), self.max_open))

    def open(self, to=None):
        """Open, firmly, with one firmer retry.

        `to` is a tick position; omit it for a full open. A partial open is only
        ever used for the APPROACH — releasing a part always opens fully, because
        a part has to fall clear of jaws that are still half shut.

        Opening must overcome BOTH the linkage friction and whatever the grip
        wedged in place (compressed padding, a board pressed into the faces). So
        the open ceiling must exceed the grip ceiling - opening at 100 after
        gripping at 110 leaves nothing to back out with, and the fingers do not
        move at all. Also a long window: this travels the full span, unlike a
        close, which stops as soon as it meets something.
        """
        # DO WHAT TELEOP DOES. On 12 Sep the manual GUI opened these fingers from
        # fully closed to 3537 at 120 raw without difficulty, while this function
        # repeatedly ended at ~1222 - i.e. CLOSED further while commanded to open,
        # stalling at exactly the position an esp grips at. The hardware was never
        # the problem. Three things differed from teleop, and all three are now
        # aligned with Aman's own configuration:
        #   * a relax step that commanded 50 raw, half the configured minimum
        #   * profile velocity 40 for the open; the calibration reopens at 480
        #   * an open current chosen above the servo's 120 ceiling
        # The relax step is gone. It existed to unwedge sponge that had crept
        # under a 1300-tick position error during a 40 s carry - a failure mode
        # belonging to the sub-100 regime we have now left.
        try:
            self.g.set_profile_velocity(OPEN_VELOCITY)
        except OSError:
            pass

        # Drive open, with one retry.
        target = self.max_open if to is None else int(to)
        pos = None
        for attempt, cur in enumerate((OPEN_CURRENT, OPEN_RETRY_CURRENT), 1):
            self.g.set_goal_position(target)
            self._set_current(cur)
            pos = self._settle(OPEN_TIMEOUT_S)
            if pos >= target - 200:
                return pos
            print(f"  open attempt {attempt}: stopped at {pos} (target {target})"
                  f"{' - retrying firmer' if attempt == 1 else ''}")
        # 3. LAST RESORT: cut torque. The servo becomes back-drivable, so the
        #    compressed padding pushes the fingers apart using its own stored
        #    energy - no goal current involved, which sidesteps the servo's hard
        #    120-raw ceiling. Needed because a part that shifts in the sponge
        #    during transport wedges deeper than the grip that created it, and
        #    no open current can then undo it (measured 5 Sep: 2037 -> 1958).
        print("  open stalled - releasing torque to let the padding spring open")
        try:
            self.g.disable_torque()
            time.sleep(0.6)
            print(f"  torque off -> fingers at {self._pos_safe()}")
            self.g.set_goal_position(target)          # target BEFORE torque returns
            self.g.enable_torque()
            self._set_current(OPEN_CURRENT)
            pos = self._settle(OPEN_TIMEOUT_S)
            if pos >= target - 200:
                print(f"  released after torque cycle (at {pos})")
                return pos
        except OSError as e:
            print(f"  torque cycle failed: {e}")

        print(f"  WARNING: fingers still not open (at {pos}). Free the part by hand.")
        return pos

    def _pos_safe(self):
        try:
            return self.g.read_present_position()
        except OSError:
            return "?"

    def _hold(self, pos):
        """Park the goal a BOUNDED bias inside the achieved stall.

        Leaving goal_position at min_open (763) while the fingers sit at ~2100
        leaves a permanent ~1300-tick error, so the servo drives at the FULL
        current ceiling for the whole lift and carry. The sponge is visco-elastic
        and crept ~90 ticks under that sustained load (measured 5 Sep: 2047->1960
        at 90 raw and 2106->2014 at 80 raw - the SAME creep at both currents, i.e.
        duration, not force), tightening the wedge beyond what 120 raw could undo.

        Parking at exactly `pos` fixed that but removed the position error
        entirely, and with it every passive sign of a lost part. The bias is the
        compromise: enough error to keep clamping, too little to creep far.
        """
        target = max(int(pos) - FREEZE_BIAS_TICKS, self.min_open)
        try:
            self.g.set_goal_position(target)
        except OSError:
            pass
        self._hold_goal = target
        return target

    def close(self, current=GRIP_CURRENT, width_mm=None, label=None):
        self._grip_current = int(current)
        self._probe_spent  = 0
        try:
            self.g.set_profile_velocity(CLOSE_VELOCITY)   # slow onto the object
        except OSError:
            pass
        self._set_current(current)
        self.g.set_goal_position(self.min_open)
        pos = self._settle()
        self._hold(pos)
        holding, self.last_note = self.classify(pos, width_mm, label)
        return pos, holding

    def probe(self, label=None):
        """ACTIVE grasp check. Returns (holding, travel_ticks, note).

        Re-commands a close for a bounded window and measures how far the fingers
        get. Arrested => something is between them. Free-running => it is gone.

        This is the only check that survives the bounded freeze, because the
        passive reading is near the goal in BOTH cases. It is deliberately an
        experiment rather than an observation.

        For classes with no thickness signature it returns holding=True with a
        note, so callers never turn "cannot tell" into "failed".
        """
        if label is not None and label not in PROPRIOCEPTIVE_CLASSES:
            return True, 0, f"no proprioceptive signature for '{label}' - vision only"

        before = int(self.g.read_present_position())
        after = None
        try:
            self._set_current(getattr(self, "_grip_current", GRIP_CURRENT))
            self.g.set_goal_position(self.min_open)
            t0 = time.time()
            while time.time() - t0 < PROBE_WINDOW_S:
                time.sleep(0.05)
        finally:
            # Re-park the goal wherever the fingers ACTUALLY ended, ALWAYS, even if
            # the block above threw. Leaving the goal at min_open would recreate the
            # runaway-creep wedge the bounded freeze exists to prevent - and it would
            # do so on a part the arm is still carrying.
            p = self._pos_safe()
            if isinstance(p, int):
                after = p
                self._hold(p)

        if after is None:
            # Could not read the servo. Report "holding" rather than inventing a
            # failure: this is an instrument fault, not evidence about the grasp.
            return True, 0, "probe could not read the servo - result is not evidence"

        travel = before - after
        self._probe_spent = getattr(self, "_probe_spent", 0) + max(travel, 0)
        note = ""
        if self._probe_spent > PROBE_BUDGET_TICKS:
            note = (f"probe budget exceeded ({self._probe_spent} ticks cumulative) - "
                    f"the fingers are ratcheting inward, stop probing this grasp")
        return travel <= PROBE_MOVE_TICKS, travel, note

    def position(self):
        return self.g.read_present_position()

    def release(self):
        if self.g:
            try:
                self.open()
                self.g.disable_torque()
            finally:
                self.g.close()
            self.g = None


class MockGripper:
    def connect(self):            print("  [dry] gripper connected")
    def open(self, to=None):      print(f"  [dry] open{'' if to is None else f' to {to}'}"); return 3096
    def ticks_for_width(self, w): return None
    def close(self, current=GRIP_CURRENT):
        print(f"  [dry] close @ {current}"); return 1262, True
    def probe(self, label=None):  print("  [dry] probe"); return True, 0, ""
    def position(self):           return 1262
    def release(self):            print("  [dry] gripper released")


# ----------------------------------- arm -----------------------------------
class DryArm:
    def connect(self):            print("  [dry] arm connected")
    def jump(self, x, y, z, u):   print(f"  [dry] JUMP ({x:.1f},{y:.1f},{z:.1f}) u={u:.1f}"); return "OK"
    def go(self, x, y, z, u=None):print(f"  [dry] GO   ({x:.1f},{y:.1f},{z:.1f})"); return "OK"
    def disconnect(self):         print("  [dry] arm disconnected")


class RealArm:
    def connect(self):
        from pickerbot_lib import sender
        self.s = sender
        sender.connect()
        # If the controller faults it stops WITHOUT replying, and a plain recv
        # would block forever (Ctrl-C cannot interrupt it on Windows). Time out
        # instead, so we can say what happened and release the part.
        if getattr(sender, "clientSocket", None):
            sender.clientSocket.settimeout(ARM_TIMEOUT_S)
    def jump(self, x, y, z, u):   return self.s.epsonJump(x, y, z, u)
    def go(self, x, y, z, u=None):return self.s.epsonGo(x, y, z) if u is None else self.s.epsonGo(x, y, z, u)
    def disconnect(self):         self.s.disconnect()


# ---------------------------------- yaw ----------------------------------
def resolve_yaw(part, override):
    if override is not None:
        return override, "manual override"
    if os.path.exists(YAW_CALIB):
        c = json.load(open(YAW_CALIB))
        u = c["sign"] * part["yaw"] + c["offset_deg"]
        return ((u + 180) % 360) - 180, f"yaw_calib.json (rms {c.get('rms_deg')} deg)"
    raise SystemExit("No yaw_calib.json and no --yaw given. Run yaw_calib.py first, "
                     "or pass --yaw <degrees> to set U by hand.")


# ---------------------------------- main ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="One vision-guided pick, stepped by hand.")
    ap.add_argument("--part", type=int, default=0, help="index into pick_list (0 = topmost)")
    ap.add_argument("--yaw", type=float, default=None, help="override U in degrees")
    ap.add_argument("--place", type=str, default=None, help='place pose "x,y,z" (gripper-down reachable)')
    ap.add_argument("--current", type=int, default=GRIP_CURRENT,
                    help=f"grip current, raw units (default {GRIP_CURRENT}; higher = firmer)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global PLACE
    if args.place:
        PLACE = tuple(float(v) for v in args.place.split(","))

    picks = json.load(open(PICK_LIST))
    if not picks:
        raise SystemExit("pick_list.json is empty - run pose_seg.py first.")
    p = picks[args.part]
    u, src = resolve_yaw(p, args.yaw)

    print(f"\nTarget: [{args.part}] {p['label']}  @({p['x']:.1f}, {p['y']:.1f}, {p['z']:.1f})")
    print(f"  vision yaw {p['yaw']:.1f}  ->  U {u:.1f}   ({src})")
    print(f"  aspect {p.get('aspect','?')}"
          + ("   <- NEAR-SQUARE, yaw unreliable" if p.get("aspect", 9) < 1.25 else ""))
    print(f"  place pose {PLACE}")
    if args.dry_run:
        print("  DRY RUN - no hardware will move.")
    input("\nHand on the e-stop. Press Enter to begin...")

    arm  = DryArm() if args.dry_run else RealArm()
    grip = MockGripper() if args.dry_run else Gripper()
    arm.connect(); grip.connect()

    hover_z = p["z"] + APPROACH_MM - TOOL_Z_OFFSET_MM
    grasped = False
    try:
        grip.open()
        print("\n-- hover above the part, jaws aligned --")
        _ok(arm.jump(p["x"], p["y"], hover_z, u))
        print("  CHECK: are the open jaws straddling the part's SHORT width?")
        if not input("  Looks right? [y/N] ").strip().lower().startswith("y"):
            print("  stopping - fix the yaw before picking.")
            return

        # ---- stepped descent ----
        print(f"\n-- descent: Enter = down {STEP_MM:.0f}mm, u = up, g = grip, a = abort --")
        drop = 0.0
        while True:
            cmd = input(f"  [{drop:+.0f} mm below hover] > ").strip().lower()
            if cmd == "a":
                print("  aborted by operator.")
                return
            if cmd == "g":
                break
            if cmd == "u":
                drop -= STEP_MM
            else:
                if drop + STEP_MM > APPROACH_MM + MAX_DROP_MM:
                    print(f"  REFUSING: that is more than {MAX_DROP_MM:.0f}mm below the part top.")
                    continue
                drop += STEP_MM
            _ok(arm.go(p["x"], p["y"], hover_z - drop))

        grasp_dz = APPROACH_MM - drop        # relative to the part's top surface
        print(f"\n  gripping here. GRASP_DZ_MM = {grasp_dz:+.1f} "
              f"(fingertips this far above the part top)")

        pos, holding = grip.close(args.current)
        print(f"  gripper: position={pos}  holding={holding}  (current {args.current} raw)")
        if not holding:
            print("  -> NOTHING GRASPED. Opening and stopping.")
            grip.open()
            return
        grasped = True

        input("\n  Grasp looks good? Press Enter to LIFT (or Ctrl-C to stop)...")
        _ok(arm.go(p["x"], p["y"], hover_z))

        # Re-check AFTER the lift: this is where a marginal grip actually fails.
        # The passive read is only a cheap pre-filter now - under the bounded
        # freeze the fingers sit near their goal whether or not the part is there,
        # so the ACTIVE probe below is what actually decides.
        after = grip.position()
        drop = pos - after
        print(f"  lifted. gripper position {pos} -> {after}  (fell {drop} ticks)")
        still, travel, note = grip.probe(p["label"])
        print(f"  probe: travelled {travel} ticks -> holding={still}"
              + (f"   [{note}]" if note else ""))
        if not still or drop > SLIP_DROP_TICKS:
            print("  !! LOST IT DURING THE LIFT - the fingers ran on toward closed.")
            print("  !! The part is probably no longer held. Check before continuing.")
            if not input("  Continue anyway? [y/N] ").strip().lower().startswith("y"):
                grip.open()
                return

        input("  Press Enter to move to the PLACE pose...")
        _ok(arm.jump(PLACE[0], PLACE[1], PLACE[2], u))
        input("  Press Enter to RELEASE...")
        grip.open()
        print("\n  PLACED. First real pick complete.")

    except ArmFault as e:
        print(f"\n  !! {e}\n  !! stopping for safety.")
    except (TimeoutError, OSError) as e:
        print(f"\n  !! NO REPLY FROM THE ARM ({e}).")
        print("  !! The controller has almost certainly faulted and stopped.")
        print("  !! Check the RC+ output window for the motion error, then reset the task.")
        print("  !! Most likely cause: the commanded pose is unreachable in the CURRENT")
        print("  !! wrist orientation (gripper-down). Check your --place pose height.")
    except KeyboardInterrupt:
        print("\n  interrupted.")
    finally:
        if grasped:
            print("  (if the part is still held, it is released now)")
        grip.release()
        arm.disconnect()
        print("\nDone. Tell me GRASP_DZ_MM and the gripper positions.")


if __name__ == "__main__":
    main()
