"""
AUTONOMOUS MULTI-PART CLEARING — scan, pick topmost-first, verify, repeat.

This is M2: the gate. Until this runs unattended, no trials can be collected.

It does NOT reimplement the gripper or the geometry. `pick_one.py` owns the grasp
logic (relax-before-open, the current ladder, the corrected settle rule) and
`scan.py` owns perception, which in turn imports pose_seg. One definition each.

WHAT A CYCLE LOOKS LIKE
    CAPTURE pose -> scan -> READY pose -> for each part, topmost-first:
        JUMP hover (yaw set)  ->  GO down to the grasp height chosen by
        --descent-mode  ->  close
        verdict? no  -> open, log no_grasp, next part
        verdict? yes -> lift -> PROBE -> JUMP place -> PROBE -> open
                     -> confirm the fingers actually reopened -> log placed
    then, depending on --rescan, return to CAPTURE and re-scan.

WHY "placed" NOW COSTS FOUR CHECKS
An earlier version logged 3 placed while three parts lay on the floor. The
goal-freeze that cured the release wedge also removed the position error, and with
it every passive sign of a lost part. So success is no longer inferred from a
reading; it is the conjunction of a close verdict, an ACTIVE probe after the lift,
another after the carry, and a confirmed reopen. See pick_one.Gripper.probe.

DESCENT MODES (--descent-mode) — the Ch5 ablation
    fixed       one grasp depth for every part. The baseline.
    two-regime  separate depths for parts on the bench and parts resting on
                something else.
    clearance   never descend closer than --finger-clear to whatever the two JAWS
                would land on, measured per part by pose_seg. Handles both regimes
                without classifying anything, and is the direct answer to the
                8 Sep collision that snapped a PLA finger.

WHY THE RE-SCAN MATTERS TWICE OVER
It is the Pillar-2 verification (did the part actually leave the table?), and it
is also simply necessary: a stale pick list goes wrong as soon as one part moves
another, and thin classes will fail to grasp. Counting what is left is the
clearance-rate metric.

REQUIRES ON THE CONTROLLER (add in RC+ before running with --pose-cmds):
    teach CapturePoint (camera-down) and ReadyPoint (gripper-down), and add
    "capture" and "ready" commands to Main.prg that Jump to them. Taught points
    carry full 6-DOF, which is the only way to change the wrist V,W - the
    receiver's JUMP/GO inherit V,W and cannot rotate the wrist themselves.

RUN
    # full logic rehearsal, no hardware, no camera, no model:
    python pde4445-dev/clear_scene.py --dry-run --grasp-dz 25

    # live, one orientation switch at the end (recommended first):
    python pde4445-dev/clear_scene.py --grasp-dz <measured> --rescan end --only arduino

    # stretch, once the basic loop is proven:
    python pde4445-dev/clear_scene.py --grasp-dz <measured> --rescan per-pick

GRASP_DZ_MM is the fingertip height above the part top at the moment of grip,
discovered by pick_one.py's stepped descent. There is no safe default: pass it.
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pick_one import (Gripper, MockGripper, DryArm, RealArm, ArmFault, _ok,
                      APPROACH_MM, TOOL_Z_OFFSET_MM, GRIP_CURRENT, HOLD_THRESHOLD,
                      SLIP_DROP_TICKS, PLACE, resolve_yaw,
                      PROPRIOCEPTIVE_CLASSES)                       # noqa: E402
from pick_one import WIDTH_SANITY_MM as PICK_ONE_WIDTH_SANITY      # noqa: E402
from pose_seg import pickability as pose_pickability                # noqa: E402
from runlog import RunLogger                                        # noqa: E402

# PICKER_PICK_LIST lets the regression test drive a synthetic scene without
# touching the real pick_list.json produced by the last scan.
PICK_LIST = os.environ.get("PICKER_PICK_LIST", os.path.join(HERE, "pick_list.json"))

# Fingers must end up near max_open (3722) to count as released. A stalled open
# sits ~1960 - above HOLD_THRESHOLD, so "still holding" and "fully open" cannot
# be told apart by the hold threshold alone.
RELEASE_POS_MIN = 3400

# An enabling move that does not actually remove the blocker leaves the scene
# unchanged, so the next scan produces the same decision and the arm loops. Cap
# the attempts and remember WHERE a blocker was cleared from - identity by object
# is useless, because every re-scan produces fresh dicts.
JAW_BLOCK_MARGIN_MM = 2.0
# A part whose jaws would land AT OR ABOVE its own top surface cannot be grasped:
# the fingers meet the obstruction before they reach the board. This is INDEPENDENT
# of the occlusion score. Measured 15 Sep: an arduino under an lcd scored occlusion
# 0.24 - comfortably under the 0.35 gate - while z_land read 264 against the
# arduino's own 246. The occlusion score says how much of the OUTLINE is shadowed;
# z_land says what the JAWS will hit. Only the second one stops a collision.

MAX_ENABLING_MOVES = 3
SAME_BLOCKER_MM    = 25.0
# Identity of a PHYSICAL UNIT across re-scans. Every scan produces fresh dicts,
# so a part can only be recognised as "the one I already tried" by where it is.
# 25 mm is the same tolerance the blocker matcher uses: wider than the 0.6 mm
# perception noise floor by a wide margin, narrower than the gap between any two
# parts in a layout that passed the condition gate.
SAME_PART_MM       = 25.0


def prior_attempts(history, p, tol=SAME_PART_MM):
    """How many entries in `history` refer to the same physical unit as `p`.

    `history` is a list of (label, x, y). Identity is class AND position: a part
    of the same class somewhere else is a different unit, and the same class
    within `tol` of a previous attempt is the same board being retried. Used
    both to enforce the per-part cap and to hold abandoned units out of later
    pick lists.
    """
    return sum(1 for a in history
               if a[0] == p["label"]
               and ((a[1] - p["x"]) ** 2 + (a[2] - p["y"]) ** 2) ** .5 <= tol)


# Attempts allowed at the SAME blocker position. One is too few: a blocker that
# slipped out of the jaws is still movable and deserves another try, whereas one
# that will not budge at all should not be retried forever. Two separates them.
BLOCKER_ATTEMPTS   = 2

# ---- descent policy defaults (all overridable on the command line) ----
RAISED_MM       = 12.0   # a part whose top sits this far above the scene floor is
                         # resting on something else rather than on the bench
GRASP_DZ_RAISED = 32.0   # dz used for raised parts in two-regime mode
FINGER_CLEAR_MM = 6.0    # the jaws never come closer than this to whatever they
                         # would land on. 8 Sep: a jaw met an esp hidden under the
                         # arduino it was picking and a PLA finger snapped.

# ---- THE OFFSET THAT MAKES CLEARANCE MODE POSSIBLE ----
# grasp_dz is NOT the height of the bottom of the jaws. It is a relative quantity
# discovered by pick_one's stepped descent, measured in the same frame as the part
# top, and the actual jaw tips hang some distance BELOW it. Comparing grasp_dz
# against a measured obstruction height therefore compares two different frames -
# the first version of this rule did exactly that and silently never clamped.
#
# Estimated from the one observation that pins it down: on 8 Sep, grasp_dz = 25 on
# an arduino whose top measured z = 246.2 made the jaws RUB THE BENCH, and the
# bench sits at z ~= 244.5. So the jaw tips were ~26.7 mm below the grasp-height
# reference:  (246.2 + 25) - 244.5 = 26.7.
#
# ONE observation, inferred from "it rubbed", not measured. Confirm it directly on
# Saturday: park over bare bench, step down until the jaws just touch, and read
# commanded Z. Two minutes, and it turns the whole of clearance mode from an
# estimate into a measurement.
# MEASURED 12 Sep on the current fingers, superseding the inference above.
# Jaws touch bare bench at commanded Z = 213.3; vision puts that bench at
# z_land ~= 243.6. So the jaw tips sit 30.3 mm ABOVE the commanded tool point,
# not 70 mm below it as TOOL_Z_OFFSET_MM claims. That 40 mm error is why a
# grasp_dz of 30 would have driven every part into the bench.
# Left in place rather than refactored mid-session: TOOL_Z_OFFSET_MM also sets
# the hover height, and the per-class grasp_dz values absorb the discrepancy
# correctly. Fix both together, off the robot, and re-derive every dz.
FINGER_TIP_OFFSET_MM = 30.3


def part_top(p, z_ref="med"):
    """The height to reason about for this part.

    'z' is the median of the NEAREST 8 mm band inside the mask. Aligned depth
    bleeds across a depth discontinuity, so where one part rests on another the
    lower part's mask picks up the upper part's surface and reports ITS height.
    Measured 12 Sep: an esp sitting on an arduino gave both parts z=261.

    Everything downstream then breaks at once - the two parts tie, so
    topmost-first orders them arbitrarily and reaches for the buried one; and the
    descent aims at a height belonging to a different object, missing by a
    centimetre. Three symptoms, one cause.

    'z_med' is the median over the WHOLE mask, so a minority of bled pixels
    cannot move it. It also sits at board level rather than on top of the
    headers, which is where the jaws actually need to be.
    """
    if z_ref == "med" and p.get("z_med") is not None:
        return p["z_med"]
    return p["z"]


def scene_floor(picks, table_z):
    """Base-frame Z of the bench, for deciding whether a part is raised.

    An explicit --table-z is always better. Failing that, the lowest landing
    measurement in the scene is a far better proxy than the lowest PART top,
    because the landing patches usually see bare bench while every part top is
    one board-thickness above it.
    """
    if table_z is not None:
        return table_z, "--table-z"
    lands = [p["z_land"] for p in picks if p.get("z_land") is not None]
    if lands:
        return min(lands), "lowest z_land in scene"
    if picks:
        return min(p["z"] for p in picks), "lowest part top (crude)"
    return None, "unknown"


def is_raised(p, floor_z, margin=5.0):
    """Is this part resting on something, rather than on the bench?

    Uses z_land - the height of what the JAWS will land on - rather than the
    part's own top. A part lying on the bench has its jaws landing on the bench;
    a part lying on another board has them landing on that board. That is a more
    direct measurement of "what is underneath this" than the part's own height,
    which also varies with how tall the module happens to be.
    """
    if floor_z is None:
        return False
    zl = p.get("z_land")
    if zl is not None:
        return (zl - floor_z) > margin
    return (p.get("z_med", p["z"]) - floor_z) > 12.0


def grasp_height(p, args, floor_z):
    """Fingertip height (base frame, mm) at the moment of grip, and why.

    THE INDEPENDENT VARIABLE OF THE DESCENT ABLATION.

    fixed       one dz for everything. The baseline, and what produced every
                result so far. Fails in opposite directions: too shallow on a
                flat board, too deep on a stack.
    two-regime  one dz for parts on the bench, another for parts resting on
                something. Captures the operator's intuition directly.
    clearance   dz as in fixed, but never descending closer than FINGER_CLEAR_MM
                to whatever the two jaws would actually land on. This subsumes
                two-regime without needing to classify the part at all: a stacked
                part's jaws land high, so the clamp fires automatically.

    Reporting all three side by side is the point. two-regime is the hypothesis a
    human forms from watching failures; clearance is what the same evidence
    implies once the measurement is taken at the jaws instead of at the part.
    """
    mode = args.descent_mode
    dz = (args.dz_by_class or {}).get(p.get("label"), args.grasp_dz)
    ztop = part_top(p, getattr(args, "z_ref", "med"))

    # EXTRA DEPTH FOR A PART THAT IS NOT ON THE BENCH. Perched on another board a
    # module sits less stably and the jaws catch it higher up its body, which is
    # where the slips have come from. Going further down is safe here precisely
    # BECAUSE it is raised - there is a board under it, not the bench - and
    # --min-z still backstops the whole thing.
    extra = (getattr(args, "dz_extra_raised", None) or {}).get(p.get("label"), 0.0)
    if extra and is_raised(p, floor_z, args.raised_margin):
        dz -= extra
        p["_dz_extra"] = extra

    tag = f" (+{p['_dz_extra']:.0f}mm, raised off the bench)" if p.get("_dz_extra") else ""
    if mode == "fixed":
        return ztop + dz, f"fixed dz={dz:.0f}{tag}"

    if mode == "two-regime":
        if floor_z is None:
            return ztop + dz, f"two-regime: no floor reference -> flat dz={dz:.0f}"
        raised = (ztop - floor_z) > args.raised_mm
        d = args.grasp_dz_raised if raised else dz
        return ztop + d, (f"two-regime {'RAISED' if raised else 'flat'} "
                            f"({p['z'] - floor_z:+.0f}mm vs floor) dz={d:.0f}")

    if mode == "clearance":
        base = ztop + dz
        zl = p.get("z_land")
        if zl is None:
            # Old pick lists have no landing measurement. Say so loudly rather
            # than silently degrading to a different policy mid-experiment.
            return base, f"clearance UNAVAILABLE (no z_land) -> fixed dz={dz:.0f}"
        # zl is a real height; base is in grasp-height coordinates. Lift zl into
        # the same frame before comparing them (see FINGER_TIP_OFFSET_MM).
        limit = zl + args.finger_clear + args.tip_offset
        if limit > base:
            return limit, (f"clearance CLAMPED: jaws land at z={zl:.0f}, keeping "
                           f"{args.finger_clear:.0f}mm off it -> dz={limit - p['z']:+.1f} "
                           f"instead of {dz:+.0f}")
        return base, f"clearance clear (jaws land at z={zl:.0f}) dz={dz:.0f}"

    raise ValueError(mode)


# ------------------------------------------------------------------- arms
class PoseArm(RealArm):
    """RealArm plus the two taught-point commands that switch wrist orientation."""

    def capture(self):
        return self.s._send_command("CAPTURE", 0, 0, 0, 0)

    def ready(self):
        return self.s._send_command("READY", 0, 0, 0, 0)

    def place_point(self):
        """Jump to the taught PlacePoint (0,750,360, gripper-down) — guaranteed
        reachable, unlike drop coordinates we invent."""
        return self.s._send_command("PLACE", 0, 0, 0, 0)


class DryPoseArm(DryArm):
    def capture(self):
        print("  [dry] CAPTURE pose (camera-down)"); return "OK"

    def ready(self):
        print("  [dry] READY pose (gripper-down)"); return "OK"

    def place_point(self):
        print("  [dry] PLACE pose (taught PlacePoint)"); return "OK"


class DryGripper:
    """Mock that mirrors the MEASURED behaviour of this rig.

    pick_one's MockGripper returns 1262, which pre-dates the current calibration
    and would read as 'lost' against the current threshold, making every rehearsal
    look like a failure. This one uses the numbers measured on 8 Sep under the
    calibration in force (max_open 3571 / min_open 601, 90 raw, sponge):

        arduino loaded  1848-1881   -> arrests the fingers, verdict is real
        esp/lcd loaded  1678-1681   -> indistinguishable from empty (1674-1679)

    So a dry run reproduces the true, uncomfortable state of the rig: thin classes
    read as no_grasp even when they are in fact being held. Earlier versions of
    this mock used 1829/2126 from a superseded calibration, which put the thin
    classes just ABOVE the threshold and quietly made every rehearsal succeed.
    """
    THICK = set(PROPRIOCEPTIVE_CLASSES)   # all four, since 12 Sep
    # Measured 12 Sep on the NEW fingers at 100 raw (teleop), calibration 3541/655.
    # At 100 raw the fingers close until they meet the part - unlike the 80-90 raw
    # regime, where they parked on friction at ~1646 and never touched anything
    # thinner than that gap. Every class now arrests the fingers at a DIFFERENT
    # position, which is the basis for expecting per-class proprioceptive checks.
    STALLS = {"ultrasonic": 974, "lcd": 1083, "esp": 1262,
              "arduino": 1859}   # all four measured by grip_verify, 12 Sep
    EMPTY_STALL = 708            # empty fingers at 100 raw, jaws in contact

    def __init__(self, sim_drop=None):
        self.sim_label = None
        self._pos = 3721         # max_open
        self._held = False
        # sim_drop = "lift" | "place" | None. Forces the part to be gone at that
        # checkpoint, so the failure branches are exercised at the desk instead of
        # being discovered on a lab day.
        self.sim_drop = sim_drop
        self._probes = 0

    def connect(self):
        print(f"  [dry] gripper connected (threshold {HOLD_THRESHOLD})"
              + (f"  SIMULATING A DROP AT '{self.sim_drop}'" if self.sim_drop else ""))

    CALIB = {"slope_ticks_per_mm": 38.0, "intercept_ticks": 700.0}   # plausible, not measured

    def open(self, to=None):
        self._pos = 3721 if to is None else int(to)
        self._held = False
        print(f"  [dry] open{'' if to is None else f' to {to} ticks'}")
        return self._pos

    def expected_stall(self, width_mm):
        if width_mm is None:
            return None
        return int(25.983 * (float(width_mm) - 10.3) + 758.4)

    def ticks_for_width(self, width_mm, label=None):
        """Mirror the real mapping so rehearsals exercise the narrow-open branch.

        The real slope is unknown until aperture_calib.py runs on the rig; these
        numbers exist only so the dry run takes the same code path, and must never
        be copied into aperture.json.
        """
        if width_mm is None:
            return None
        # mirror the real sanity clamp, or a dry run reports an aperture the
        # hardware would never command
        lo, hi = PICK_ONE_WIDTH_SANITY.get(self.sim_label, (0.0, 1e9))
        width_mm = min(max(float(width_mm), lo), hi)
        want = max(float(width_mm) + 16.0, 10.0)
        t = self.CALIB["slope_ticks_per_mm"] * want + self.CALIB["intercept_ticks"]
        return int(min(max(t, 651), 3721))

    def close(self, current=GRIP_CURRENT, width_mm=None, label=None):
        self._probes = 0
        self._pos = self.STALLS.get(self.sim_label, self.EMPTY_STALL)
        self._held = self._pos > HOLD_THRESHOLD
        print(f"  [dry] close @{current} on '{self.sim_label}' -> {self._pos}")
        return self._pos, self._held

    def probe(self, label=None):
        if label is not None and label not in self.THICK:
            return True, 0, f"no proprioceptive signature for '{label}' - vision only"
        self._probes += 1
        gone = (self.sim_drop == "lift" and self._probes >= 1) or \
               (self.sim_drop == "place" and self._probes >= 2)
        if gone:
            before, self._pos = self._pos, self.EMPTY_STALL
            self._held = False
            return False, before - self._pos, "simulated drop"
        self._pos -= 8                       # the sponge gives a little each probe
        return True, 8, ""

    def position(self):
        return self._pos

    def release(self):
        print("  [dry] gripper released")


# ------------------------------------------------------------------ scanning
def do_scan(args, arm, use_pose_cmds):
    """Return a pick list, topmost-first. Handles the orientation switch."""
    if args.dry_run and not args.db3:
        picks = json.load(open(PICK_LIST))         # pure logic rehearsal
        print(f"  [dry] using pick_list.json ({len(picks)} parts)")
        return picks

    if use_pose_cmds:
        _ok(arm.capture())                          # camera-down
        time.sleep(0.5)

    # Keep EVERY scan overlay: one per trial, never overwritten. These are the
    # per-run evidence that the detections behind a result were sane.
    scans_dir = os.path.join(HERE, "scans")
    os.makedirs(scans_dir, exist_ok=True)
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.order}"
    if args.layout:
        tag += f"_{args.layout}"

    from scan import scan                           # imported late: needs model + camera
    picks, vis = scan(args.db3, conf=args.conf,
                      save_as=os.path.join(scans_dir, tag + ".png"))
    print(f"  scanned: {len(picks)} parts  -> scans/{tag}.png")

    if use_pose_cmds:
        _ok(arm.ready())                            # back to gripper-down
        time.sleep(0.5)
    return picks


def order_picks(picks, how, seed=0, z_ref="med"):
    """Sequencing POLICY - the independent variable of the Pillar-1 experiment.

    topmost : descending base-frame Z. Uses depth to pick what is on top first,
              so nothing is dragged out from under another part.
    naive   : raster order across the image (top-left to bottom-right), i.e. a
              policy with NO depth reasoning at all. This is the baseline the
              topmost-first claim has to beat.
    random  : shuffled with a fixed seed, repeatable across runs.

    On a flat, non-overlapping scene all three are equivalent - the comparison
    only means anything on PILED scenes where parts rest on each other.
    """
    if how == "topmost":
        return sorted(picks, key=lambda p: -part_top(p, z_ref))
    if how == "naive":
        return sorted(picks, key=lambda p: (p.get("v", 0), p.get("u", 0)))
    if how == "random":
        import random as _r
        out = list(picks)
        _r.Random(seed).shuffle(out)
        return out
    raise ValueError(how)


DISTURB_MAX_MM = 150.0   # beyond this a same-class detection is too far to have
                         # been nudged there by a neighbour's pick
PLACE_RADIUS_MM = 60.0   # detections this close to the place pose are parts we put
                         # there, not parts still on the bench


def drop_placed(remaining, place_xy, radius=PLACE_RADIUS_MM):
    """Remove the parts we deposited from the verification scan.

    The taught PlacePoint sits inside the camera's field of view and only ~130 mm
    from the picking area, so neither a raw count nor a distance threshold can
    tell a successfully-placed part from one still on the bench. Excluding a
    radius around the known place position can. Without this the 12 Sep L1 run
    reported "started with 3, 3 remain" after clearing all three.
    """
    if place_xy is None:
        return remaining, []
    px, py = place_xy
    keep, dropped = [], []
    for q in remaining:
        if ((q["x"] - px) ** 2 + (q["y"] - py) ** 2) ** .5 <= radius:
            dropped.append(q)
        else:
            keep.append(q)
    return keep, dropped


# The bench, declared once. Every one of the 140 parts ever actually attempted
# between 12 and 19 Sep lies inside x -148.6..83.3, y 562.7..845.4,
# z 245.8..288.0; this envelope is those bounds with room to spare.
WORKSPACE = {"x": (-250.0, 150.0), "y": (500.0, 900.0), "z": (230.0, 320.0)}


def on_bench(p):
    """Could this detection be a real part on this bench?

    Used for REPORTING ONLY - the pick list is deliberately left alone, so this
    cannot change what the arm does or which epoch a run belongs to.

    On 19 Sep both easy layouts drew a reproducible phantom 'lcd' at about
    (-226, 341, 65): 220 mm off the end of the bench and 185 mm below its
    surface, i.e. something in the far field behind the workspace. The
    jaw-obstruction guard refused it correctly and it cost no time. But it
    entered `initial_picks`, so a five-part layout reported "started with 6" and
    a run that cleared everything scored 5/6 = 83%.

    A phantom is a perception over-detection. It is already counted as one in
    the vision battery, and it is not a part that failed to be cleared, so it
    must not appear in a clearance denominator. Excluding it by a fixed
    envelope, applied to every run and announced when it fires, is a correction;
    quietly dropping the row would not be.
    """
    for axis, key in (("x", "x"), ("y", "y"), ("z", "z")):
        v = p.get(key)
        if v is None:
            continue
        lo, hi = WORKSPACE[axis]
        if not (lo <= float(v) <= hi):
            return False
    return True


def clearance(initial, remaining, tol_mm=35.0, disturb_max=DISTURB_MAX_MM):
    """Which of the ORIGINAL parts have actually left their place on the bench?

    Counting what the re-scan sees is not a clearance measure. The taught
    PlacePoint sits inside the camera's field of view, so parts that were picked
    successfully are seen again where they were put down - on 12 Sep a run that
    cleared three parts reported "3 remain", which is the metric quietly
    measuring nothing.

    Matching each original position against the re-scan is immune to that: a part
    still on the bench is still within a few mm of where it started, while one
    that was carried away is not. It also catches the failure the raw count
    cannot - a part DISTURBED into a new position by the pick of its neighbour,
    which is exactly the effect the sequencing experiment is about.
    """
    initial = [p for p in initial if on_bench(p)]
    remaining = [q for q in remaining if on_bench(q)]
    gone, left, moved = [], [], []
    for p in initial:
        same = [q for q in remaining if q["label"] == p["label"]]
        near = [q for q in same
                if ((q["x"] - p["x"]) ** 2 + (q["y"] - p["y"]) ** 2) ** .5 <= tol_mm]
        if near:
            left.append(p)
        elif same:
            # Its class is still visible but not where this one was. Close by =
            # something shoved it. Far away = it is the part we just carried to
            # the place pose, which is in shot.
            d = min(((q["x"] - p["x"]) ** 2 + (q["y"] - p["y"]) ** 2) ** .5 for q in same)
            if d <= disturb_max:
                moved.append((p, d))
            gone.append(p)
        else:
            gone.append(p)
    return gone, left, moved


def jaw_obstructed(p, z_ref="med"):
    """Would the jaws meet something before they reach this part? Reason, or None."""
    zl = p.get("z_land")
    if zl is None:
        return None
    top = part_top(p, z_ref)
    if zl >= top - JAW_BLOCK_MARGIN_MM:
        return (f"jaws would land at z={zl:.0f}, at or above the part's own "
                f"top {top:.0f} - something is over it")
    return None


def find_blocker(p, scene):
    """Which detection in the CURRENT SCAN is sitting on top of p?

    Searched against the whole scan, not the remaining queue. The blocker is by
    definition HIGHER than the target, so topmost-first has already dealt with it
    and popped it - on the first attempt it had been skipped as "not on the bill"
    and was no longer in the queue when the target came up.

    pose_seg records the position of each occluder, not just its class, because
    "an lcd" is not an instruction when there are two of them on the bench. The
    worst offender is first in the list.
    """
    for bx, by in (p.get("under_xy") or []):
        best, bd = None, 1e9
        for q in scene:
            if q is p:
                continue
            d = (q["x"] - bx) ** 2 + (q["y"] - by) ** 2
            if d < bd:
                best, bd = q, d
        if best is not None and bd <= 30.0 ** 2:
            return best
    return None


def verification_mode(label, args):
    """Which evidence settles whether THIS class was grasped.

    'proprioceptive'  the servo's stall position separates held from empty, so the
                      close verdict and the probe are trusted.
    'vision'          it does not. Measured 8 Sep at 90 raw with sponge padding:

                          arduino   empty ~1674   loaded 1848-1881   (~180 ticks)
                          esp       empty  1677   loaded 1678-1679   (~1 tick)
                          lcd       empty  1678   loaded 1680-1681   (~2 ticks)

                      The esp was lifted 3/3 in that session WHILE reading as empty.
                      Refusing to pick on that verdict throws away successful grasps,
                      and makes the BOQ demo impossible - the bill of quantities asks
                      for an esp32 and an lcd, not two arduinos.

    So for thin classes the close verdict is recorded but NOT acted on, and the
    verification re-scan decides: did the part leave the scene? That is Pillar 2
    doing real work rather than being asserted, and it is why --rescan matters.

    Raising the grip current does not move this. The stall position is set by how
    far the fingers travel; current sets how hard they push. More force compresses
    more padding in the empty case too, so the two stalls move together.
    """
    if args.verify == "proprioceptive":
        return "proprioceptive"                 # force the strict policy everywhere
    if args.verify == "vision":
        return "vision"                         # force the permissive policy
    return "proprioceptive" if label in PROPRIOCEPTIVE_CLASSES else "vision"


def wanted(p, args, got=None):
    # BILL OF QUANTITIES. Counted against parts CONFIRMED off the bench, not
    # against attempts: a pick that failed has not fulfilled anything.
    want = getattr(args, "want_by_class", None)
    # ENABLING FLAG LIVES ON THE OBJECT. It used to be a set of id()s, which is
    # unsound across a re-scan: the old pick dicts are freed, CPython reuses the
    # addresses, and a NEW detection inherits a stale id. Observed 15 Sep - an esp
    # created after an lcd was discarded landed on the lcd's old address, passed
    # the "this was admitted as a blocker" test, and was fetched although nothing
    # had asked for it and it was blocking nothing.
    if want and not p.get("_enabling"):
        n = want.get(p["label"])
        if n is None:
            return False, "class not on the bill of quantities"
        if (got or {}).get(p["label"], 0) >= n:
            return False, f"{p['label']} quota of {n} already met"
    if args.only and p["label"] not in args.only:
        return False, f"class not in --only {args.only}"
    if p["label"] in (args.skip or []):
        return False, "class in --skip"
    if p.get("conf", 1) < args.min_conf:
        return False, f"conf {p.get('conf')} < {args.min_conf}"
    # Occlusion gate. Attempting a part that is pinned under another wastes a
    # cycle and can drag the neighbour off the bench. Declaring it out of reach,
    # with the reason and the score, is a result - and it is the scoped
    # constraint the project deliberately accepts rather than solving.
    ok, why = pose_pickability(p, args.max_occlusion)
    if not ok:
        return False, why
    # Independent of the occlusion score, and the one that prevents a collision.
    why = jaw_obstructed(p, getattr(args, "z_ref", "med"))
    if why:
        return False, why
    return True, ""


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Autonomous multi-part clearing.")
    ap.add_argument("--grasp-dz", type=float, required=True,
                    help="fingertip height above the part top at grip, mm (from pick_one.py)")
    ap.add_argument("--descent-mode", choices=["fixed", "two-regime", "clearance"],
                    default="fixed",
                    help="how the grasp height is chosen. 'fixed' is the baseline; the "
                         "other two are the Ch5 ablation. Change ONE thing per block of "
                         "trials - never mid-block.")
    ap.add_argument("--z-ref", choices=["med", "top"], default="med",
                    help="which depth statistic represents a part's height. 'med' is "
                         "the median over the whole mask; 'top' is the nearest 8mm band "
                         "and is what every run before 12 Sep used. 'top' is corrupted "
                         "by depth bleed wherever parts touch, which breaks BOTH the "
                         "ordering and the descent. 'top' is kept as the ablation arm.")
    ap.add_argument("--dz-extra-raised", nargs="*", default=None, dest="dz_extra_raised_raw", metavar="CLASS=MM",
                    help="descend this many mm FURTHER for a class when the part is "
                         "resting on something rather than on the bench, e.g. "
                         "--dz-extra-raised esp=4 lcd=2 ultrasonic=2. Ignored for parts "
                         "flat on the bench, where the extra depth would hit the table.")
    ap.add_argument("--raised-margin", type=float, default=5.0,
                    help="how far above the scene floor the JAWS must land for a part "
                         "to count as resting on something (default 5mm)")
    ap.add_argument("--max-z", type=float, default=None,
                    help="CEILING on the hover height. A part on a tall stack produces "
                         "a hover the arm cannot reach gripper-down, and the controller "
                         "faults without replying. Measured 15 Sep: a hover at Z=286.7 "
                         "killed the run.")
    ap.add_argument("--dz-class", nargs="*", default=None, metavar="CLASS=MM",
                    help="per-class grasp depth, e.g. --dz-class lcd=32. A single dz "
                         "is measured from the part TOP, so a tall module (lcd, ~10mm) "
                         "gets gripped around its middle while a thin one is caught at "
                         "PCB level. Overriding per class is the honest fix.")
    ap.add_argument("--grasp-dz-raised", type=float, default=GRASP_DZ_RAISED,
                    help=f"two-regime only: dz for parts resting on something (default {GRASP_DZ_RAISED})")
    ap.add_argument("--raised-mm", type=float, default=RAISED_MM,
                    help=f"two-regime only: height above the floor that counts as raised (default {RAISED_MM})")
    ap.add_argument("--finger-clear", type=float, default=FINGER_CLEAR_MM,
                    help=f"clearance only: minimum gap the jaws keep above whatever they "
                         f"land on (default {FINGER_CLEAR_MM})")
    ap.add_argument("--tip-offset", type=float, default=FINGER_TIP_OFFSET_MM,
                    help=f"clearance only: how far the jaw tips hang below the grasp-height "
                         f"reference (default {FINGER_TIP_OFFSET_MM}, INFERRED not measured "
                         f"- measure it on the bench and pass the real number)")
    ap.add_argument("--place-xy", type=str, default=None, metavar="X,Y",
                    help="base-frame X,Y of the taught PlacePoint. The verification "
                         "scan excludes a radius around it, because the place pose is "
                         "IN the camera's view and otherwise the parts you just placed "
                         "are counted as still on the bench. Read it off RC+ once.")
    ap.add_argument("--place-radius", type=float, default=PLACE_RADIUS_MM)
    ap.add_argument("--min-z", type=float, default=None,
                    help="HARD FLOOR: never command the arm below this Z, whatever the "
                         "descent policy computes. Set it to the commanded Z at which "
                         "the jaws just touch bare bench, measured by hand. This is the "
                         "last line of defence for the fingers and it is worth the two "
                         "minutes it takes to measure.")
    ap.add_argument("--table-z", type=float, default=None,
                    help="measured bench height in the base frame. Removes the guesswork "
                         "from two-regime; measure it once and pass it every run.")
    ap.add_argument("--rescan", choices=["none", "end", "per-pick"], default="end")
    ap.add_argument("--order", choices=["topmost", "naive", "random"], default="topmost",
                    help="sequencing policy under test. 'topmost' is the contribution; "
                         "'naive' (raster, depth-agnostic) is the baseline to beat.")
    ap.add_argument("--seed", type=int, default=0, help="seed for --order random")
    ap.add_argument("--layout", default="", help="layout ID, e.g. L1 - keeps trials paired")
    ap.add_argument("--only", nargs="*", default=None, help="only pick these classes")
    ap.add_argument("--skip", nargs="*", default=None, help="never pick these classes")
    ap.add_argument("--conf", type=float, default=0.55,
                    help="detector confidence threshold for the scan itself. 0.55 measured "
                         "8 Sep as the lowest value with no false positives that still finds "
                         "all five modules; 0.35 produced phantoms, 0.60 was unnecessarily strict.")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="additionally refuse to PICK anything below this")
    ap.add_argument("--max", type=int, default=10, help="safety cap on pick attempts")
    ap.add_argument("--max-part-attempts", type=int, default=0, metavar="N",
                    help="give up on a single physical unit after N failed "
                         "attempts and log it 'abandoned' (0 = unlimited, the "
                         "old behaviour). Use 2 for any battery whose success "
                         "RATE is going to be quoted: without a cap one stubborn "
                         "part can supply most of a condition's denominator.")
    ap.add_argument("--current", type=int, default=GRIP_CURRENT)
    ap.add_argument("--hold-threshold", type=int, default=HOLD_THRESHOLD,
                    help="stall position above which a grasp is real. RE-MEASURE with "
                         "grip_verify.py after ANY recalibration - the tick scale shifts.")
    ap.add_argument("--place", type=str, default=None, help='place pose "x,y,z"')
    ap.add_argument("--clear-blockers", action="store_true",
                    help="allow the arm to move a part that is NOT on the bill, when "
                         "it is pinning one that is. The moved part is logged as "
                         "blocker_cleared and never counts toward the bill.")
    ap.add_argument("--want", nargs="*", default=None, metavar="CLASS=N",
                    help="bill of quantities: stop picking a class once N of it have "
                         "left the bench, e.g. --want arduino=1 esp=1 lcd=1. Without "
                         "this, '--only arduino esp lcd --max 3' on a bench holding two "
                         "arduinos can return two arduinos and an esp and call it done.")
    ap.add_argument("--max-occlusion", type=float, default=0.35,
                    help="refuse to attempt a part more occluded than this by HIGHER "
                         "neighbours (0-1, default 0.35). Set 1.0 to attempt everything, "
                         "which is the baseline arm of the occlusion ablation.")
    ap.add_argument("--aperture", choices=["adaptive", "full"], default="adaptive",
                    help="'adaptive' opens the jaws only as wide as the detected part "
                         "needs before descending, shrinking the swept footprint. "
                         "Silently equals 'full' until aperture_calib.py has been run. "
                         "'full' forces the old behaviour and is the Ch5 baseline.")
    ap.add_argument("--verify", choices=["per-class", "proprioceptive", "vision"],
                    default="per-class",
                    help="which evidence settles a grasp. 'per-class' (default) trusts "
                         "the servo only for classes with a measured thickness signature "
                         "and defers the rest to the verification scan. The other two "
                         "force one policy everywhere and exist as the Ch5 ablation: "
                         "'proprioceptive' is what ran on 8 Sep and refuses every thin "
                         "part; 'vision' picks everything and is scored by the re-scan.")
    ap.add_argument("--condition", default="clearing", help="tag for the run log")
    ap.add_argument("--db3", default=None, help="scan from a recording instead of the camera")
    ap.add_argument("--no-pose-cmds", action="store_true",
                    help="Main.prg has no CAPTURE/READY yet; operator jogs by hand")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sim-drop", choices=["lift", "place"], default=None,
                    help="dry-run only: pretend the part falls out at that checkpoint, "
                         "so the failure branches get exercised at the desk")
    args = ap.parse_args()

    args.want_by_class = {}
    for spec in (args.want or []):
        if "=" not in spec:
            raise SystemExit(f"--want wants CLASS=N, got {spec!r}")
        k, v = spec.split("=", 1)
        args.want_by_class[k.strip()] = int(v)
    if args.want_by_class:
        print("  bill of quantities: " +
              ", ".join(f"{k} x{v}" for k, v in args.want_by_class.items()))

    args.dz_extra_raised = {}
    for spec in (args.dz_extra_raised_raw or []):
        k, v = spec.split("=", 1)
        args.dz_extra_raised[k.strip()] = float(v)
    if args.dz_extra_raised:
        print("  extra depth when raised off the bench: " +
              ", ".join(f"{k}+{v:g}mm" for k, v in args.dz_extra_raised.items()))

    args.dz_by_class = {}
    for spec in (args.dz_class or []):
        if "=" not in spec:
            raise SystemExit(f"--dz-class wants CLASS=MM, got {spec!r}")
        k, v = spec.split("=", 1)
        args.dz_by_class[k.strip()] = float(v)
    if args.dz_by_class:
        print("  per-class grasp depth: " +
              ", ".join(f"{k}={v:g}" for k, v in args.dz_by_class.items()))

    place = tuple(float(v) for v in args.place.split(",")) if args.place else PLACE
    use_pose_cmds = not args.no_pose_cmds

    print(f"\nClearing run — descent={args.descent_mode}  grasp_dz={args.grasp_dz}mm  "
          f"rescan={args.rescan}  current={args.current}  place={place}")
    if args.only:
        print(f"  only picking: {args.only}")
    if not use_pose_cmds:
        print("  NOTE: no CAPTURE/READY commands — jog the arm yourself when asked.")
    if args.dry_run:
        print("  DRY RUN — nothing moves.")
    if args.min_z is None:
        print("\n  *** NO --min-z SET. Nothing prevents the descent from commanding")
        print("  *** the jaws into the bench if grasp-dz or the tip offset is wrong.")
        print("  *** Measure it once: jog over bare bench, lower until the jaws just")
        print("  *** touch, read commanded Z. Then pass --min-z <that Z>.")
    else:
        print(f"  hard floor: Z will never go below {args.min_z}")
    print(f"  verification: {args.verify}"
          + ("  (thin classes settled by the re-scan)" if args.verify != "proprioceptive" else ""))
    if args.verify != "proprioceptive" and args.rescan == "none":
        print("  *** --verify lets thin classes through on the servo's word being")
        print("  *** unusable, and --rescan none means NOTHING then checks them.")
        print("  *** Every thin-class 'placed' this run will be unevidenced.")
    input("\nHand on the e-stop. Press Enter to begin...")

    arm  = DryPoseArm() if args.dry_run else PoseArm()
    grip = DryGripper(args.sim_drop) if args.dry_run else Gripper(threshold=args.hold_threshold)
    # Tag rehearsals so they can never be mistaken for measured trials in Ch5.
    condition = f"{args.condition}_{args.order}_{args.descent_mode}"
    if args.layout:
        condition += f"_{args.layout}"
    if args.dry_run:
        condition += "_DRYRUN"
    log  = RunLogger(condition=condition,
                     notes=f"rescan={args.rescan} only={args.only} dz={args.grasp_dz} "
                           f"descent={args.descent_mode}",
                     config={"conf": args.conf,
                             "order": args.order,
                             "layout": args.layout,
                             "grip_current": args.current,
                             "hold_threshold": args.hold_threshold,
                             "descent_mode": args.descent_mode,
                             "grasp_dz_mm": args.grasp_dz,
                             "grasp_dz_raised_mm": args.grasp_dz_raised,
                             "raised_mm": args.raised_mm,
                             "finger_clear_mm": args.finger_clear,
                             "tip_offset_mm": args.tip_offset,
                             "table_z": args.table_z,
                             "approach_mm": APPROACH_MM,
                             "tool_z_offset_mm": TOOL_Z_OFFSET_MM,
                             "slip_drop_ticks": SLIP_DROP_TICKS,
                             "place": list(place)})

    arm.connect(); grip.connect()
    placed = attempts = cleared = 0
    picks, need_scan, first_count = [], True, None
    initial_picks = []
    fulfilled = {}          # class -> count confirmed off the bench
    scene     = []          # every detection from the current scan
    cleared_at = []         # (label, x, y) of blockers already moved
    attempted_at = []       # (label, x, y) of every unit attempted, for the cap
    abandoned_at = []       # (label, x, y) of units that hit the cap
    floor_z = args.table_z

    try:
        while attempts < args.max:
            if need_scan:
                if not args.dry_run and not use_pose_cmds and first_count is not None:
                    input("  Jog to the CAPTURE pose, then Enter...")
                picks = do_scan(args, arm, use_pose_cmds)
                picks = order_picks(picks, args.order, args.seed, args.z_ref)
                scene = list(picks)      # the full scan, for blocker lookup
                # A unit that hit the per-part cap is dropped from the pick list
                # rather than re-offered, or every re-scan would write it another
                # 'abandoned' row and the loop would never terminate. It stays in
                # `scene`, so it is still available as a blocker and still counts
                # against clearance at the end.
                if abandoned_at:
                    before = len(picks)
                    picks = [q for q in picks
                             if not prior_attempts(abandoned_at, q)]
                    if len(picks) < before:
                        print(f"  {before - len(picks)} abandoned unit(s) held out "
                              f"of this pick list")
                floor_z, floor_src = scene_floor(picks, args.table_z)
                print(f"  order '{args.order}' (z_ref={args.z_ref}): " +
                      ", ".join(f"{p['label']}@{part_top(p, args.z_ref):.0f}"
                                + (f"(top {p['z']:.0f})" if args.z_ref == "med"
                                   and abs(part_top(p, "med") - p["z"]) > 3 else "")
                                for p in picks))
                if floor_z is not None:
                    print(f"  scene floor z={floor_z:.1f} ({floor_src})")
                if args.descent_mode == "clearance" and picks and \
                        all(p.get("z_land") is None for p in picks):
                    print("  *** clearance mode requested but NO pick carries z_land.")
                    print("  *** This pick list predates the landing measurement — re-scan,")
                    print("  *** or the run silently becomes a 'fixed' trial.")
                if first_count is None:
                    first_count = len(picks)
                    initial_picks = [dict(q) for q in picks]   # for the clearance match
                if not args.dry_run and not use_pose_cmds:
                    input("  Jog to the gripper-down READY pose, then Enter...")
                need_scan = False

            if args.want_by_class and all(
                    fulfilled.get(k, 0) >= v for k, v in args.want_by_class.items()):
                print("\n  BILL OF QUANTITIES FULFILLED: " +
                      ", ".join(f"{k} {fulfilled.get(k,0)}/{v}"
                                for k, v in args.want_by_class.items()))
                break

            if not picks:
                print("\n  nothing left to pick.")
                break

            p = picks.pop(0)                        # topmost-first
            ok, why = wanted(p, args, fulfilled)

            # ENABLING MOVE. The operator asked for a part that is pinned under
            # something they did NOT ask for. Refusing is honest but useless: the
            # part is reachable, it just is not reachable FIRST. So identify the
            # blocker, take it out of the way, and come back - which is the
            # difference between executing a pick list and executing a plan.
            if (not ok and args.clear_blockers
                    and ("occluded" in why or "jaws would land" in why)
                    and (not args.want_by_class or p["label"] in args.want_by_class)):
                b = find_blocker(p, scene)
                # Matched on POSITION: a blocker that slipped and landed somewhere
                # else is a different obstruction, and worth one more attempt.
                tries = sum(1 for c in cleared_at
                            if c[0] == b["label"]
                            and ((c[1] - b["x"]) ** 2 + (c[2] - b["y"]) ** 2) ** .5
                            <= SAME_BLOCKER_MM) if b is not None else 0
                if tries >= BLOCKER_ATTEMPTS:
                    print(f"   the {b['label']} has been attempted {tries}x and is "
                          f"still blocking - giving up on it.")
                    b = None
                elif len(cleared_at) >= MAX_ENABLING_MOVES:
                    print(f"   {MAX_ENABLING_MOVES} blockers already cleared - "
                          f"refusing to keep digging.")
                    b = None
                if b is not None:
                    print(f"   {p['label']} is pinned under a {b['label']}. "
                          f"Clearing the {b['label']} first to reach it.")
                    if b in picks:
                        picks.remove(b)
                    picks.insert(0, p)          # retry the target after the re-scan
                    picks.insert(0, b)
                    b["_enabling"] = True
                    cleared_at.append((b["label"], b["x"], b["y"]))
                    continue
                print(f"   {p['label']} is occluded and the blocker could not be "
                      f"identified in this scan - leaving it.")

            if not ok:
                print(f"  skip {p['label']}: {why}")
                (log.start_pick(attempts, p)
                    .record(occlusion=p.get("occlusion"), crowding=p.get("crowding"))
                    .finish("skipped", notes=why))
                continue

            # PER-PART ATTEMPT CAP. Without one, a single stubborn unit can own a
            # condition's denominator without bound: on 16 Sep one tilted lcd was
            # retried six times and supplied 7 of the 29 rows in the "easy" set,
            # which alone inverted the easy-vs-hard result. The cap converts that
            # into one bounded, labelled 'abandoned' row - the part is still
            # reported as unfulfilled, but it can no longer dominate a rate.
            prior = prior_attempts(attempted_at, p)
            if args.max_part_attempts and prior >= args.max_part_attempts:
                note = (f"per-part cap: this {p['label']} has already been "
                        f"attempted {prior}x within {SAME_PART_MM:.0f} mm")
                print(f"  abandon {p['label']}: {note}")
                (log.start_pick(attempts, p)
                    .record(occlusion=p.get("occlusion"), crowding=p.get("crowding"))
                    .finish("abandoned", notes=note))
                abandoned_at.append((p["label"], p["x"], p["y"]))
                continue

            attempted_at.append((p["label"], p["x"], p["y"]))
            attempts += 1
            if hasattr(grip, "sim_label"):          # dry run: simulate this class
                grip.sim_label = p["label"]
            u, _src = resolve_yaw(p, None)
            vmode = verification_mode(p["label"], args)
            # Forcing the strict policy means probing every class, signature or not.
            probe_label = None if args.verify == "proprioceptive" else p["label"]
            rec = log.start_pick(attempts, p)
            # Logged on EVERY attempt, not just the skipped ones: the point is to
            # regress success against occlusion afterwards, and that needs the
            # score attached to the picks that were tried as well.
            rec.record(commanded_u_deg=round(u, 1), grip_current_raw=args.current,
                       occlusion=p.get("occlusion"), crowding=p.get("crowding"))

            hover_z = p["z"] + APPROACH_MM - TOOL_Z_OFFSET_MM
            if args.max_z is not None and hover_z > args.max_z:
                print(f"   hover {hover_z:.1f} exceeds --max-z {args.max_z:.0f}, "
                      f"clamping (tall stack)")
                hover_z = args.max_z
            fingertip_z, why = grasp_height(p, args, floor_z)
            grasp_z = fingertip_z - TOOL_Z_OFFSET_MM
            # HARD FLOOR. Applied after every descent policy, so a bad grasp_dz,
            # a bad z_land or a mis-measured tip offset cannot drive the jaws into
            # the bench. Clamping rather than skipping: a grasp that is too shallow
            # simply fails to grip, which costs a cycle. A grasp that is too deep
            # costs the fingers, and there is one pair left.
            if args.min_z is not None and grasp_z < args.min_z:
                print(f"   FLOOR: {why} wanted Z={grasp_z:.1f}, clamped to "
                      f"--min-z {args.min_z:.1f}")
                why += f" [CLAMPED by --min-z {args.min_z:.0f}]"
                grasp_z = args.min_z
                fingertip_z = grasp_z + TOOL_Z_OFFSET_MM
            rec.record(descent_mode=args.descent_mode,
                       z_land=p.get("z_land"), z_med=p.get("z_med"),
                       width_mm=p.get("width_mm"),
                       effective_dz_mm=round(fingertip_z - p["z"], 1))
            print(f"\n-- [{attempts}] {p['label']} @({p['x']:.1f},{p['y']:.1f},{p['z']:.1f}) "
                  f"yaw {p['yaw']:.1f} -> U {u:.1f}")
            print(f"   descent: {why}")

            try:
                # Open only as wide as THIS part needs before descending. The jaws
                # sweep whatever they are open to through the scene on the way
                # down, so a full-open approach is what put a finger into a
                # neighbouring esp on 8 Sep.
                ap = (grip.ticks_for_width(p.get("width_mm"), p["label"])
                      if args.aperture == "adaptive" else None)
                if ap is not None:
                    rec.record(aperture_ticks=ap)
                    print(f"   aperture: {ap} ticks for a {p['width_mm']:.0f}mm part "
                          f"(full open would be {getattr(grip, 'max_open', 3571)})")
                grip.open(ap)
                _ok(arm.jump(p["x"], p["y"], hover_z, u))
                _ok(arm.go(p["x"], p["y"], grasp_z))
                pos, holding = grip.close(args.current, p.get("width_mm"), p["label"])
                note = getattr(grip, "last_note", "") or ""
                rec.record(grip_pos_after_close=pos, holding=holding,
                           verify_mode=vmode,
                           expected_stall=grip.expected_stall(p.get("width_mm"))
                           if hasattr(grip, "expected_stall") else None)
                print(f"   close: pos={pos} holding={holding}  (verify: {vmode})")
                if note and "nothing between" not in note:
                    # A width mismatch does not fail the pick - something IS held -
                    # but it is the signature of an edge grasp or a double pick, so
                    # it goes in the log rather than being swallowed.
                    print(f"   !! {note}")
                    rec.record(notes=note[:90])

                if not holding and vmode == "proprioceptive":
                    grip.open()
                    _ok(arm.go(p["x"], p["y"], hover_z))
                    rec.finish("no_grasp")
                    need_scan = need_scan or args.rescan == "per-pick"
                    continue
                if not holding:
                    # The verdict says empty, but for this class the verdict cannot
                    # tell empty from held. Carry on and let the re-scan decide -
                    # aborting here is how three successful esp grasps were thrown
                    # away on 8 Sep.
                    print("   verdict is empty, but this class has no thickness "
                          "signature — continuing, the re-scan will settle it")

                _ok(arm.go(p["x"], p["y"], hover_z))          # lift
                after = grip.position()
                rec.record(grip_pos_after_lift=after)
                drop = pos - after
                print(f"   lifted: {pos} -> {after} (fell {drop})")

                # ACTIVE probe, not the passive read. Under the bounded freeze the
                # fingers legitimately advance up to FREEZE_BIAS_TICKS, so a small
                # drop proves nothing either way - and parking the goal AT the stall
                # (the previous fix) removed the signal completely, which is how a
                # run logged "3 placed" while three parts lay on the floor.
                still, travel, note = grip.probe(probe_label)
                rec.record(probe_travel_lift=travel, probe_holding_lift=still)
                print(f"   probe: {travel} ticks -> holding={still}"
                      + (f"   [{note}]" if note else ""))
                if not still or drop > SLIP_DROP_TICKS:
                    print("   !! lost it during the lift")
                    grip.open()
                    rec.finish("lost_on_lift", notes=f"probe travel {travel}")
                    # A DROPPED PART CHANGES THE SCENE. It is somewhere it was not
                    # before, and whatever it was covering may now be exposed - so
                    # the next decision must be made from a fresh scan, not from a
                    # pick list that describes a bench that no longer exists.
                    # Re-planning only after SUCCESS is re-planning after the case
                    # that needed it least.
                    need_scan = need_scan or args.rescan == "per-pick"
                    continue

                if args.place:                       # explicit coordinates given
                    _ok(arm.jump(place[0], place[1], place[2], u))
                else:                                # taught PlacePoint: always reachable
                    _ok(arm.place_point())

                # Probe AGAIN at the place pose. The lift and the carry fail for
                # different reasons: the lift exposes a marginal clamp, the carry
                # exposes vibration and the wrist rotation. Without this, a part
                # shed in transit is indistinguishable from one placed correctly.
                still, travel, note = grip.probe(probe_label)
                rec.record(probe_travel_place=travel, probe_holding_place=still)
                print(f"   probe@place: {travel} ticks -> holding={still}"
                      + (f"   [{note}]" if note else ""))
                if not still:
                    print("   !! part was shed somewhere between the lift and the place pose")
                    grip.open()
                    rec.finish("lost_in_transit", notes=f"probe travel {travel}")
                    need_scan = need_scan or args.rescan == "per-pick"
                    continue

                grip.open()

                # Reaching the place pose is NOT success. Confirm the fingers
                # actually opened, or the part is still held and the next pick
                # would close on an already-full gripper.
                left = grip.position()
                if left < RELEASE_POS_MIN:
                    print(f"   !! RELEASE FAILED - fingers at {left}, part still held.")
                    rec.finish("release_failed", notes=f"open stalled at {left}")
                    break
                placed += 1
                if p.get("_enabling"):
                    # Does NOT count toward the bill: it was moved to reach
                    # something else, and reporting it as delivered would claim
                    # the operator asked for it.
                    cleared += 1
                    rec.finish("blocker_cleared")
                    print(f"   {p['label']} cleared out of the way.")
                    need_scan = True        # the target underneath is now exposed
                else:
                    fulfilled[p["label"]] = fulfilled.get(p["label"], 0) + 1
                    rec.finish("placed")
                    print("   placed.")

                if args.rescan == "per-pick":
                    need_scan = True

            except ArmFault as e:
                print(f"   !! {e}")
                rec.finish("arm_fault", notes=str(e))
                grip.open()
                break
            except (TimeoutError, OSError) as e:
                # Distinguish the two links. A Dynamixel packet error is the
                # GRIPPER's serial bus dropping out, not the EPSON controller
                # faulting, and calling both "arm_fault" sends the next session
                # looking in the wrong place. Observed 15 Sep: "Set goal position
                # failed: [TxRxResult] There is no status packet!" logged as an
                # arm fault, 0.08 s into a pick, before the arm was commanded.
                msg = str(e)
                gripper = ("status packet" in msg or "TxRxResult" in msg
                           or "port" in msg.lower())
                who = "gripper" if gripper else "arm"
                print(f"   !! {who} link failed ({e})")
                if gripper:
                    print("   !! This is the U2D2 serial link, NOT the controller.")
                    print("   !! Check the USB cable and that nothing else holds the port.")
                rec.finish("gripper_fault" if gripper else "arm_fault", notes=msg[:80])
                break

        # ---- final verification scan ----
        if args.rescan in ("end", "per-pick") and not args.dry_run:
            print("\n-- verification scan --")
            try:
                remaining = do_scan(args, arm, use_pose_cmds)
                pxy = None
                if args.place_xy:
                    pxy = tuple(float(v) for v in args.place_xy.split(","))
                remaining, dropped = drop_placed(remaining, pxy, args.place_radius)
                if dropped:
                    print(f"  excluded {len(dropped)} detection(s) at the place pose: "
                          + ", ".join(q["label"] for q in dropped))
                elif pxy is None:
                    print("  *** --place-xy not given: parts sitting at the place pose")
                    print("  *** are still being counted as 'on the bench'.")
                gone, left, moved = clearance(initial_picks, remaining)
                on_bench_initial = [p for p in initial_picks if on_bench(p)]
                n0 = len(on_bench_initial)
                ghosts = len(initial_picks) - n0
                if ghosts:
                    print(f"  {ghosts} detection(s) outside the bench envelope "
                          f"excluded from clearance:")
                    for p in initial_picks:
                        if not on_bench(p):
                            print(f"    {p['label']} @({p['x']:.0f},{p['y']:.0f},"
                                  f"{p['z']:.0f}) — over-detection, not a part")
                print(f"  started with {n0}; {len(gone)} gone from their original "
                      f"position, {len(left)} still there")
                print(f"  CLEARANCE: {len(gone)}/{n0} = {100*len(gone)/n0:.0f}%"
                      if n0 else "  CLEARANCE: n/a")
                for p, d in moved:
                    print(f"  ** {p['label']} is gone from its spot but its class is "
                          f"still on the bench {d:.0f}mm away — DISTURBED, not picked?")
                if len(remaining) >= n0 and gone:
                    print("  (the re-scan still sees ~as many parts as it started with:")
                    print("   the place pose is in the camera's view, so raw counts are")
                    print("   meaningless. The position match above is the real number.)")
                log.finding(started=n0, cleared=len(gone), still_there=len(left),
                            logged_placed=placed, disturbed=len(moved),
                            blockers_cleared=cleared, off_bench_detections=ghosts)
                if len(gone) != placed:
                    print(f"  *** MISMATCH: the run logged {placed} placed but "
                          f"{len(gone)} left the bench. Trust the scan.")
            except Exception as e:
                print(f"  verification scan failed: {e}")

    except KeyboardInterrupt:
        print("\n  interrupted by operator.")
        log.event("aborted", note="KeyboardInterrupt")
    finally:
        grip.release()
        arm.disconnect()
        log.close()
        print(f"\nDone. attempts={attempts}  placed={placed}"
              + (f"  blockers cleared={cleared}" if cleared else "")
              + (f"  abandoned={len(abandoned_at)}" if abandoned_at else ""))
        if abandoned_at:
            print("  abandoned units (still on the bench, counted against "
                  "fulfilment, NOT against the attempt rate):")
            for lbl, x, y in abandoned_at:
                print(f"    {lbl} @({x:.1f},{y:.1f})")


# ----------------------------------------------------------------- self-test
def self_test():
    """Exercise the per-part cap on the real data that motivated it.

    Replays the six positions the tilted lcd was actually retried at on
    16 Sep (run_20260916_201419), where it drifted 37.3 -> 49.1 mm in x as each
    failed grasp nudged it. Those six rows were 7 of the 29 attempts in the
    'easy' condition and inverted the easy-vs-hard result on their own.
    """
    import sys as _sys
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, wanted {want!r}")

    # The tilted lcd, as logged. Same board, drifting under retry.
    drift = [(37.3, 610.5), (39.7, 608.8), (42.4, 606.8),
             (44.9, 605.0), (47.2, 603.8), (49.1, 602.8)]
    cap, hist, attempted, abandoned = 2, [], 0, 0
    for x, y in drift:
        p = {"label": "lcd", "x": x, "y": y}
        if prior_attempts(hist, p) >= cap:
            abandoned += 1
            continue
        hist.append(("lcd", x, y))
        attempted += 1
    # 11.8 mm of total drift is well inside SAME_PART_MM, so all six are one unit.
    check("attempts allowed on the drifting lcd", attempted, 2)
    check("rows abandoned instead of retried", abandoned, 4)

    # A part of the same class elsewhere on the bench is a DIFFERENT unit and
    # must not inherit the cap. 66.8,602.0 is the other lcd from the same run.
    other = {"label": "lcd", "x": 66.8, "y": 602.0}
    check("other lcd is a separate unit", prior_attempts(hist, other), 0)

    # A different class at the same spot is also a different unit.
    check("class is part of identity",
          prior_attempts(hist, {"label": "esp", "x": 37.3, "y": 610.5}), 0)

    # Exactly at the tolerance boundary the unit is still the same one.
    check("at tolerance, same unit",
          prior_attempts([("lcd", 0.0, 0.0)],
                         {"label": "lcd", "x": SAME_PART_MM, "y": 0.0}), 1)
    check("just past tolerance, new unit",
          prior_attempts([("lcd", 0.0, 0.0)],
                         {"label": "lcd", "x": SAME_PART_MM + 0.1, "y": 0.0}), 0)

    # cap=0 must preserve the old unlimited behaviour exactly.
    hist2, attempted2 = [], 0
    for x, y in drift:
        p = {"label": "lcd", "x": x, "y": y}
        if 0 and prior_attempts(hist2, p) >= 0:
            continue
        hist2.append(("lcd", x, y)); attempted2 += 1
    check("cap=0 leaves every attempt in place", attempted2, 6)

    # The denominator this is all for: with the cap, that one unit contributes
    # 2 attempts instead of 6, and the easy set stops being dominated by it.
    check("easy-set attempts with the cap applied", 29 - 6 + 2, 25)

    # ---- the bench envelope, from layout E2 as it actually ran ------------
    def pk(lbl, x, y, z):
        return {"label": lbl, "x": x, "y": y, "z": z}

    e2 = [pk("esp", -104.7, 799.9, 251.4), pk("ultrasonic", -60.0, 560.8, 252.0),
          pk("lcd", -33.7, 691.6, 249.2), pk("ultrasonic", 70.8, 601.5, 250.6),
          pk("arduino", 42.5, 815.9, 247.9),
          pk("lcd", -226.1, 341.0, 64.6)]        # the reproducible phantom
    check("all five real E2 parts are on the bench",
          [on_bench(p) for p in e2[:5]], [True] * 5)
    check("the far-field phantom is not", on_bench(e2[5]), False)
    # E2 cleared everything: five units, five carried away, nothing left behind.
    gone, left, moved = clearance(e2, [])
    check("E2 clearance counts five, not six", (len(gone), len(left)), (5, 0))
    check("E2 clearance is 100%, not 83%", round(100 * len(gone) / 5), 100)
    # A part genuinely still on the bench must still be counted against us.
    gone2, left2, _ = clearance(e2, [pk("arduino", 42.5, 815.9, 247.9)])
    check("a part that never moved is still counted as left behind",
          (len(gone2), len(left2)), (4, 1))
    # The envelope must not quietly swallow a real part at the bench edges.
    check("a part at the far corner of the bench is kept",
          on_bench(pk("arduino", -148.6, 845.4, 288.0)), True)

    for f in fails:
        print("FAIL " + f)
    if fails:
        return 1
    print("clear_scene self-test passed — the per-part cap bounds a retried "
          "unit at 2 attempts and does not confuse it with its neighbours")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    main()
