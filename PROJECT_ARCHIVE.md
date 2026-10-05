# Picker-Bot — project archive

**Status:** archived 5 October 2026. Thesis submitted 25 Sep 2026; viva demonstration delivered.
Nothing here is under active development.

**Author of this work:** Faseeh Mohammed (M01088120), MSc Robotics, PDE4445, Middlesex University Dubai.
**Companion project:** `GripSense` — A. Mishra (M00983641). The gripper hardware, its travel-limit
calibration and its force characterisation belong to that dissertation and are **cited, not claimed** here.
This project owns the task layer: where to go, when to close, and what the result means.

**Read this file first.** It is written so that any person with no prior context can
understand what the system is, why each decision was made, what every file does, and what is
still open. Numbers quoted here were checked against the code and data in this repository.
Anything unverified is marked **[not measured]**.

---

## 1. What this is about

### 1.1 The problem

A university electronics lab lends out component kits. They come back incomplete and unsorted,
so the lab accumulates a **pile of loose, mixed microelectronic modules** — Arduino Unos, ESP32
boards, LCD panels, ultrasonic sensors. The next student needs *the specific parts their circuit
asks for*, out of that pile.

Robots are good at two adjacent jobs, and neither is this one:

- **Clearing a table.** Grab whatever is easiest, repeat until empty. The robot never needs to
  know *what* anything is.
- **Bin picking.** Ten thousand copies of one part, with a CAD model of it. Only the grasp pose
  is in question, never the identity.

Here the pile is **deliberately mixed**, and the robot must decide *what each thing is* before
deciding whether to pick it. **Telling the parts apart is the job.**

### 1.2 The reframe: a bill of quantities

The task is posed as **BOQ retrieval**: a parts list goes in (`arduino: 1, esp32: 1, lcd: 1`),
and a **fulfilment report** comes out — what was delivered, what was short, what was deliberately
left alone, and *why* in each case.

Two consequences, and both are the point:

1. **Classification becomes load-bearing.** Confusing an ESP32 for an LCD fails the task. In
   pile-clearing it costs nothing.
2. **Success is defined at the task level**, not the grasp level: "did the operator get what they
   asked for", not "did the fingers close".

Reporting what was *left alone* matters as much as what was fetched. Selective retrieval is half
"fetch the right things" and half "don't touch the rest".

### 1.3 The envelope (stated a priori, never quietly)

A scene is **in scope** when every part is separable by a **vertical lift alone**. That admits
parts evenly spaced, neighbouring and in contact, overlapping, stacked flat, or resting inclined
on a neighbour.

**Mechanically interlocked piles — where pin headers hook and lifting one part loads another —
are excluded before any data was collected and were never attempted.** Solving them needs a
physics simulator, a learned separation policy, or extra hardware; none was available. The claim
is that the boundary was *stated and held*, not that the true boundary was found.

### 1.4 The one-sentence contribution

A **cheap, training-free loop** — sort by depth, pick the topmost, verify with two independent
signals, re-scan after every attempt — that fulfils a parts list from a cluttered bench, with the
envelope and every failure mode measured and reported honestly.

### 1.5 The through-line

Every hard problem in this project was solved by **looking again with the camera**, not by adding
a sensor or a better gripper:

| Question | Answer |
|---|---|
| Where is the part? | Look. |
| Which one first? | Look at the depth. |
| Is anything on top of it? | Look at the outline. |
| Did the pick work? | Look again afterwards. |
| Did I knock something over doing it? | Look again afterwards. |

The gripper only ever reports finger position. **When the gripper's verdict and the camera's
disagree, the camera wins** — and §9.3 is the measured proof that this was the right call.

---

## 2. Research questions and what they returned

| | Question | Answer |
|---|---|---|
| **RQ1** | How reliably does single-view segmentation with aligned depth recover class, planar pose and top-surface height, and how does it degrade across arrangements, illumination and out-of-distribution objects? | **202/225 = 89.8%** found. Perfect on spaced and touching-but-not-overlapping scenes; **74%** on overlapping ones; **80%** in low light, which *also* invents parts (12 of the 13 over-detections). Unseen distractors: ignored, zero false alarms. |
| **RQ2** | Does topmost-first sequencing beat order-agnostic selection, and does it depend on re-planning after each pick? | **Null under re-planning** — both orderings cleared identically (3/3, 4/4, 4/4). Remove the re-scan and they separate sharply: depth-driven completed **2 of 2** paired runs, raster **0 of 2**, each stopped by the operator. *Re-planning is doing the work; depth ordering is what makes the loop safe.* |
| **RQ3** | Can the two verification signals separate grasp failure from perception failure, and what escapes both? | They separate, and **one-directionally**. Three grasps passed every gripper-side check while the board never left the bench — caught only by the re-scan. The inverse (camera says gone, gripper says held) did not occur. |
| **RQ4** | What governs the residual BOQ deficit — perception, graspability, or clutter regime? | **Clutter, acting *through* perception.** Fulfilment falls **93% → 57%** between measured conditions under an unchanged gripper. Failing grasps in crowded scenes carry contaminated geometry (an LCD measured 59.3 mm against a true 36 mm); the same classes succeed in isolation; a pre-registered descent test on an isolated raised part succeeded 5/5. |

Two answers are narrower than the questions anticipated. RQ2 returns a null under re-planning;
RQ3's blind spots are bounded rather than absent. Both are reported as such.

---

## 3. Hardware and the three links

| Piece | What | Owner |
|---|---|---|
| Arm | EPSON **VT6-A901S** 6-axis on an **RC8** controller, SPEL+ TCP receiver | this project (task layer) |
| Camera | Intel RealSense **D435i**, on the J6 flange | this project |
| Gripper | **Dynamixel XM430-W210-T**, current-based position control, via U2D2 | GripSense (A. Mishra) |
| Host | One Windows laptop runs all three | — |

**Camera and gripper sit on the same flange roughly 90° apart.** This is the geometric fact the
whole design turns on: **the pose that sees a part is not the pose that grasps it.** The arm
moves to a taught camera-down pose to scan, then a taught gripper-down pose to pick.

*Why the camera is on the wrist:* there was no other fixed mounting position available in the
lab. A fixed mount would be the obvious upgrade — it would remove the capture↔ready transit and
remove the (unmeasured) assumption that the arm returns to the capture pose repeatably.

### Three links, three distinct failure signatures

| Link | Carries | Fails as |
|---|---|---|
| Laptop ⇄ RC8, TCP `192.168.1.2:2001` | text commands, `OK` replies | **silence** (controller faulted) or **desync** (PC one reply ahead) |
| Laptop ⇄ D435i, USB 3 | aligned colour + depth frames | `Couldn't resolve requests` — **never run through a USB adapter** (§12.1) |
| Laptop ⇄ XM430, U2D2 serial | goal position, goal current, present position | `no status packet` → logged `gripper_fault`, *not* `arm_fault` |

---

## 4. The pipeline, end to end

```
 operator: "arduino 1, esp 1, lcd 1"
      │
 boq_fetch.py ── scan ─► inventory ─► plan (topmost within class) ─► subprocess ─┐
                                                                                 ▼
 clear_scene.py loop:  CAPTURE ─► scan.py ─► pose_seg.py ─► sort ─► gates ─► READY
      ▲                                                              │
      │      open(aperture) ► JUMP hover ► GO grasp_z ► close ► verdict ► lift
      │      ► probe ► PLACE/DISCARD ► probe ► open ► reopened? ► log
      └──────────────────────── re-scan after every pick ◄──────────────────────┘
                                      │
                        final verification scan ─► clearance by position-matching
```

### 4.1 Perception, in detail

One aligned frame → YOLOv8n-seg instance masks → per-mask pose. Per part, `pose_seg.part_pose`
emits:

| Field | What | Why it exists |
|---|---|---|
| `x, y, z` | centroid in base mm; `z` from the **nearest 8 mm depth band** | the top surface |
| `z_med` | median depth over the **whole mask** | on a tilted board the nearest band is the *raised corner*, which references the grasp several mm too high and closes the fingers on an edge or on air. **This was a measured failure, and `z_med` is the fix.** |
| `yaw`, `ax1/ax2` | long axis by PCA, converted in the **base frame** | a physical heading, not a pixel one |
| `aspect` | long/short extent | below 1.25 the long axis is ambiguous → yaw unreliable |
| `width_mm` | short-axis width | sets the jaw aperture |
| `z_land` | 10th percentile of a 15×15 depth patch under **each jaw**, worse of the two | **what the jaws will actually hit.** Not the min (one speckle would veto every descent); not the median (that averages an obstruction into the bench). |
| `tilt_deg`, `tilt_drop_mm` | least-squares plane over ~800 mask points | later found to be a *planarity residual*, not a tilt — see §11.4 |
| `crowding` / `occlusion` / `under_xy` / `nudge` | whole-scene pass | §4.2 |

**Masks, not oriented boxes.** The inherited detector produced OBBs; this work retrains on
instance masks. The reason is the *depth fusion*, not detection accuracy: height is sampled
**inside** each detection, and a box's corners contain bench pixels and slivers of neighbours —
worst precisely in the clutter the system targets. The representation is a consequence of the
sequencing claim.

**The ordering interaction worth knowing.** The same extremal height that makes a part rank
*first* under topmost-first is what *mis-locates* its grasp. That is not a coincidence; it is why
`z` and `z_med` are both emitted and compared.

### 4.2 Occlusion, measured on the outline

For each part: trace its outline (mask minus a 1-px erosion). For every other part, dilate its
mask by ~4 px. `contact` = share of my outline inside that dilated neighbour.

- **crowding** = sum of contact over all neighbours (capped at 1).
- **occlusion** = the same sum, but **only over neighbours more than 4 mm higher** (by `z_med`).

So occlusion asks *"how much of my edge has something sitting above it"*, not *"am I touching
anything"*. One side of four ≈ 0.25, pinned two sides ≈ 0.5, ringed ≈ 1.0.

**Why perimeter, not area.** An area fraction does not scale with burial: a 100×100 px part
sharing one whole edge scores ~0.09 by area whether barely touching or half covered, because only
a thin strip is ever within reach of the neighbour. Perimeter contact means the same thing across
part sizes, so a single threshold is meaningful. (Also, the covered area is literally not in the
image to be measured.) **Caveat: 0.35 is a design threshold between "one side" and "two sides"
— it was never fitted to data, and outline-vs-area was never compared head to head.**

Crowding and occlusion are deliberately **separate**: a part can be surrounded yet perfectly
graspable because everything around it is lower.

**Nudge** (which way to push a pinned part toward free space) is computed and drawn on the
overlay but **never executed**. Showing the decision was the contribution; performing it needs
new arm motions and new trials.

### 4.3 Verification — why "placed" costs four checks

An earlier build logged **3 placed with three parts on the floor**. The goal-freeze that cured a
release wedge also removed the position error, and with it every passive sign of a lost part. So
success is no longer *inferred from a reading*; it is a conjunction:

1. **close verdict** — stall position above `HOLD_THRESHOLD`
2. **active probe after the lift**
3. **active probe at the place pose** (lift exposes a marginal clamp; the carry exposes vibration
   and wrist rotation — different failure modes)
4. **confirmed reopen** (`position ≥ 3400`), or the next pick closes on a full gripper
5. …plus the **re-scan**, as the independent fifth witness

**The active probe** re-commands a close for 0.5 s and measures finger travel. A held board
arrests the fingers; free fingers run on toward the linkage friction floor. Measured: **482 ticks
when the part was gone, 0–1 ticks when held.** No overlap. It is deliberately an *experiment*
rather than an observation, because under the bounded freeze the passive reading is near the goal
in both cases.

**Slip detection here is position-based** (`SLIP_DROP_TICKS = 150`, ≈ 5.8 mm of extra closing
during the lift) plus the probe. GripSense's `SlipWatch` is **current-based** (continuous, 200 Hz,
triggers on a torque drop). Different signal, different mechanism, built independently.

### 4.4 Why the re-scan runs after *every* pick

Three reasons, and all three actually happened:

1. **A bad grasp can look perfect to the gripper** — the three false successes (§9.3). Nothing
   "slipped", so a slip-triggered re-scan would never have fired.
2. **A *successful* pick invalidates the plan** — lifting one part uncovers another. Without a
   re-scan, "buried" means *never*; with one, it means *not yet*.
3. **You can succeed and still make a mess** — one trial cleared its target and dragged a
   neighbouring ESP32 **50 mm**. The pick succeeded; the plan afterwards was still wrong.

Measured value: per-pick re-scanning was worth **50 percentage points of clearance** (4/4 vs 2/4)
on the ablation layout.

### 4.5 Clearance is measured by position-matching, not counting

The taught place pose sits **inside the camera's field of view**, so counting what the re-scan
sees is meaningless — a run that cleared three parts once reported *"three still on the bench"*.
Instead each **original** part is matched against the re-scan: still within tolerance of where it
started = left behind; gone = cleared. This also catches a part **displaced** by a neighbour's
pick, which a headcount never could.

**Clearance inherits the detector's recall:** a detection miss is indistinguishable from a
removal. Stated as a limit, not hidden.

---

## 5. Calibration — four independent fits, each reporting its own residual

| # | What | Method | Result |
|---|---|---|---|
| 1 | **Camera → robot base** (`handeye.json`) | Click table markers in the aligned colour frame → median of a 7×7 depth patch → deproject to camera metres. Touch each marker with the tool tip, read XYZ from RC+ Jog & Teach. Solve the best-fit rigid transform by **SVD (Kabsch/Umeyama)** with a **reflection guard**. | **2.67 mm RMS**, 11 pairs, scale 0.994 |
| 2 | **Wrist yaw** (`yaw_calib.json`) | Hover, nudge U until the open jaws straddle the short width, record (vision yaw, U). Fit `U = sign·yaw + offset` for **both signs**, circular mean on doubled angles (180°-periodic). | sign **+1**, offset **−88.8°**, **1.5° RMS**, 5 pairs |
| 3 | **Jaw gap ↔ ticks** (`aperture.json`) | Command an open position, **measure the gap with calipers**, least-squares line. | **25.983 ticks/mm**, intercept 758.4, **0.12 mm RMS** |
| 4 | **Held vs empty** (`grip_profiles.json`) | Close on air, close on each class, repeat. | empty **708**, threshold **830** |

**Design notes worth keeping:**

- **A 3D rigid transform, not a homography.** Phase 1 (`pickerbot_lib/calibration.py`) used a 2D
  homography. That has no height — and height is the entire sequencing signal.
- **Scale is a sanity check, never applied.** 0.994 says the depth stream and the jog display
  agree on units; forcing scale = 1 would hide a bad touch.
- **Read `R` to understand the geometry:** `R[2][2] = −0.9995`, so the optical axis is within ~2°
  of straight down and camera-z is base-z negated. That is why *"largest base Z"* = *"closest to
  the camera"* = *"on top"*.
- **Yaw is fitted for both signs** because with the gripper pointing down, tool-Z is anti-parallel
  to base-Z and can reverse the sense of rotation. If the recorded yaws are only ~0° and ~90°
  apart then +90 ≡ −90 (mod 180) and the sign is **undetermined** — the script refuses to save.
- **Near-square parts are excluded** from yaw calibration: an undirected long axis makes their
  yaw meaningless.
- **Aperture is measured with calipers, not by closing on known parts.** Closing is contaminated
  by exactly the friction and compliance effects under investigation; a free-gap measurement never
  touches the object.

### 5.1 Precision is not accuracy

Perception noise floor (`noise_floor.py`, 15 frames, nothing touched, all 4 parts tracked 15/15):
**σx = σy = 0.07 mm, σz = 0.39 mm, σyaw = 0.13°.**

Against a hand-eye RMS of 2.67 mm, perception is **~40× more precise than the calibration is
accurate**. Placement error is dominated by **hand-eye and the arm, not by vision**. A better
camera or model would tighten a group that is already tight; it would not move the group.

### 5.2 The two Z frames — a known debt

- Vision `z` = the part's **surface height** in the base frame.
- Commanded Z = the **Tool-1 point**.
- `TOOL_Z_OFFSET_MM = 70` converts for the hover: `hover_z = z + 80 − 70`.
- **Measured 12 Sep:** jaws touch bare bench at commanded **Z = 213.3**, while vision puts that
  bench at `z_land` ≈ **243.6**. A **30.3 mm** frame gap.

The nominal 70 is wrong about the fingertips. The per-class `dz` values absorb the discrepancy
and `--min-z 213` is the hard floor. **Never compare a vision `z` to a commanded `Z` directly.**
Fixing both together, off the robot, and re-deriving every `dz` is the correct repair.

---

## 6. The Epson link

### 6.1 Controller side — `Epson/Pickerbot_Receiver/Pickerbot_Receiver/Main.prg` (SPEL+, 197 lines)

Init: `Motor On`, **`Power Low`**, `Speed 10`, `Accel 10,10`, **`Tool 1`** (the string-calibrated
TCP). Then `SetNet #201, "0.0.0.0", 2001, CRLF` → `OpenNet As Server` → `WaitNet`. Loop:
blocking `Input #201` → `ParseStr` on spaces → independent `If` blocks.

| Command | Does | Why |
|---|---|---|
| `jump x y z u` | `Go Here+Z(50)` → `(x,y,z+50,u)` → `(x,y,z,u)` | **Three PTP `Go`s, not SPEL `Jump`.** `Jump` is a SCARA gate motion and raises **error 4034** on the 6-axis VT6. Lift–translate–descend reproduces the collision-safe shape. |
| `go x y z` | straight PTP, keeps U | the final descent and the lift |
| `capture` | `Go cap` | taught point, camera-down |
| `ready` | `Go gripperdown` | taught point, gripper-down |
| `place` | `Go PlacePoint` | taught, guaranteed reachable |
| `discard` | `Go discard` (Point 8, x = −251) | where blockers go — **outside** the counted workspace |
| `speed n` | clamps 5–50, sets `Speed`/`SpeedR` | **`Power Low` overrides it**: 15.73 s round trip at 10 *and* 40. Speed tuning was abandoned. |
| `standby`, `move`, `pick` | housekeeping / legacy | |

**Three properties of this file that were each paid for:**

1. **Wrist orientation is a hidden precondition.** The receiver sets X, Y, Z and **U only** — V
   and W are *inherited from wherever the arm already is*. Only a **taught point** carries full
   6-DOF, which is why `capture` and `ready` exist and why the capture pose (Z≈673, taught
   camera-down) is **unreachable** gripper-down.
2. **Every branch sets `ok = 1` and replies `OK`; a final `If ok = 0` replies `ERR unknown
   command`.** Before this, an unmatched command fell through to `Loop` and waited for the next
   `Input` **without replying** — the PC blocked in `recv()` forever. Observed 26 Sep: a run died
   mid-pick with a part in the jaws. **A typo must produce an error, not a deadlock.**
3. Errors **2902 / 2910** (network drop) re-open the server socket and wait.

### 6.2 PC side — `pickerbot_lib/sender.py`

Plain blocking TCP, one line out (`CMD x y z u\r\n`), one line in. **Lock-step request/response is
the entire protocol.**

- **`_drain()` runs before every command.** One stray `OK` in the buffer makes every later `recv`
  return the *previous* command's reply, so the PC runs one step ahead of the robot. Observed
  8 Sep: the gripper closed while the arm was still descending, and a CAPTURE scan grabbed its
  frame before the arm had moved (a sideways view of the room, 0 parts, while the run logged
  "3 placed"). **Stale replies survive across connections.** Draining makes the link self-healing
  and warns when it fires.
- **`sock.settimeout(30)`** — a faulted controller stops *without replying*, and a blocking `recv`
  cannot be interrupted with Ctrl-C on Windows.
- Non-`OK` reply → `ArmFault` → logged against the attempt, run stops cleanly.
- **CRLF line endings are mandatory in `Main.prg`** — an LF-only file made the RC+ compiler raise
  *Error 3100, Line 1 syntax error*.
- **RC+ compiles its editor buffer, which can be older than the file on disk.** "I rebuilt it"
  does not establish what the controller holds. `dday/which_build.py` asks the controller itself,
  with a 6 s timeout, and **treats silence as the answer** (= old build).

---

## 7. The gripper layer

`pick_one.py::Gripper` drives GripSense's raw driver (`gripper_settings`). The XM430 runs
**current-based position control**: a goal position plus a *current ceiling* = a force ceiling.
That ceiling, and readable position, are what make a grasp measurable.

| Action | Sequence | Why |
|---|---|---|
| `open(ticks)` | velocity **480**, 120 raw, settle → retry → last resort **torque off** so the sponge springs the fingers apart | Opening must exceed the grip current or the fingers move **0 ticks**. The old `open()` was broken by *our own code* — a 50 raw "relax" step (half the configured minimum) and velocity 40 where the calibration reopens at 480. The hardware was never the problem. |
| `close(current)` | velocity 60 → set current → goal `min_open` → `_settle` → **bounded freeze** → `classify` | slow onto the object |
| `_settle` | wait 0.20 s, then require **3 consecutive stable reads** | comparing two reads 60 ms apart returns instantly — the servo has not started moving yet, so the "stall" reported is the *starting* position |
| `_hold` | park goal **`FREEZE_BIAS_TICKS = 25`** inside the stall | Goal at `min_open` leaves a ~1300-tick error → full current for the whole carry → the visco-elastic sponge creeps (~90 ticks, *the same at 80 and 90 raw — duration, not force*) and wedges beyond what 120 raw can undo. Goal *at* the stall removes the error — and with it **every passive sign of a lost part**. 25 ticks keeps both clamp force and observability. |
| `probe()` | re-command a close for 0.5 s, measure travel; re-park the goal in a `finally` | see §4.3 |

### 7.1 The measured numbers (100 raw, calibration 3541 / 655, new fingers + sponge)

| | ticks |
|---|---|
| empty stall | **708** (spread 1 tick over six closes) |
| `HOLD_THRESHOLD` | **830** |
| ultrasonic | 901 (spread **59**) |
| lcd | 1143 (spread 2) |
| esp | 1237 (spread 4) |
| arduino | 1859 (spread 9) |

**The threshold rule:** a threshold goes **below the lowest *real* grasp, not below the highest
failed one**. 830 clears empty by 122 ticks and sits 48 below the weakest hand-measured real
grasp (an ultrasonic at 878). Erring high costs a cycle and logs an honest `no_grasp`; **erring
low writes a false success into the trial data**, which is far more expensive. (It was briefly
lowered to 750 to admit an ultrasonic that stalled at 796 — the next run repeated it at 789,
logged "placed", and the verification scan found the part still on the bench.)

**The ultrasonic is the binding case** everywhere: two cylindrical cans give the jaws *line*
contact, so it rolls and settles deeper under load — hence a 59-tick spread against 2–9 for
flat-faced parts, and 30 of its 54 closes landing within 70 ticks of the threshold (against 0 of
180 for the other three classes).

### 7.2 The single mistake that caused most of the gripper mysteries

**The operating current is 100–120 raw.** GripSense's config declares `current.min: 100`,
`current.max: 120`; its calibration probes at 110. **Every run before 12 Sep used 80–90 raw —
below the floor.** Empty fingers commanded to `min_open`: 90 raw → stall **1646** (a *gap*, no
contact); 100 → 721; 110 → 690; 120 → 671. That 1646 gap is **wider than an esp, an lcd or an
ultrasonic** — so those parts were never gripped at all. The earlier explanation ("the sponge
conforms around a thin board and hides it") was wrong: **there was no contact to hide.**

Consequence: every pre-12-Sep conclusion about thin classes being "proprioceptively invisible" is
an artefact. After the fix, **all four classes separate from empty by ≥ 193 ticks**, and
`PROPRIOCEPTIVE_CLASSES` expanded from `{arduino}` to all four.

### 7.3 `classify()` asks two questions

1. **Is anything between the fingers?** stall > threshold. Within `MARGINAL_BAND = 70` → flagged
   MARGINAL, not rejected.
2. **Is it the size vision said it was?** `expected = 25.983·(width_mm − 10.3) + 758.4`. Error
   beyond `STALL_TOL_TICKS = 120` → **WIDTH MISMATCH**: an edge grasp, two parts at once, or a
   mislabelled detection.

Sponge compression is a **constant 10.3 mm** (range 9.6–11.0) across a 16–53 mm range of parts,
which is what makes a stall predictable from a *vision-measured* width to within 20 ticks
(0.8 mm) — including for classes never individually characterised. **Vision supplies the width;
the servo independently confirms it.**

`WIDTH_SANITY_MM` clamps a physically impossible reading. Measured 12 Sep: a merged mask reported
an arduino at **78.6 mm** (true short axis 53.4). Acting on it would have opened the jaws to a
107 mm span, putting a finger 54 mm from centre — exactly where the neighbouring esp was sitting.

### 7.4 Width-adaptive aperture

The jaws open only to `width + 2 × 8 mm`, not fully. The jaws sweep whatever they are open to
**through the scene on the way down**; a full-open approach is what put a finger into a
neighbouring esp on 8 Sep and **snapped a printed finger clean off**. On a 62 mm board the
adaptive aperture is ~31 mm narrower than full open. The measurement was already being taken for
something else, which makes this the cheapest of the three descent guards.

---

## 8. Codebase map

Root is `C:\Users\10463\MB_faseeh\REPO\picker-bot`. **`pde4445-dev/` is the thesis work**;
everything outside it is phase 1 (PDE4435) or support.

### 8.1 `pde4445-dev/` — the live system

| File | Lines | Status | What it does |
|---|---|---|---|
| **`boq_fetch.py`** | 538 | **ACTIVE — the front door** | Parses the bill (`boq.txt` or `--interactive` typed input, with aliases: `esp32→esp`, `hcsr04→ultrasonic`). Scans, filters far-field phantoms via `on_bench`, prints the **inventory the robot can actually see**, asks what is needed, plans topmost-first *within each requested class*, reports BLOCKED vs PARTIALLY COVERED vs MARGINAL detections, then hands the selection to `clear_scene.py` as a **subprocess** (`PICKER_PICK_LIST` env var). Never reimplements the pick. `--self-test` runs clean. |
| **`clear_scene.py`** | 1510 | **ACTIVE — the executor** | The autonomous loop: scan → order → gates → pick → verify → re-scan. Owns the descent policy (`fixed` / `two-regime` / `clearance`), the occlusion and jaw-obstruction gates, enabling moves, the per-part attempt cap, per-class grip current, the bill accounting, and the final clearance measurement. All ablations are command-line flags. `--self-test` replays the real drifting-LCD data. |
| **`pick_one.py`** | 757 | **ACTIVE — grasp logic + constants** | One hand-stepped pick (how `GRASP_DZ_MM` was discovered), and the `Gripper` class every other script imports: current ladder, corrected settle rule, bounded freeze, active probe, `classify`, width-adaptive aperture, sanity clamps. **The authoritative header for every tick constant.** |
| **`pose_seg.py`** | 373 | **ACTIVE — the geometry** | Mask → robot pose. `part_pose`, `occlusion_pass`, `pickability`, `land_z`, `deproject`, `cam_to_robot`, `load_handeye`. **Single definition of how a mask becomes a robot coordinate** — live and offline both import it, so they cannot drift apart. |
| **`scan.py`** | 271 | **ACTIVE — frame source** | Live single-frame grab (depth ladder: 848×480@30 → 640×480@30 → 848×480@15 → 640×480@15; **colour locked at 1280×720**) or `.db3` playback. Builds the overlay (masks, pick order, long axis, cyan landing crosses, occlusion score, orange nudge arrow) and writes `pick_list.json`. Imports all geometry from `pose_seg`. |
| **`runlog.py`** | 292 | **ACTIVE — the evidence** | One CSV row per attempt plus a per-session JSON snapshot of the configuration. Fixed outcome vocabulary: `placed, blocker_cleared, no_grasp, lost_on_lift, lost_in_transit, release_failed, arm_fault, gripper_fault, skipped, abandoned, aborted`. Success-rate denominators exclude skips, faults and aborts. |
| **`analyse_runs.py`** | 989 | **ACTIVE — the analysis** | Pools `runs/*.csv` into the Chapter 5 tables. **Exists to enforce one rule: never pool across configuration epochs.** Writes `analysis/runs.md` with every number derived, not typed. |
| **`eval_vision.py`** | 745 | **ACTIVE — the vision battery** | Scores all recordings against `captures/manifest.csv` ground truth, with no robot: per-scene, per-class, per-condition, confidence sweep, inference time, ordering divergence, overlays. `--self-test` validates scoring without the model. |
| **`pnp_gate.py`** | 417 | **ACTIVE** | Assigns a layout's **condition by measurement before the arm moves**, so difficulty is never a label applied by eye afterwards. Writes `gates/gates.jsonl`. |
| **`order_check.py`** | 196 | **ACTIVE** | Asks whether a layout has the **statistical power** to test the ordering claim — i.e. whether raster order actually violates a measured support relation here. Moves nothing. Built after 16 Sep L3 silently became a null because raster happened to be a valid top-down order. |
| **`noise_floor.py`** | 308 | **ACTIVE** | Opens the camera **once**, grabs N frames of a static scene, reports pose repeatability. Separate from `capture_scene` because 10 open/close cycles in a minute broke the Windows UVC stack. |
| **`capture_scene.py`** | 297 | **ACTIVE** | Records tagged `.db3` scenes in bulk with a manifest row each, so the battery can run all week at a desk. Reads the **stream profile** for `depth_res` (reading the array post-`align` recorded the aligned size and produced a wrong conclusion once). |
| **`aperture_calib.py`** | 206 | **ACTIVE** | Fits jaw gap (mm) → ticks with calipers. **Re-run after any finger or calibration change.** |
| **`grip_verify.py`** | 236 | **ACTIVE** | Measures empty vs loaded stalls for the current calibration. **Re-run after every recalibration** — positions are only meaningful relative to the calibration they were measured under. Note: it suggests a threshold from whatever class is in the jaws, so always re-derive across all four. |
| **`bench_plane.py`** | 390 | **ACTIVE (analysis)** | Fits the bench as a **plane** rather than one number. Gradient −0.94 mm per 100 mm in y (≈2.6 mm across the working range, 0.54°); all four classes agree on the sign independently. Shows `--min-z` as a single contact height is an approximation. |
| **`calib_capture.py`** | 73 | **ACTIVE** | Hand-eye, camera side: click markers, log camera-frame points. |
| **`calib_solve.py`** | 81 | **ACTIVE** | Hand-eye solver (Kabsch/Umeyama + reflection guard), per-marker residuals, writes `handeye.json`. |
| **`yaw_calib.py`** | 174 | **ACTIVE** | Fits `U = sign·yaw + offset` properly, with residuals, both signs, and a refusal to save when the sign is undetermined. |
| **`test_descent.py`** | 629 | **ACTIVE — regression** | Desk test: simulated servo + synthetic scene, three full dry runs including forced drops. Enforces **"no first-time code paths on a lab day"**. |
| **`gripsense_path.py`** | 49 | **ACTIVE — helper** | Locates the side-by-side GripSense repo and puts its `Lib` on `sys.path`. Override with `GRIPSENSE_REPO`. |
| **`probe_escalate.py`** | 241 | **ONE-OFF measurement** | Tested whether more current could separate thin parts from air. Answer: the question dissolved once 100 raw was used (§7.2). |
| **`release_test.py`** | 146 | **ONE-OFF measurement** | Timed three release strategies against the "cannot let go" failure. |
| **`arm_check.py`** | 32 | **UTILITY** | One safe move; confirms the link, motion, and that V/W survive. Run before the loop. |
| **`pickplace_nogripper.py`** | 91 | **UTILITY** | Pick-place motion rehearsal with no gripper, gripper-down orientation. |
| **`yaw_jog.py`** | 73 | **SUPERSEDED** by `yaw_calib.py` (this was the by-eye version). |
| **`grip_class.py`** | 270 | **SUPERSEDED** — per-class profile learning, two-laptop era. |
| **`grip_tune.py` / `grip_tune_raw.py`** | 125 / 157 | **SUPERSEDED** — the sweeps that produced the 100–120 raw answer. Kept for the write-up. |
| **`gripper_api.py` / `gripper_check.py`** | 118 / 33 | **SUPERSEDED** by the raw-driver `Gripper` in `pick_one.py`. Carries `GRIP_CURRENT 110` — **do not use as a preflight.** |
| **`gripper_server.py` / `net_gripper.py`** | 167 / 84 | **OBSOLETE** — two-laptop TCP bridge. Carries constants from a dead calibration (`HOLD_THRESHOLD_TICKS 1230`, `min_open −53`). |
| **`orchestrate_pick.py`** | 108 | **OBSOLETE** — pre-dates gripper-down orientation and the measured threshold. |
| **`yolo_test.py`** | 116 | **LEGACY** — steps a recording through the old OBB model to build a training set. |
| **`3d_Points.py`, `detect_items.py`, `measure_tilt.py`, `pipeline.py`, `plane_subtract.py`, `read_frame.py`** | 27–61 | **SCRATCH** — July exploration scripts against a single hardcoded `.db3` (`20260719_123818.db3`). Depth-threshold blob finding, plane subtraction, first 3D deprojection. Historical only. |
| `dday/which_build.py` | 89 | **ACTIVE (viva prep)** | Asks the controller which `Main.prg` it is running. Cannot hang. |
| `dday/speed_test.py` | 126 | **ONE-OFF** | Measured what raising PTP speed buys. Answer: nothing — `Power Low` clamps it. |
| `report/make_f1.py`, `report/make_f2.py` | 101 / 102 | **ACTIVE** | Regenerate thesis figures from archived photos and captures. |

### 8.2 Outside `pde4445-dev/` — phase 1 and support

| Path | Status | What |
|---|---|---|
| `pickerbot_lib/sender.py` | **ACTIVE** | The TCP client used by everything (§6.2). |
| `pickerbot_lib/calibration.py` | **SUPERSEDED** | Phase-1 **homography** (pixel → table plane). No height — replaced by the 3D rigid transform. |
| `pickerbot_lib/detection.py`, `config.py`, `__init__.py` | **LEGACY** | Phase-1 YOLOv8-**OBB** inference and config. |
| `pickerbot.py`, `detect_and_classify.py`, `teleop_mouse.py` | **LEGACY** | Phase-1 (PDE4435) entry points. |
| `tools/calibration_clicker.py`, `camera_alignment.py`, `sort_and_tag_pixels.py` | **LEGACY** | Phase-1 calibration helpers. |
| `legacy/cv_discovery.py`, `keras_inference.py`, `main_orchestrator.py` | **LEGACY** | Pre-YOLO contour method and the first orchestrator. |
| `Epson/Pickerbot_Receiver/` | **ACTIVE** | The RC8 project. `Main.prg` is the receiver (§6.1); `robot1.pts` holds the taught points (`cap`, `gripperdown`, `PlacePoint`, `discard` = Point 8). |
| `models/best.pt`, `models/legacy.pt` | **LEGACY** | Phase-1 OBB weights. **The thesis model is `pde4445-dev/best.pt`.** |
| `docs/blog/` | reference | Three write-ups: building the gripper API, the vision-to-robot lab day, SAR posts. |
| `cad/`, `data/` | reference | Fixtures and phase-1 calibration data. |
| `README.md` | **STALE for PDE4445** | Describes the **phase-1** system (OBB + homography). Accurate for PDE4435, misleading for this thesis. |

### 8.3 Documentation inside `pde4445-dev/`

| File | Lines | What |
|---|---|---|
| `LAB_LOG.md` | 1096 | **The primary record.** Every lab session, every measurement, every failure, with the reasoning at the time. If a number here is disputed, this is where it was born. |
| `PROJECT_STATE.md` | 497 | Living record of decisions and measured constants. **Warning: the gripper table near the top is the 80–90 raw era (HOLD 1977/1816) and is superseded by §7.1.** |
| `ROADMAP.md` | 231 | Plan to submission, milestones, evaluation design. |
| `FINAL_LAB_DAY.md` | 312 | 19 Sep runbook and preflight checklist. |
| `SATURDAY.md` | 217 | 12 Sep runbook. |
| `analysis/runs.md` | 244 | Generated by `analyse_runs.py`. **Do not hand-edit.** |
| `report/report.tex`, `report/dissertation.tex` | — | The submitted document (IEEEtran, 2-column). `dissertation.tex` is the later edit. |
| `report/TECH_WALKTHROUGH.md` | — | Zero-to-hero technical walkthrough (the long-form version of §4–§7). |
| `report/PRESENTATION_README.md` | — | The 12-slide viva brief + backup slides + Q&A appendices. |
| `report/PRESENTATION_README_v1_14slides.md` | — | The earlier 14-slide version. |
| `dday/README.md` | 38 | Viva-day workspace notes. |

### 8.4 Data assets

| Path | Size | What |
|---|---|---|
| `captures/` | the bulk of ~60 GB | **45 scored `.db3` recordings** — scattered 10, regular 10, bunched 10, adversarial 5, low light 10 — plus `static/` for the noise floor. Each has a `manifest.csv` row giving its condition and ground-truth contents (5 modules: 2 arduino, 1 esp, 1 lcd, 1 ultrasonic). **The bench was re-arranged between every capture**, so these are 45 *independent layouts*, not repeats. |
| `runs/` | — | 139 CSVs (102 with real rows, 483 logged attempts) + `sessions.jsonl` (166 session configs) + `discarded/` (21 header-only or desync-invalid runs, **prefixed and kept, never deleted**). |
| `scans/` | 392 PNGs | One overlay per scan, never overwritten — the per-run evidence that the detections behind a result were sane. |
| `gates/gates.jsonl` | 28 | Pre-run condition measurements from `pnp_gate.py`. |
| `vision_eval/` | — | Battery output: `per_scene.csv`, `per_class.csv`, `per_condition.csv`, `sweep.csv`, `ordering.csv`, `repeatability.csv`, `detections.json`, `summary.md`, `overlays/`. |
| `best.pt` | — | **The thesis model.** YOLOv8n-seg, dated **8 Aug 2026**. |
| `*.json` | — | `handeye`, `yaw_calib`, `aperture`, `bench_plane`, `noise_floor`, `grip_profiles`, `grip_profiles_100raw_submitted` (backup of the submitted state), `pick_list`, `boq_plan`, `boq_pick_list`. |

**Model provenance / leakage:** `best.pt` is dated 8 Aug 2026; all 45 evaluation recordings were
captured 8, 12 and 15 Sep. The test data did not exist when the model was trained. It is the same
bench and the same physical parts, so it is a **fair test but not a hard one** — stated in the
report.

---

## 9. Results

### 9.1 Vision battery — 45 recordings, 225 parts, conf 0.55, no robot

| Condition | Scenes | Expected | Found | Recall | Over-detections |
|---|---|---|---|---|---|
| scattered | 10 | 50 | 50 | **100%** | 0 |
| regular | 10 | 50 | 50 | **100%** | 0 |
| **bunched** | 10 | 50 | 37 | **74.0%** | 1 |
| adversarial | 5 | 25 | 25 | **100%** | 0 |
| **low light** | 10 | 50 | 40 | **80.0%** | **12** |
| **total** | **45** | **225** | **202** | **89.8%** | **13** |

By class: arduino 81/90 (90.0%), esp 36/45 (80.0%), lcd 42/45 (93.3%), ultrasonic 43/45 (95.6%).
Inference: median **347 ms** (first inference, 7066 ms, excluded as warm-up).

**How to read these.** This is **recall ("found rate")**, not accuracy — over-detections are
reported separately. It is **count-based, not IoU**: a scene can score full marks with masks on
the wrong objects, which is why every overlay is saved and spot-checked. 100% over 50 parts means
*"about 93% or better"*, not perfect.

**The dominant perception failure is same-class instance separation.** 9 of the 13 bunched misses
are Arduinos — the only duplicated class — because two identical adjacent boards merge into one
instance. The confidence sweep proves this is **not** a threshold problem: dropping to 0.25
recovers only 3 of 13 while inventing 18 more. **The missing parts are never proposed at any
threshold.**

**Clutter and darkness fail differently.** Clutter *misses* parts; low light misses parts **and
invents them** (12 of the 13 over-detections).

**conf 0.55 is defended by the curve, not by impression:** 0.45 and 0.55 give identical recall
and 0.55 halves the over-detections; 0.75 buys the last over-detection for 4 more misses.

### 9.2 Ordering

- **0 of 45** recordings produced identical topmost-vs-raster sequences (mean Kendall τ = −0.07).
  **Divergence is not power:** sorting a flat scene by height sorts by *component thickness*, so a
  trial there compares raster against a shuffle.
- Under a **12 mm height-spread gate**, only **20 of 45** scenes can test the claim. Scattered and
  regular span 3.9–8.4 mm and were excluded on that basis.
- **With per-pick re-scanning: 3/3, 4/4, 4/4 — a null.** Reported as one.
- The separation is **in the path, not the outcome**: on the two-support layout the raster arm
  logged two false successes and dragged a neighbouring ESP32 **50 mm**.
- **Remove the re-scan and they separate sharply:** depth-driven completed **2 of 2** paired runs,
  raster **0 of 2**, each stopped by the operator. (A third raster run on an *earlier
  configuration* ended the same way; the two configurations are not pooled.)
- **The confound, stated:** the e-stop is an **operator judgement and the operator was not
  blinded** — same person built the layouts, knew which policy was running, and expected the
  raster arm to misbehave. The defensible claim is *"the baseline reached states the operator
  judged unsafe"*, never *"the baseline collided"*. What is **not** operator-dependent is in the
  logs: raster took an LCD while an ESP32 rested on it (the violation `order_check` flagged
  *before* the run), the LCD slipped on the lift, the ESP32 was displaced, and the next pick was
  aimed at a position the stale plan still believed.
- A stronger design would use an **automatic abort criterion** (refuse any pick whose target moved
  more than N mm since the scan that planned it). That is future work.
- **Also measured:** the static occlusion gate does **not** substitute for re-scanning. With
  `--rescan end` it can only score the opening frame, and it did not prevent the 16 Sep stop.

### 9.3 Verification — the strongest result

**Three grasps passed every gripper-side check while the board never left the bench.**

- Stalls **1123, 1159, 1166** against an LCD profile of **1143** — so something really was
  between the fingers, and still was at the drop-off.
- Both probes read **≤ 1 tick** — textbook "held".
- The parts had moved **0.7, 2.5 and 6.7 mm**. They were still on the bench.
- Only the re-scan caught them. Nothing "slipped", so a slip-triggered re-scan would never have
  fired.

**Do not offer a mechanism for these.** The log records *how far apart the fingers are*, not what
they are touching. The honest statement is: **the two checks disagreed, and the camera was right.**
"It gripped a corner and the board slipped out" is refuted by the drop-off probe, which found the
fingers arrested.

**Why raising the threshold does not fix it.** The line is boxed in — empty closes at 708, the
weakest real grasp reaches ~878, and the threshold already sits at 830. And it would not have
helped: later *good* LCD grabs read 1200–1211, so the three bad ones came out **closer to the
profile than the good ones did.** There is no threshold that keeps the good and drops the bad.

**It went the other way once too:** a run logged 5 placements against 4 confirmed departures.
With one signal you cannot tell which number to believe. *That is the whole case for two.*

### 9.4 Closed loop — isolated vs crowded

12 layouts certified against pre-registered thresholds: 6 isolated, 6 crowded, 30 parts each,
mean perimeter contact **0.000 vs 0.123**, unchanged gripper configuration.

| | Attempts | Succeeded | Per attempt | **Parts delivered** |
|---|---|---|---|---|
| isolated | 34 | 28 | 82% | **28/30 = 93%** |
| crowded | 33 | 17 | 52% | **17/30 = 57%** |

`lost_on_lift`: **11 crowded vs 3 isolated**. Median cycle 41–48 s throughout (45.3 isolated /
44.8 crowded). Under the attempt cap, the worst single part supplies 6% of a condition's
denominator, against 24% before the cap.

**The deficit is perception, not grasping.** Three lines of evidence converge: failing crowded
grasps carry contaminated geometry (an LCD reported **59.3 mm** against a true 36 mm, so the
descent referenced a top surface partly belonging to its neighbour); the same classes succeed on
isolated layouts under an unchanged gripper; and a pre-registered descent test on an isolated
raised part succeeded **5 of 5**.

**Systematic vision-to-gripped width bias:** +0.6 mm (esp), +8.7 (arduino), +11.6 (ultrasonic),
**+22.1 (lcd)** — from mask bleed, mask merging where parts touch, and PCA axis flips on
near-square parts.

### 9.5 The descent-depth test (what "5/5" means)

Five **separate pick attempts** on an **isolated** ESP32 resting on an LCD (~12 mm up), nothing
else touching: **3 picks at dz = 35 mm, 2 picks at dz = 33 mm.**

- **Pre-registered hypothesis:** the ESP32 failures were "the jaws not reaching far enough down a
  raised part". If true, these should have failed.
- **Result: 5 of 5 succeeded at both depths. Not confirmed.** Being *raised* is not what breaks
  the pick — the failing ESP32s were in **crowded** scenes.
- **What it does NOT show:** 3 vs 2 with no failures either side **cannot distinguish 35 from 33**.
  Nothing is claimed about which depth is better. It is 5 trials, not a rate.
- **Small counter-signal:** the deeper descent closed on something ~1 mm wider, the same direction
  as the 16 Sep finding that extra depth over a *wide* support starts to enclose the support too.
  This argues against "just go deeper" as a general rule.
- **Protocol weaknesses, admitted:** the stack was rebuilt at a different bench position each
  trial (x from −138 to −29) rather than at a marked spot, and the support was an LCD rather than
  the Arduino specified. These are not tightly paired trials.
- **It covers a part sitting *flat* on a support, not a tilted one.** That is why tilt stays open.

**Method point worth keeping:** an operator hypothesis formed from watching failures was
pre-registered, tested with one variable changed, and **not confirmed**. That is a better use of
the last lab hour than twelve more successful picks.

### 9.6 Logged-attempt inventory

| Configuration epoch | Attempts | Succeeded | Rate |
|---|---|---|---|
| pre-calibration (before the 12 Sep finger swap) | 29 | 23 | 79% |
| 12 Sep, threshold 750/790 (accepted empty air) | 12 | 8 | 67% |
| 12 Sep, threshold 830 | 41 | 32 | 78% |
| 15 Sep, BOQ development (software changed between runs) | 20 | 12 | 60% |
| 16 Sep, tilt-measurement battery | 48 | 33 | 69% |
| 19 Sep, gated battery (cap + measured conditions) | 90 | 66 | 73% |
| **total reported** | **240** | *(never pooled)* | |
| *26 Sep, viva rehearsal (post-submission, not evidence)* | *53* | *40* | *75%* |

483 rows across 102 real runs, of which 240 are attempts inside the six reported epochs.
**The 240 is an inventory of testing, never a success rate.** Attempts exclude deliberate skips,
controller/serial faults and operator aborts. Success is reported two ways throughout: **per
attempt** (excludes refusals) and **per part present** (counts them).

---

## 10. Safety nets

| Net | Stops |
|---|---|
| `--min-z` hard floor (213), applied *after* every descent policy | jaws into the bench. Clamps rather than skips: too shallow costs a cycle, too deep costs a finger. |
| `jaw_obstructed()` — refuse if `z_land ≥ part top − 2 mm` | descending onto a neighbour. **Independent of occlusion**, and the only one of the two that stops a collision. |
| width-adaptive aperture + `WIDTH_SANITY_MM` | sweeping wide-open jaws through neighbours |
| occlusion gate 0.35 | grasping a pinned part |
| `--envelope on` (x −250..150, y 500..900, z 230..320) | far-field phantoms entering the bill, or destroying `scene_floor` |
| per-part cap 3 · blocker cap 2 · ≤ 3 enabling moves | infinite retry on one stubborn unit |
| `THRESHOLD_CALIB_MAX_OPEN` check | a recalibrated gripper silently voiding every tick constant |
| `_drain()` + 30 s socket timeout + `ERR` fallback | desync, silent controller, unknown command |
| `release_failed` → `break` | closing on an already-full gripper |
| `input()` before every motion, hand on the e-stop | everything else |

---

## 11. Known limits, debts and open items

### 11.1 The approach corridor is unguarded — the clearest blind spot

Both guards reason about the **end** of the descent: occlusion asks what lies *above* a part;
`z_land` asks what lies *under the jaws where they stop*. **Neither asks what the open jaws sweep
through on the way down.**

**Both operator e-stops in this project were this failure** (16 Sep L4 and 19 Sep H1). In the
19 Sep case: occlusion **0.000**, crowding **0.031**, `z_land` 246.7 against a part top of 252.1
(so the jaws were landing on **bare bench**, 5.4 mm below the part). Every guard passed it
*correctly by its own definition*, the jaws were open ~1901 ticks ≈ **44 mm**, and a neighbour
stood in the path. The arm was stopped **before contact**.

> ⚠ The submitted report prints 0.031 as "occlusion" for this case. The log shows **0.031 is
> crowding; occlusion was 0.000.** Say "crowding 0.031".

**The fix needs no new sensor:** sweep a cylinder of the commanded aperture diameter from hover
height to grasp height and refuse if any other part's points fall inside it. **Every scan already
carries the point set.** This is a check that was not written, not one that could not be.

### 11.2 Tilted parts — works, but never a controlled test

Tilted parts (one edge on the bench, the other propped on a neighbour) were **picked routinely
throughout** — **23 tilted attempts, 13 delivered** before submission, and the demonstration video
is an Arduino leaning on a joystick. But the descent experiment used a part sitting **flat** on a
support, so the tilted case was never isolated with one variable changed. **Flagged open rather
than claimed.** The honest statement: *it works; I did not prove why it sometimes does not.*

### 11.3 `clearance` descent mode is implemented, not validated

All three descent modes exist and are unit-tested. **`clearance` has never commanded a motion.**
What ran live is `fixed` + `jaw_obstructed` + `--min-z`. Reported as implemented, not validated —
do not debut it in a demonstration.

### 11.4 Limits of the measurements themselves

- **The tilt estimate is a planarity residual, not a tilt.** Five consecutive scans of one
  untouched LCD returned 1.1, 1.1, 1.9, 3.3, 2.2 mm against a 3.0 mm threshold. It classifies
  nothing and was **cut from the layout gate**. The signal actually used for slanted boards is the
  gap between the nearest depth band and the mask median (`z − z_med > 6 mm`).
- **The hard descent floor rests on a single paper-drag contact height** at one spot.
  `bench_plane.py` shows the bench is not level (0.54°, ~2.6 mm across the working range); the
  floor should be a **fitted plane**, not a point. Re-measuring on the final day placed contact
  2.3 mm from where the floor assumed.
- **Hand-eye was fitted at one parking of the arm.** That the arm returns to the capture pose
  repeatably each cycle is **assumed, not separately measured [not measured]**. Mechanically the
  camera is eye-in-hand; functionally the transform is valid only at that taught pose, so it
  behaves as a fixed camera.
- **The vision-to-gripped width bias is characterised well enough to correct from the logs** —
  and was not corrected.
- **Detection metrics are count-based, not IoU.** Overlays are the guard.
- **Single fixed capture pose** — no second viewpoint, so occluded parts are recoverable only
  across successive scans.
- **Metric definitions admit two unobserved failures:** a part that falls where the camera cannot
  see it, and a board landing face down (the training set contains only face-up presentations),
  would both be scored as delivered. Neither occurred in these trials; both follow from the
  metric's definition.
- **`grip_profiles.json` is currently misleading.** It holds post-submission experiment values
  (lcd 1460 and ultrasonic 891 at **120 raw**; arduino/esp entries wiped).
  **`grip_profiles_100raw_submitted.json` is the submitted state.** Restore from that backup.

### 11.5 Housekeeping left undone

- `analyse_runs.py` / `LAB_LOG.md` mis-date lab session L4 as "16 Sep"; the real date is 12 Sep.
- `scan.py:61` comment says the depth ladder "matches all 35 recordings" — it is 45 now.
- The root `README.md` describes the **phase-1** system (OBB + homography) and is misleading for
  PDE4445.
- `report/_head.md` is an empty leftover file; delete it.
- Task #19, "fit a per-class planarity baseline from `gates.jsonl`", was never completed.
- `--place-xy 0,550` appears in some command histories and is **wrong** — the taught PlacePoint is
  at (209, 775). Drop the flag or pass `209,775`.

---

## 12. Hard-won lessons (the ones that cost real time)

### 12.1 Hardware

1. **A USB-A→USB-C adapter was the root cause of every camera failure** — `Couldn't resolve
   requests`, `no frames captured`, and the Windows Media Foundation errors (0x800701b1,
   `MFCreateDeviceSource` "cannot find the path"). Plugged direct, the camera streamed 15 frames
   continuously. **Never run this rig through a USB adapter.**
2. **Ten camera open/close cycles in a minute is more than the Windows UVC stack will take.** Hold
   the pipeline open for repeated measurements.
3. **The servo's Current Limit register caps Goal Current** below 160 on this unit. `_set_current`
   walks a ladder down until a value is accepted rather than crashing.
4. **Opening must exceed gripping.** Opening at 100 after gripping at 110 moves the fingers 0 ticks.

### 12.2 Protocol and state

5. **A reply left in the TCP buffer makes the PC run one step ahead of the robot — permanently,
   and across connections.** Drain before every command.
6. **Silence is not a safe failure.** Every command must produce a reply, including bad ones.
7. **RC+'s editor buffer can be older than the file on disk.** Ask the controller, not the IDE.
8. **CRLF, not LF**, in `Main.prg`.
9. **A motion interface exposing 4 of 6 DOF silently makes the previous state part of the
   command.** The arm's *starting pose* is a hidden precondition of every subsequent motion.

### 12.3 Measurement discipline

10. **Never pool across configuration epochs.** A changed finger, threshold or height reference
    makes two rows incomparable. Several software changes were *corrections to defects that were
    themselves producing failures*.
11. **Cap attempts per physical unit.** Without a cap, one stubborn LCD supplied 7 of 29 attempts
    in a condition and **inverted the easy-vs-hard result on its own**.
12. **Assign difficulty by measurement before the run, never by eye afterwards.** The 16 Sep
    battery came out backwards (hard 77%, easy 68%) partly because of this.
13. **Check a layout has power before running it.** A paired ordering trial measures nothing if
    raster order happens to be a valid top-down order (16 Sep L3: both arms 4/4, zero information).
14. **A threshold goes below the lowest *real* positive, not below the highest false one.**
15. **Verify the tool before trusting the result.** The first repeatability run reported σ ≈ 13 mm
    and σ_yaw ≈ 52°. Three tells gave it away: only 2 of 5 tracks survived despite 100% detection;
    52° is the PCA sign flip (yaw is defined mod 180); and 13 mm lateral error is **falsified by
    5/5 bench picks on a 53 mm part**. The matcher was broken, not the bench.
16. **Warm-up contaminates timing.** The first inference took 6.26 s against a ~120 ms steady
    state; report the median with the first excluded.
17. **Read the stream profile, not the array.** `depth_res` read *after* `align.process()` records
    the aligned size, not the sensor mode — and produced a confidently wrong diagnosis.
18. **Log every attempt.** `GRASP_DZ_MM` was measured successfully on 2 Sep, never written down,
    and had to be re-measured.
19. **Keep invalid runs, prefixed and separated** (`runs/discarded/`), rather than deleting them.
20. **No first-time code paths on a lab day.** `test_descent.py` exists to enforce this.

### 12.4 Reporting

21. **An honest deferral is a result; a silent failure is not.** Every refusal is logged with its
    reason and its score.
22. **Report the null.** The ordering null is what makes the no-re-scan result believable.
23. **State the confound yourself**, before anyone asks — the unblinded operator, the small n, the
    count-based metrics, the single-point descent floor.
24. **Separate the piles.** A blocker moved to reach something else goes to its own taught pose,
    because a blocker on the delivery pile is indistinguishable from a fulfilment to anyone
    watching the arm, and `placed` vs `blocker_cleared` would have nothing to point at.

---

## 13. Running it again from cold

```bash
# 0. Preflight — no motion
python pde4445-dev/test_descent.py              # desk regression, no hardware
python pde4445-dev/dday/which_build.py          # what Main.prg is the controller running?
python pde4445-dev/arm_check.py                 # one safe move, hand on the e-stop
python pde4445-dev/scan.py --conf 0.55          # 5/5 parts? depth mode 848x480?

# 1. Only if the gripper was recalibrated — every tick constant is void until this runs
python pde4445-dev/grip_verify.py 100           # re-derive HOLD_THRESHOLD across ALL classes
python pde4445-dev/aperture_calib.py            # re-fit mm -> ticks

# 2. Only if anything moved the camera or the arm base
python pde4445-dev/calib_capture.py             # click markers, then touch them with the tool tip
python pde4445-dev/calib_solve.py               # -> handeye.json, want RMS < 5 mm
python pde4445-dev/yaw_calib.py                 # -> yaw_calib.json

# 3. The demonstration
python pde4445-dev/boq_fetch.py --interactive --live --run \
    --grasp-dz 25 --min-z 213 --table-z 240 --max-occlusion 0.35

# 4. Offline analysis — no hardware needed
python pde4445-dev/eval_vision.py               # -> vision_eval/
python pde4445-dev/analyse_runs.py              # -> analysis/runs.md
```

**Preconditions for anything that moves:** the SPEL+ receiver is running ("Robot ready, listening
to network"); the arm starts at the **gripper-down** pose; the camera is plugged **direct** into
USB 3; the gripper is calibrated and `max_open` still reads **3541** (if not, every tick constant
in §7.1 is void); hand on the e-stop.

---

## 14. If you are an agent picking this up

**Read in this order:** this file → `report/TECH_WALKTHROUGH.md` (deeper technical detail) →
`LAB_LOG.md` (why every number is what it is) → the code.

**Trust order when sources disagree:**
1. the code's own header comments (`pick_one.py` for constants, `clear_scene.py` for policy)
2. `LAB_LOG.md`
3. `analysis/runs.md` and `vision_eval/` (generated, never hand-edited)
4. `report/report.tex` / `dissertation.tex`
5. `PROJECT_STATE.md` — **contains superseded sections, clearly later-dated; check the date**

**Things that look like bugs and are not:**
- `jump` is three `Go` moves (SPEL `Jump` faults on a 6-axis arm).
- Independent `If` blocks in `Main.prg` with an `ok` flag, rather than a chain — deliberate, with
  a fallback reply.
- `--envelope` defaults to `off` so an old run can still be reproduced command-for-command.
- `z` and `z_med` are both emitted and are *supposed* to differ on tilted parts.
- `FINGER_TIP_OFFSET_MM = 30.3` contradicts `TOOL_Z_OFFSET_MM = 70`. Both are correct in their own
  frame; see §5.2.
- The dry-run mock deliberately reproduces the *uncomfortable* state of the rig rather than
  letting every rehearsal succeed.

**Do not, without re-measuring:** change any tick constant; pool run data across epochs; enable
`clearance` descent mode on hardware; use `gripper_api.py` / `gripper_check.py` as a preflight;
trust `grip_profiles.json` over the submitted backup; or quote the 26 Sep rehearsal as evidence.

**The single most useful thing to understand:** the gripper can only answer *"is something between
my fingers?"*. The question the task needs answered is *"did the right part leave the bench?"* —
a fact about the table, not about the fingers. **No finger sensor crosses that gap. Only looking
does.**
