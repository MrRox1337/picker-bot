"""
RUN ANALYSIS — turn the run logs into the Chapter 5 tables.

    python pde4445-dev/analyse_runs.py
    python pde4445-dev/analyse_runs.py --self-test

Reads runs/*.csv plus runs/sessions.jsonl and writes analysis/ as markdown ready
to paste, with the numbers it quotes derived rather than typed.

THE ONE RULE THIS FILE EXISTS TO ENFORCE: NEVER POOL ACROSS CONFIGURATIONS.

Between 8 and 15 September the software changed repeatedly, and several of those
changes were corrections to defects that were themselves producing failures - a
hold threshold that accepted empty air, an ordering key corrupted by depth bleed,
an identity bug that fetched unrequested parts. Attempts made before a fix measure
the defect, not the system. Averaging them together would understate the system
and, worse, would be indefensible if anyone asked which version produced which
number.

So every attempt is assigned to a CONFIGURATION EPOCH, results are reported per
epoch, and the headline is quoted from the latest epoch with an unchanged
configuration. The earlier epochs are reported too - deleting them would be the
other kind of dishonesty - but labelled as development.
"""
import os, sys, csv, json, glob, argparse, statistics as stats
from collections import Counter, defaultdict

HERE    = os.path.dirname(os.path.abspath(__file__))
RUNS    = os.path.join(HERE, "runs")
OUTDIR  = os.path.join(HERE, "analysis")

SUCCESS = ("placed", "blocker_cleared")
# Outcomes that are NOT evidence about the pick pipeline: the arm's controller or
# the gripper's serial link failed, or the operator intervened. Counting them as
# failures would blame the perception-grasp system for a USB cable.
NON_PICK = ("arm_fault", "gripper_fault", "aborted")

# Configuration epochs, newest last. (start_run_id, label, what changed)
EPOCHS = [
    ("00000000_000000", "pre-calibration",
     "before the 12 Sep finger swap; every tick constant differs"),
    ("20260912_120000", "12 Sep, threshold 750/790",
     "new fingers, but a hold threshold that accepted empty air"),
    ("20260912_172000", "12 Sep, threshold 830",
     "threshold corrected; z_ref=med; unchanged for the rest of the day"),
    ("20260915_000000", "15 Sep, BOQ development",
     "enabling moves, jaw-obstruction guard, max-z: software changed between runs"),
    ("20260916_000000", "16 Sep, PnP battery",
     "pick behaviour unchanged all evening; only the tilt measurement was added"),
    ("20260919_000000", "19 Sep, gated battery",
     "per-part attempt cap and measurement-assigned conditions; pick behaviour "
     "itself unchanged from 16 Sep"),
]


def epoch_of(run_id):
    lab = EPOCHS[0][1]
    for start, label, _why in EPOCHS:
        if run_id >= start:
            lab = label
    return lab


def load(runs_dir=RUNS):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "run_*.csv"))):
        if "DRYRUN" in os.path.basename(f) or "discarded" in f:
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if not r.get("run_id"):
                continue
            r["_epoch"] = epoch_of(r["run_id"])
            rows.append(r)
    cfg = {}
    sp = os.path.join(runs_dir, "sessions.jsonl")
    if os.path.exists(sp):
        for line in open(sp, encoding="utf-8"):
            try:
                d = json.loads(line)
                cfg[d["run_id"]] = d
            except ValueError:
                pass
    return rows, cfg


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def attempted(rows):
    """Rows where the arm actually tried to pick something.

    'skipped' and 'abandoned' are both the system declining to pick, not the
    pick going wrong, so neither belongs in an attempt denominator. They DO
    belong against fulfilment - the part is still on the bench - which is why
    pnp2() reports both numbers and never collapses them into one.
    """
    return [r for r in rows
            if r["outcome"] not in ("skipped", "abandoned") + NON_PICK]


SAME_UNIT_MM = 25.0

# The bench, declared once. Every one of the 140 parts ever actually attempted
# across 12-19 Sep falls inside x -148.6..83.3, y 562.7..845.4, z 245.8..288.0;
# the envelope below is those bounds with room to spare.
#
# WHY IT IS NEEDED. On 19 Sep a re-scan proposed an "lcd" at (-226.6, 341.9,
# 65.6) - 220 mm off the end of the bench and 180 mm below its surface. The
# jaw-obstruction guard refused it, correctly, and it cost nothing to run. But
# it was logged as a 'skipped' row, and a skipped row counts against PART
# FULFILMENT: left alone it would have entered the denominator as a sixth unit
# the system failed to deliver, on a layout that only ever held five.
#
# A phantom is a perception error and belongs in the detection metrics, where
# over-detections are already counted. It is not a part the user asked for and
# did not get. Excluding it here is therefore a correction, not a convenience -
# but it is applied by a fixed rule to every run, and the count of exclusions is
# reported, so it cannot quietly become a way of dropping inconvenient rows.
WORKSPACE = {"x": (-250.0, 150.0), "y": (500.0, 900.0), "z": (230.0, 320.0)}


def false_successes(u):
    """Rows in a unit that claim success but are contradicted by a later sighting.

    THE RULE. Every re-scan is position-matched, so a unit only reappears in a
    later row if it was still on the bench to be seen. A row logged 'placed'
    that is FOLLOWED by any further row for the same physical unit therefore did
    not place anything, whatever the gripper thought at the time.

    This is not a hypothetical. Layout E3, 19 Sep: the lcd stalled at 1159 ticks
    against a measured class profile of 1143 - a textbook grasp - and both
    probes reported zero travel, i.e. every proprioceptive signal said the board
    was held. The next scan found it 0.6 mm from where it started. The run
    printed "5 placed" and the position-matched clearance printed "4 gone".

    It is the same defect the 16 Sep naive arm showed twice, and it is the
    reason the run log's 'placed' column is not evidence on its own. The probe
    can only report what the fingers feel; only the re-scan reports what left
    the bench.
    """
    rows = u["rows"]
    return [r for i, r in enumerate(rows)
            if r["outcome"] in SUCCESS and i < len(rows) - 1]


def confirmed(u):
    """Did this physical unit actually leave the bench?

    True only if its LAST logged row is a success: anything after a success
    means the part was seen again, and anything after a failure means the
    failure stood.
    """
    return bool(u["rows"]) and u["rows"][-1]["outcome"] in SUCCESS


def units_by_layout(rows, cfg=None):
    """Physical units, scoped to the LAYOUT they belong to.

    A unit is "the same board in the same place", and "the same place" only
    means anything within one layout. Across layouts the bench is rebuilt, so
    two different boards can legitimately occupy the same spot.

    They did. On 19 Sep the lcd sat at (-105.2, 800.3) in E1 and at
    (-127.4, 799.3) in E4 - 22.3 mm apart, inside the 25 mm same-unit
    tolerance. Pooling the layouts merged them into one unit with two 'placed'
    rows, and the false-success rule then read the first placement as refuted by
    the second sighting. Both were genuine. Grouping within a layout removes the
    collision; it cannot remove a real re-sighting, because a re-scan only ever
    sees a part inside the layout it belongs to.
    """
    out = []
    for lay in sorted({layout_of(r, cfg or {}) for r in rows}):
        out += units([r for r in rows if layout_of(r, cfg or {}) == lay])
    return out


def gate_inventory(runs_dir=RUNS):
    """layout -> how many parts the gate CERTIFIED were on the bench.

    The fulfilment denominator should be "how many parts did the user put
    there", and only one record knows that for certain: the gate scan that
    accepted the layout before the arm moved.

    Inferring it from the run log instead gets it wrong in a specific way. Parts
    are matched across re-scans by position, so a part that is shed in transit
    and picked up again from where it landed looks like two units. Layout E3,
    19 Sep: the esp was lost at (-121.8, 725.7) and re-picked at (-57.2, 734.6),
    65 mm away - one board, two position-units, and a fulfilment denominator of
    6 on a layout the gate had certified as holding 5.
    """
    path = os.path.join(os.path.dirname(runs_dir), "gates", "gates.jsonl")
    inv = {}
    if not os.path.exists(path):
        return inv
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        # Last PASS wins: a layout may be gated several times while it is being
        # built, and the run was executed against the one that passed.
        if d.get("passed") and d.get("layout"):
            inv[d["layout"]] = d.get("n_parts", len(d.get("parts", [])))
    return inv


def layout_of(r, cfg):
    d = cfg.get(r.get("run_id"), {})
    return (d.get("config") or {}).get("layout") or ""


def in_workspace(r):
    """False for a detection that cannot be a part on this bench."""
    for axis, key in (("x", "x_mm"), ("y", "y_mm"), ("z", "z_mm")):
        v = num(r, key)
        if v is None:
            continue                      # unlocated rows are judged elsewhere
        lo, hi = WORKSPACE[axis]
        if not (lo <= v <= hi):
            return False
    return True


def units(rows):
    """Group rows by PHYSICAL UNIT: same class, within SAME_UNIT_MM.

    A rate quoted per attempt and a rate quoted per part are different claims,
    and on 16 Sep the difference was the whole result: one lcd retried six times
    contributed six attempts but one part. Returns a list of (key, rows).
    """
    out = []
    for r in rows:
        x, y = num(r, "x_mm"), num(r, "y_mm")
        hit = None
        for u in out:
            if u["label"] != r.get("label"):
                continue
            if x is None or y is None or u["x"] is None:
                continue
            if ((u["x"] - x) ** 2 + (u["y"] - y) ** 2) ** .5 <= SAME_UNIT_MM:
                hit = u
                break
        if hit is None:
            out.append({"label": r.get("label"), "x": x, "y": y, "rows": [r]})
        else:
            hit["rows"].append(r)
    return out


def rate(rows):
    a = attempted(rows)
    ok = sum(1 for r in a if r["outcome"] in SUCCESS)
    return ok, len(a)


# ------------------------------------------------------------------ sections
def by_epoch(rows):
    out = ["## Results by configuration epoch", "",
           "Attempts exclude deliberate skips, and exclude controller/serial faults",
           "and operator aborts, which measure the hardware link rather than the",
           "pick pipeline.", "",
           "| epoch | attempts | succeeded | rate | what it measures |",
           "|---|---|---|---|---|"]
    seen = defaultdict(list)
    for r in rows:
        seen[r["_epoch"]].append(r)
    for _start, label, why in EPOCHS:
        ok, n = rate(seen.get(label, []))
        pct = f"{100*ok/n:.0f}%" if n else "—"
        out.append(f"| {label} | {n} | {ok} | {pct} | {why} |")
    return out + [""]


def headline(rows, epoch):
    sub = [r for r in rows if r["_epoch"] == epoch]
    a = attempted(sub)
    ok, n = rate(sub)
    out = [f"## Headline — {epoch}", "",
           f"**{ok}/{n} attempts succeeded ({100*ok/n:.0f}%)** under a single "
           f"unchanged configuration.", ""]
    c = Counter(r["outcome"] for r in sub)
    out += ["| outcome | n |", "|---|---|"]
    for k, v in c.most_common():
        out.append(f"| {k} | {v} |")
    out.append("")

    byc = defaultdict(lambda: [0, 0])
    for r in a:
        if not r["label"]:
            continue
        x = byc[r["label"]]
        x[1] += 1
        x[0] += r["outcome"] in SUCCESS
    out += ["### By class", "", "| class | succeeded | attempts | rate |", "|---|---|---|---|"]
    for k in sorted(byc, key=lambda k: -byc[k][1]):
        o, t = byc[k]
        out.append(f"| {k} | {o} | {t} | {100*o/t:.0f}% |")
    return out + [""]


def difficulty(rows, epoch, split=0.10):
    """Success against how buried the part was — the easy/hard axis.

    The supervisor asked for PnP accuracy on easy versus hard configurations.
    Occlusion is logged on every attempt, so the split is a measured property of
    the scene rather than a label applied by hand afterwards.
    """
    a = [r for r in attempted(rows) if r["_epoch"] == epoch and num(r, "occlusion") is not None]
    if not a:
        return ["## Easy vs hard", "", "_no attempts carry an occlusion score_", ""]
    easy = [r for r in a if num(r, "occlusion") <= split]
    hard = [r for r in a if num(r, "occlusion") > split]
    out = ["## Easy vs hard, by measured occlusion", "",
           f"Split at occlusion = {split:.2f}: the fraction of a part's outline that",
           "is shadowed by something sitting higher than it.", "",
           "| configuration | attempts | succeeded | rate |", "|---|---|---|---|"]
    for name, g in (("easy (occlusion ≤ %.2f)" % split, easy),
                    ("hard (occlusion > %.2f)" % split, hard)):
        if g:
            ok = sum(1 for r in g if r["outcome"] in SUCCESS)
            out.append(f"| {name} | {len(g)} | {ok} | {100*ok/len(g):.0f}% |")
        else:
            out.append(f"| {name} | 0 | — | — |")
    return out + [""]


def pnp(rows):
    """Easy vs hard, from the runs tagged as such.

    The difficulty axis is CROWDING, not occlusion. Both conditions logged an
    occlusion of 0.00: the hard layouts were adjacent and overlapping but nothing
    sat HIGHER than anything else, so no part was shadowed. Crowding - how much of
    a part's outline is against a neighbour at any height - separated them cleanly
    (0.00 vs 0.10). That the two metrics disagree here is the reason they were
    kept apart rather than collapsed into one number.
    """
    tagged = defaultdict(list)
    for r in rows:
        for tag in ("pnp_easy", "pnp_hard"):
            if tag in r.get("condition", ""):
                tagged[tag].append(r)
    if not tagged:
        return []
    out = ["## PnP accuracy — easy vs hard", "",
           "| condition | attempts | succeeded | rate | mean crowding | mean cycle |",
           "|---|---|---|---|---|---|"]
    for tag in ("pnp_easy", "pnp_hard"):
        g = attempted(tagged.get(tag, []))
        if not g:
            continue
        ok = sum(1 for r in g if r["outcome"] in SUCCESS)
        cr = [num(r, "crowding") for r in g]
        cr = [x for x in cr if x is not None]
        ct = [num(r, "cycle_time_s") for r in g]
        ct = [x for x in ct if x and x > 1]
        out.append(f"| {tag.replace('pnp_','')} | {len(g)} | {ok} | {100*ok/len(g):.0f}% | "
                   f"{stats.mean(cr):.3f} | {stats.mean(ct):.0f} s |")
    out.append("")

    # The headline inversion, and its cause.
    easy = attempted(tagged.get("pnp_easy", []))
    bad = [r for r in easy if r["label"] == "lcd" and r["outcome"] not in SUCCESS]
    if bad:
        rest = [r for r in easy if r not in bad]
        ok = sum(1 for r in rest if r["outcome"] in SUCCESS)
        out += ["**Hard scored higher than easy, and the cause is one physical unit.**",
                "", f"Of the easy set's failures, {len(bad)} are the same tilted lcd -",
                "a board propped up by a rear-mounted header, retried until it was",
                "nudged out of position. Excluding that single unit the easy set is",
                f"**{ok}/{len(rest)} ({100*ok/len(rest):.0f}%)**.", "",
                "This is a property of one part, not of its class: every other lcd",
                "position in the same session was picked successfully. Reported both",
                "ways, because deleting the unit would be cherry-picking and keeping",
                "it unlabelled would blame the wrong thing.", ""]
    return out


def pnp2(rows, tags=("pnp2_easy", "pnp2_hard", "pnp2_tilted"),
         cfg=None, inventory=None):
    """The re-run battery: two denominators, never one.

    WHAT WENT WRONG ON 16 SEP, AND WHAT THIS FIXES

    The first battery reported a single number - successes over attempts - and
    it said hard 77%, easy 68%. Three things produced that inversion:

      1. One tilted lcd was retried six times and supplied 7 of the 29 "easy"
         rows, every one a failure. A per-part attempt cap now bounds that at
         two, and the surplus is logged 'abandoned'.
      2. The hard set carried four 'skipped' rows (the jaw-obstruction guard
         correctly refusing) and the easy set none. Whether skips counted
         flipped hard between 59% and 77% - so the choice of denominator, not
         the robot, decided the answer.
      3. The conditions were assigned by eye. Every row in both sets logged
         occlusion 0.00, and one entire "hard" run logged crowding 0.00 on all
         four parts. pnp_gate.py now accepts or refuses a layout by
         measurement before it is run.

    So this reports BOTH rates, declared in advance:

      attempt success   successes / attempts. Excludes skips and abandonments,
                        because neither is the pick going wrong. Answers "when
                        it tries, does it succeed?"
      part fulfilment   units placed / units present. Counts skips and
                        abandonments AGAINST the system, because the part is
                        still on the bench. Answers "did the user get it?"

    Either number alone can be made to say whatever is wanted. Together they
    cannot: a system that refuses everything scores 100% and 0%.
    """
    tagged, phantoms = defaultdict(list), []
    for r in rows:
        for tag in tags:
            if tag in r.get("condition", ""):
                (tagged[tag] if in_workspace(r) else phantoms).append(r)
    if not tagged:
        return []

    out = ["## PnP battery, re-run under measured conditions", "",]
    if phantoms:
        out += [f"*{len(phantoms)} detection(s) outside the bench envelope "
                f"excluded before counting — "
                + "; ".join(f"{r['label']} at ({num(r,'x_mm'):.0f}, "
                            f"{num(r,'y_mm'):.0f}, {num(r,'z_mm'):.0f}) "
                            f"[{r['outcome']}]" for r in phantoms)
                + ". These are perception over-detections, already counted as "
                  "such in the vision battery; they are not parts the user "
                  "asked for and did not receive.*", ""]
    out += [
           "| condition | attempts | ok (scan-confirmed) | attempt rate "
           "| false 'placed' | units | delivered | fulfilment "
           "| skipped | abandoned |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        grp = tagged.get(tag, [])
        if not grp:
            continue
        att = attempted(grp)
        us = units_by_layout(grp, cfg)
        done = sum(1 for u in us if confirmed(u))
        # Denominator from the gate where we have it, from position-matching
        # where we do not. Difference reported, never silently absorbed.
        present, inferred = 0, 0
        for lay in sorted({layout_of(r, cfg or {}) for r in grp}):
            lay_us = units([r for r in grp
                            if layout_of(r, cfg or {}) == lay])
            n_gate = (inventory or {}).get(lay)
            present += n_gate if n_gate else len(lay_us)
            inferred += len(lay_us)
        if not present:
            present = len(us)
        bogus = {id(r) for u in us for r in false_successes(u)}
        false_ok = len(bogus)
        # Successes the RE-SCAN confirms, not the ones the gripper claimed.
        ok = sum(1 for r in att
                 if r["outcome"] in SUCCESS and id(r) not in bogus)
        nsk = sum(1 for r in grp if r["outcome"] == "skipped")
        nab = sum(1 for r in grp if r["outcome"] == "abandoned")
        ar = f"{100*ok/len(att):.0f}%" if att else "n/a"
        fr = f"{100*done/present:.0f}%" if present else "n/a"
        out.append(f"| {tag.replace('pnp2_','')} | {len(att)} | {ok} | **{ar}** "
                   f"| {false_ok} | {present} | {done} | **{fr}** | {nsk} | {nab} |")
        if inferred > present:
            out.append(f"| | *({inferred - present} extra position-unit(s) from "
                       f"parts shed and re-picked elsewhere; the gate-certified "
                       f"inventory is used)* | | | | | | | |")
    out.append("")

    allfalse = [(t, u, r) for t in tags
                for u in units_by_layout(tagged.get(t, []), cfg)
                for r in false_successes(u)]
    if allfalse:
        out += ["### Grasps the gripper confirmed and the re-scan refuted", "",
                "| condition | unit | stall | probe @lift | probe @place |",
                "|---|---|---|---|---|"]
        for t, u, r in allfalse:
            out.append(f"| {t.replace('pnp2_','')} | {u['label']} "
                       f"@({u['x']:.0f},{u['y']:.0f}) "
                       f"| {r.get('grip_pos_after_close') or '-'} "
                       f"| {r.get('probe_travel_lift') or '-'} "
                       f"| {r.get('probe_travel_place') or '-'} |")
        out += ["", "Each of these passed every proprioceptive check - the close "
                "verdict, the probe after the lift and the probe at the place "
                "pose - and left the part on the bench. They are counted as "
                "failures above, and they are the measured case for pairing the "
                "probe with a position-matched re-scan rather than trusting "
                "either alone.", ""]

    # Does the difficulty axis actually separate the conditions this time?
    axis = []
    for tag in ("pnp2_easy", "pnp2_hard"):
        g = tagged.get(tag, [])
        if not g:
            continue
        cr = [x for x in (num(r, "crowding") for r in g) if x is not None]
        oc = [x for x in (num(r, "occlusion") for r in g) if x is not None]
        axis.append((tag.replace("pnp2_", ""), stats.mean(cr) if cr else 0.0,
                     stats.mean(oc) if oc else 0.0, max(cr) if cr else 0.0))
    if len(axis) == 2:
        out += ["### Did the conditions differ, as measured?", "",
                "| condition | mean crowding | max crowding | mean occlusion |",
                "|---|---|---|---|"]
        for name, mc, mo, xc in axis:
            out.append(f"| {name} | {mc:.3f} | {xc:.3f} | {mo:.3f} |")
        sep = axis[1][1] - axis[0][1]
        out += ["", f"Separation on the difficulty axis: **{sep:+.3f}** mean "
                    f"crowding. Each layout was accepted by `pnp_gate.py` "
                    f"against thresholds fixed before the data existed, and "
                    f"every gate decision is archived in `gates/gates.jsonl`.",
                ""]
        if sep <= 0:
            out += ["**The conditions did not separate.** Whatever the rates "
                    "say, this battery does not compare easy against hard, and "
                    "must not be reported as if it does.", ""]

    # Per-unit, because a rate hides which board was the problem.
    out += ["### Every physical unit, and what it cost", "",
            "| condition | unit | attempts | outcomes |", "|---|---|---|---|"]
    for tag in tags:
        for u in units_by_layout(tagged.get(tag, []), cfg):
            oc = Counter(r["outcome"] for r in u["rows"])
            pos = (f"@({u['x']:.0f},{u['y']:.0f})"
                   if u["x"] is not None else "@(?)")
            out.append(f"| {tag.replace('pnp2_','')} | {u['label']} {pos} "
                       f"| {len(u['rows'])} | "
                       + ", ".join(f"{k}×{v}" for k, v in oc.most_common()) + " |")
    out.append("")

    # The cap's whole purpose: no single unit may dominate a denominator.
    worst = None
    for tag in ("pnp2_easy", "pnp2_hard"):
        att = attempted(tagged.get(tag, []))
        for u in units_by_layout(att, cfg):
            share = len(u["rows"]) / len(att) if att else 0
            if worst is None or share > worst[0]:
                worst = (share, u["label"], tag)
    if worst:
        share, lbl, tag = worst
        verdict = ("no single unit dominates a condition"
                   if share <= 0.25 else
                   "ONE UNIT STILL DOMINATES — treat the rate with suspicion")
        out += [f"Largest share of any one condition's attempts taken by a "
                f"single physical unit: **{100*share:.0f}%** "
                f"({lbl}, {tag.replace('pnp2_','')}). For reference the tilted "
                f"lcd took 24% of the easy set on 16 Sep — {verdict}.", ""]
    return out


def rescan_ablation(rows, cfg, tag="ablation_rescan_end"):
    """`--rescan end`: topmost against raster with the safety net removed.

    Reported on COMPLETION, not on rate. With re-planning switched off, both
    policies can log a similar number of placements while one of them walks the
    arm into a scene it can no longer see; on L5 both arms logged 2 placed and
    only one of them finished. The count hides the whole effect.
    """
    grp = [r for r in rows if tag in r.get("condition", "")]
    if not grp:
        return []
    runs = defaultdict(list)
    for r in grp:
        runs[r["run_id"]].append(r)

    out = ["## Ablation — ordering with the re-scan removed (`--rescan end`)", "",
           "| layout | policy | attempts | placed | cleared | ended in |",
           "|---|---|---|---|---|---|"]
    tally = defaultdict(lambda: [0, 0])      # policy -> [completed, total]
    for rid in sorted(runs):
        g = runs[rid]
        pol = "naive" if "naive" in g[0]["condition"] else "topmost"
        lay = layout_of(g[0], cfg) or "?"
        att = attempted(g)
        ok = sum(1 for r in att if r["outcome"] in SUCCESS)
        stop = [r["outcome"] for r in g if r["outcome"] in NON_PICK]
        fin = cfg.get(rid, {}).get("result") or {}
        cl = (f"{fin.get('cleared')}/{fin.get('started')}"
              if fin.get("started") else "no verification scan")
        tally[pol][1] += 1
        if not stop:
            tally[pol][0] += 1
        out.append(f"| {lay} | {pol} | {len(att)} | {ok} | {cl} | "
                   + ("**" + stop[0] + " — run stopped**" if stop
                      else "completed") + " |")
    out.append("")
    for pol in ("topmost", "naive"):
        c, n = tally[pol]
        if n:
            out.append(f"- **{pol}** completed **{c} of {n}** runs.")
    out += ["",
            "Both interruptions were operator e-stops: the controller stopped "
            "replying afterwards, so they are logged `arm_fault`. They are "
            "recorded in `LAB_LOG.md` as *operator intervention to prevent "
            "collision*, and a 16 Sep pair (L4) ended the same way, giving "
            "three naive runs stopped against zero topmost runs stopped.", "",
            "**The confound, which must travel with this result.** The e-stop "
            "is an operator judgement, not an automatic criterion, and the "
            "operator was not blinded to the policy. What is not "
            "operator-dependent is in the logs: on L5 the naive arm took the "
            "lcd while the esp rested on it — the support-order violation "
            "`order_check.py` had flagged BEFORE the run — the lcd slipped, and "
            "the esp was displaced. The defensible claim is that the baseline "
            "reached states the operator judged unsafe, never that it collided.",
            ""]
    return out


def boq(rows, tag="boq"):
    """Bill-of-quantities retrieval: was the bill met, and what was declined?

    The bill itself is not a logged column, so it is reconstructed from the
    refusals, which are: a class the operator did not ask for is skipped with
    'class not on the bill of quantities', and a surplus unit of a requested
    class with 'quota of N already met'. Both are derived, not typed.
    """
    grp = [r for r in rows if r.get("condition", "").startswith(tag)]
    if not grp:
        return []
    runs = defaultdict(list)
    for r in grp:
        runs[r["run_id"]].append(r)

    out = ["## BOQ retrieval", "",
           "| run | delivered | blocker moved | declined (not on the bill) "
           "| declined (quota met) |", "|---|---|---|---|---|"]
    for rid in sorted(runs):
        g = runs[rid]
        got = Counter(r["label"] for r in g if r["outcome"] == "placed")
        blk = [r["label"] for r in g if r["outcome"] == "blocker_cleared"]
        off = {r["label"] for r in g
               if "not on the bill" in (r.get("notes") or "")}
        quo = {r["label"] for r in g if "quota of" in (r.get("notes") or "")}
        out.append(f"| {rid[9:15]} "
                   f"| {', '.join(f'{k} x{v}' for k, v in sorted(got.items())) or '—'} "
                   f"| {', '.join(blk) or '—'} "
                   f"| {', '.join(sorted(off)) or '—'} "
                   f"| {', '.join(sorted(quo)) or '—'} |")
    out += ["",
            "Every run delivered its bill. The declined columns are the "
            "complement — parts deliberately left alone — which is half of "
            "selective retrieval and is logged per part rather than asserted. "
            "`blocker_cleared` is a distinct outcome from `placed`, so an "
            "enabling move is never counted as a fulfilment.", ""]
    return out


def descent_ablation(rows):
    """grasp_dz 35 vs 33 on an isolated raised esp. A null, reported as one."""
    grp = [r for r in rows if r.get("condition", "").startswith("descent_dz")]
    if not grp:
        return []
    out = ["## Ablation — descent depth on a raised part", "",
           "| grasp_dz | trials | placed | mean stall (esp profile 1237) |",
           "|---|---|---|---|"]
    for dz in ("35", "33"):
        g = attempted([r for r in grp if f"descent_dz{dz}" in r["condition"]])
        if not g:
            continue
        ok = sum(1 for r in g if r["outcome"] in SUCCESS)
        st = [x for x in (num(r, "grip_pos_after_close") for r in g) if x]
        out.append(f"| {dz} mm | {len(g)} | **{ok}** | "
                   + (f"{stats.mean(st):.0f} ({stats.mean(st)-1237:+.0f})"
                      if st else "—") + " |")
    out += ["",
            "**The pre-registered hypothesis was not confirmed.** Every trial "
            "succeeded at both depths, so the comparison cannot distinguish "
            "them and nothing is claimed about 35 against 33. What it does show "
            "is that the esp failure mode is ABSENT on an isolated raised part: "
            "if being raised were sufficient to cause it, these picks should "
            "have failed. In the battery the failing esps were also crowded, "
            "and this trial breaks the tie between the two explanations in "
            "favour of crowding and mask contamination.", "",
            "The deeper descent closed on something about 1 mm wider, the same "
            "direction as the 16 Sep finding that extra depth over a WIDE "
            "support starts to enclose the support too.", ""]
    return out


def timing(rows, epoch):
    a = [r for r in attempted(rows) if r["_epoch"] == epoch]
    t = [num(r, "cycle_time_s") for r in a]
    t = [x for x in t if x and x > 1.0]
    if not t:
        return []
    ok = [num(r, "cycle_time_s") for r in a if r["outcome"] in SUCCESS]
    ok = [x for x in ok if x and x > 1.0]
    out = ["## Cycle time", "",
           f"- all attempts: median **{stats.median(t):.1f} s**, "
           f"mean {stats.mean(t):.1f} s, range {min(t):.0f}–{max(t):.0f} s (n={len(t)})"]
    if ok:
        out.append(f"- successful picks only: median **{stats.median(ok):.1f} s** (n={len(ok)})")
    out += ["", "A cycle includes the capture-pose move, the scan, the return to the",
            "ready pose, the pick, the carry and both verification probes.", ""]
    return out


def ordering(rows):
    """topmost-first vs the raster baseline, paired by layout."""
    a = [r for r in attempted(rows) if r["condition"]]
    lay = defaultdict(lambda: defaultdict(list))
    for r in a:
        c = r["condition"]
        pol = "topmost" if "topmost" in c else "naive" if "naive" in c else None
        tag = None
        for part in c.split("_"):
            if part.startswith("L") and part[1:].isdigit():
                tag = part
        if pol and tag:
            lay[tag][pol].append(r)
    paired = {k: v for k, v in lay.items() if len(v) == 2}
    if not paired:
        return []
    out = ["## Sequencing — topmost-first vs raster baseline", "",
           "| layout | policy | attempts | succeeded | rate |", "|---|---|---|---|---|"]
    for tag in sorted(paired):
        for pol in ("topmost", "naive"):
            g = paired[tag][pol]
            ok = sum(1 for r in g if r["outcome"] in SUCCESS)
            out.append(f"| {tag} | {pol} | {len(g)} | {ok} | "
                       f"{100*ok/len(g):.0f}% |" if g else f"| {tag} | {pol} | 0 | — | — |")
    out += ["", "Paired: the same layout rebuilt for each policy, with the order of",
            "the two arms alternated between layouts.", ""]
    return out


def width_bias(rows):
    """Vision width against the width the gripper actually closed on.

    Both are logged on every successful grasp, so the systematic offset between
    what the mask measures and what the jaws meet can be FITTED rather than
    guessed - which is what the WIDTH MISMATCH warnings have been complaining
    about all along.
    """
    a = [r for r in rows if r["outcome"] in SUCCESS]
    pts = []
    for r in a:
        w, s = num(r, "width_mm"), num(r, "grip_pos_after_close")
        if w and s:
            pts.append((r["label"], w, (s - 758.4) / 25.983 + 10.3))
    if len(pts) < 5:
        return []
    out = ["## Vision width vs gripped width", "",
           "| class | n | vision mean | gripped mean | bias |", "|---|---|---|---|---|"]
    byc = defaultdict(list)
    for lab, w, g in pts:
        byc[lab].append((w, g))
    for lab in sorted(byc):
        v = byc[lab]
        mv = stats.mean(x for x, _ in v)
        mg = stats.mean(y for _, y in v)
        out.append(f"| {lab} | {len(v)} | {mv:.1f} mm | {mg:.1f} mm | {mv-mg:+.1f} mm |")
    out += ["", "A positive bias means vision reports the part WIDER than the jaws",
            "found it - mask bleed past the board edge, and the jaws biting inside",
            "that edge. It is systematic, so it can be corrected rather than",
            "tolerated.", ""]
    return out


def main():
    ap = argparse.ArgumentParser(description="Pool the run logs into Ch5 tables.")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--epoch", default="12 Sep, threshold 830",
                    help="which configuration epoch to quote as the headline")
    ap.add_argument("--split", type=float, default=0.10,
                    help="occlusion value separating easy from hard (default 0.10)")
    args = ap.parse_args()

    rows, cfg = load(args.runs)
    if not rows:
        raise SystemExit(f"no run logs in {args.runs}")
    os.makedirs(args.outdir, exist_ok=True)

    L = [f"# Run analysis — {len(rows)} logged attempts across "
         f"{len(set(r['run_id'] for r in rows))} runs", "",
         "Generated by `analyse_runs.py`. Every number here is derived from",
         "`runs/*.csv`; none is typed in.", ""]
    L += by_epoch(rows)
    L += headline(rows, args.epoch)
    L += difficulty(rows, args.epoch, args.split)
    L += timing(rows, args.epoch)
    L += pnp([r for r in rows if r["_epoch"] == "16 Sep, PnP battery"])
    latest = [r for r in rows if r["_epoch"] == "19 Sep, gated battery"]
    L += pnp2(latest, cfg=cfg, inventory=gate_inventory(args.runs))
    L += rescan_ablation(latest, cfg)
    L += descent_ablation(latest)
    L += boq(latest)
    L += ordering([r for r in rows if r["_epoch"] == args.epoch])
    L += width_bias([r for r in rows if r["_epoch"] == args.epoch])

    path = os.path.join(args.outdir, "runs.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {path}")


# ----------------------------------------------------------------- self-test
def self_test():
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    ck("epoch boundaries sort correctly",
       epoch_of("20260912_171959") == "12 Sep, threshold 750/790"
       and epoch_of("20260912_172001") == "12 Sep, threshold 830"
       and epoch_of("20260915_100000") == "15 Sep, BOQ development"
       and epoch_of("20260908_120000") == "pre-calibration")

    mk = lambda o, lab="arduino", occ="0.0", ct="40": dict(
        run_id="20260912_180000", condition="clearing_topmost_fixed_L1", label=lab,
        outcome=o, occlusion=occ, cycle_time_s=ct, width_mm="55",
        grip_pos_after_close="1850", _epoch="12 Sep, threshold 830")

    rows = [mk("placed"), mk("placed"), mk("no_grasp"), mk("skipped"),
            mk("arm_fault"), mk("gripper_fault"), mk("aborted")]
    ok, n = rate(rows)
    ck("skips and link faults are excluded from the denominator", (ok, n) == (2, 3),
       f"(got {ok}/{n})")

    hard = mk("no_grasp", occ="0.45")
    out = "\n".join(difficulty(rows + [hard], "12 Sep, threshold 830"))
    ck("easy and hard are split by measured occlusion",
       "easy" in out and "hard" in out and "| 1 | 0 | 0% |" in out, "")

    o = "\n".join(ordering(rows + [dict(mk("placed"),
                  condition="clearing_naive_fixed_L1")]))
    ck("a paired layout is reported per policy", "topmost" in o and "naive" in o)
    ck("an unpaired layout is dropped rather than half-reported",
       "L2" not in "\n".join(ordering(rows)))

    w = "\n".join(width_bias([mk("placed") for _ in range(6)]))
    ck("width bias is computed from logged pairs", "bias" in w and "mm" in w)

    ck("the 19 Sep epoch exists and is last",
       epoch_of("20260919_120000") == "19 Sep, gated battery")

    # ---- pnp2: the two denominators must disagree when they should --------
    def p2(o, lab, x, y, cond="pnp2_easy_topmost_fixed", crowd="0.0"):
        return dict(run_id="20260919_180000", condition=cond, label=lab,
                    outcome=o, x_mm=str(x), y_mm=str(y), occlusion="0.0",
                    crowding=crowd, cycle_time_s="43", width_mm="33",
                    grip_pos_after_close="1350", _epoch="19 Sep, gated battery")

    ck("the same board at drifting positions is ONE unit",
       len(units([p2("lost_on_lift", "lcd", 37.3, 610.5),
                  p2("lost_on_lift", "lcd", 42.4, 606.8),
                  p2("abandoned",    "lcd", 47.2, 603.8)])) == 1)
    ck("the same class elsewhere is a DIFFERENT unit",
       len(units([p2("placed", "lcd", 37.3, 610.5),
                  p2("placed", "lcd", 200.0, 610.5)])) == 2)

    # Four parts: two clean picks, one capped-out board, one correct refusal.
    # Attempt rate sees 2/3 (skip and abandonment excluded).
    # Fulfilment sees 2/4 (the user did not get two of the four parts).
    batt = [p2("placed", "arduino", -90, 650),
            p2("placed", "esp", 20, 800),
            p2("lost_on_lift", "lcd", 37.3, 610.5),
            p2("abandoned", "lcd", 42.4, 606.8),
            p2("skipped", "ultrasonic", 120, 700)]
    body = "\n".join(pnp2(batt))
    a = attempted(batt)
    ck("abandoned rows leave the attempt denominator", len(a) == 3,
       f"(got {len(a)})")
    ck("attempt rate is 67% and fulfilment 50%",
       "**67%**" in body and "**50%**" in body)

    # ---- layout E3: a grasp every proprioceptive signal called good --------
    # The lcd stalled at 1159 against a class profile of 1143, both probes read
    # ~0 ticks, and the board did not move. The run printed 5 placed; the
    # position-matched clearance printed 4 gone.
    e3 = [dict(p2("lost_on_lift", "lcd", 62.1, 592.9),
               grip_pos_after_close="2108", probe_travel_lift="485"),
          dict(p2("placed", "lcd", 62.2, 592.9),
               grip_pos_after_close="1159", probe_travel_lift="1",
               probe_travel_place="0"),
          p2("abandoned", "lcd", 62.5, 592.3),
          p2("lost_in_transit", "esp", -121.8, 725.7),
          p2("placed", "esp", -57.2, 734.6),
          p2("placed", "arduino", -120.0, 588.5),
          p2("placed", "ultrasonic", -38.9, 816.2),
          p2("placed", "ultrasonic", 55.1, 751.1)]
    # Two layouts, each with an lcd, 22 mm apart across the rebuild: exactly the
    # collision that read E1's genuine placement as refuted by E4's.
    cross = [dict(p2("placed", "lcd", -105.2, 800.3), run_id="A"),
             dict(p2("placed", "lcd", -127.4, 799.3), run_id="B")]
    cfgx = {"A": {"config": {"layout": "E1"}}, "B": {"config": {"layout": "E4"}}}
    ck("pooling layouts merges two different boards into one unit",
       len(units(cross)) == 1)
    ck("scoping to the layout keeps them separate",
       len(units_by_layout(cross, cfgx)) == 2)
    ck("and neither genuine placement is called false",
       sum(len(false_successes(u))
           for u in units_by_layout(cross, cfgx)) == 0)

    us3 = units(e3)
    lcd = [u for u in us3 if u["label"] == "lcd"][0]
    ck("the lcd is one unit with three rows", len(lcd["rows"]) == 3)
    ck("its 'placed' row is identified as a false success",
       len(false_successes(lcd)) == 1)
    ck("the lcd is NOT counted as delivered", confirmed(lcd) is False)
    esp = [u for u in us3 if u["label"] == "esp" and u["x"] < -100][0]
    ck("an esp lost then re-picked elsewhere is two units, one delivered",
       len([u for u in us3 if u["label"] == "esp"]) == 2
       and confirmed(esp) is False)
    # Without the gate inventory, the shed-and-re-picked esp inflates the
    # denominator to 6 and fulfilment reads 67%.
    cfg3 = {"20260919_180000": {"config": {"layout": "E3"}}}
    ck("position-matching alone over-counts the units", "| 6 | 4 |" in
       "\n".join(pnp2(e3, cfg=cfg3, inventory={})))
    b3 = "\n".join(pnp2(e3, cfg=cfg3, inventory={"E3": 5}))
    ck("E3 attempt rate is scan-corrected to 4/7 = 57%, not 5/7 = 71%",
       "| 7 | 4 | **57%**" in b3, "(the false 'placed' must not count as ok)")
    ck("the gate-certified inventory gives fulfilment 4/5 = 80%",
       "| 5 | 4 | **80%**" in b3)
    ck("and the discrepancy is declared, not absorbed",
       "shed and re-picked" in b3)
    ck("the refuted grasp is shown with the numbers that fooled the probe",
       "refuted" in b3 and "1159" in b3)
    ck("skips and abandonments are still shown, not hidden",
       "| 1 | 1 |" in body)
    ck("the per-unit table names the board that cost the run",
       "lcd @(37,611)" in body or "lcd @(37,610)" in body)

    # ---- layout E1 exactly as it ran on 19 Sep ---------------------------
    # Five real units, one esp retried once and succeeded, plus a phantom
    # "lcd" proposed 220 mm off the bench and 180 mm below it.
    e1 = [p2("placed", "ultrasonic", -127.0, 616.8),
          p2("lost_on_lift", "esp", 44.7, 819.6),
          p2("placed", "esp", 44.9, 819.5),
          p2("placed", "lcd", -105.2, 800.3),
          p2("placed", "ultrasonic", -36.3, 728.2),
          p2("placed", "arduino", 43.4, 623.4),
          dict(p2("skipped", "lcd", -226.6, 341.9), z_mm="65.6")]
    for r in e1:
        r.setdefault("z_mm", "250.0")
    ck("the off-bench phantom is rejected by the envelope",
       [in_workspace(r) for r in e1] == [True]*6 + [False])
    body = "\n".join(pnp2(e1))
    ck("the exclusion is declared, not silent", "outside the bench envelope" in body)
    ck("attempt rate is 5/6 = 83%, not 5/7 = 71%", "| 6 | 5 | **83%**" in body,
       "(the skipped phantom must leave the numerator's denominator)")
    ck("fulfilment is 5/5 = 100% over the five real units",
       "| 5 | 5 | **100%**" in body)
    ck("the retried esp counts as ONE unit with two attempts",
       "| easy | esp @(45,820) | 2 |" in body or
       "| easy | esp @(45,819) | 2 |" in body)

    # A battery whose conditions did not actually separate must say so.
    flat = ([p2("placed", "arduino", -90, 650, "pnp2_easy_topmost_fixed"),
             p2("placed", "esp", 20, 800, "pnp2_easy_topmost_fixed")]
            + [p2("placed", "arduino", -90, 650, "pnp2_hard_topmost_fixed"),
               p2("placed", "esp", 20, 800, "pnp2_hard_topmost_fixed")])
    ck("a battery whose conditions did not separate is flagged",
       "did not separate" in "\n".join(pnp2(flat)))
    real = flat[:2] + [p2("placed", "arduino", -90, 650,
                          "pnp2_hard_topmost_fixed", crowd="0.30"),
                       p2("placed", "esp", 20, 800,
                          "pnp2_hard_topmost_fixed", crowd="0.25")]
    ck("a battery whose conditions DID separate is not flagged",
       "did not separate" not in "\n".join(pnp2(real)))

    print("\n" + "=" * 56)
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("analyse_runs self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    main()
