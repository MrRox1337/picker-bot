#!/usr/bin/env python3
"""pnp_gate.py — decide a layout's CONDITION by measurement, before running it.

WHY THIS EXISTS
---------------
On 16 Sep the pick-and-place battery was split into "easy" and "hard" by the
operator's eye, and the result came out backwards: hard 77%, easy 68%. Three
defects produced that, and two of them are this script's job.

  1. The labels did not describe what was measured. EVERY row in both
     conditions logged occlusion 0.00 — so the intended difficulty axis was
     absent from the experiment entirely. Worse, one whole run in the "hard"
     set (20260916_205719) logged crowding 0.00 on all four parts: it was
     physically an easy layout wearing a hard label.
  2. Nothing recorded what the layout actually was, so the mislabelling was
     only discoverable afterwards by reading the logs.

A condition assigned after the fact from a run's own outcomes is not a
condition, it is a post-hoc grouping. So: scan the layout, measure it, and let
the measurement accept or reject it BEFORE a single pick happens. A layout that
fails the gate gets rebuilt, not run.

THE PRE-REGISTERED CRITERIA (fixed here, in code, before the data exists)

  easy    every part: crowding == 0, occlusion == 0, jaws land on the bench.
          No part may be in contact with any other. This is the "nothing is in
          the way" condition. Surface drop is RECORDED but not judged: measured
          on a static scene it swings by up to 2.2 mm between consecutive
          frames, which is its own threshold, so it classifies nothing. See
          TILT_GATED_CLASSES for the numbers.

  hard    at least MIN_HARD_PARTS parts with crowding >= HARD_CROWDING.
          Jaw-obstructed parts are ALLOWED and expected: refusing them is
          the correct behaviour and is logged 'skipped', which the analysis
          counts against fulfilment but not against the attempt rate.

  tilted  exactly one part, and it must be an lcd. The dedicated probe for the
          one physical board that fails repeatably, identified by the operator
          rather than by a drop estimate that cannot tell it apart.

Exit status is the gate: 0 = PASS (go), 1 = FAIL (rebuild), 2 = error. Every
decision is appended to gates/gates.jsonl with the numbers behind it, so the
condition of every layout run tomorrow is archived rather than remembered.

USAGE
    python pde4445-dev/pnp_gate.py --require easy  --layout E1
    python pde4445-dev/pnp_gate.py --require hard  --layout H1
    python pde4445-dev/pnp_gate.py --require tilted --layout TILT --parts 1
    python pde4445-dev/pnp_gate.py --self-test
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GATES_DIR = os.path.join(HERE, "gates")

# --- the pre-registered thresholds. Do not tune these after seeing results. ---
HARD_CROWDING   = 0.10   # perimeter contact fraction that counts as "crowded"
MIN_HARD_PARTS  = 2      # a single crowded part is an incident, not a condition
TILT_DROP_MM    = 3.0    # surface drop across the grasp width (scan.py's flag)
JAW_MARGIN_MM   = 2.0    # z_land this close to the part top = jaws hit something
DEFAULT_PARTS   = 5      # every battery layout holds five modules
# MUST equal the confidence the RUN uses. The gate certifies a layout on behalf
# of clear_scene.py; if the gate looks at 0.30 and the run looks at 0.55, the
# gate passes five parts and the arm is handed four. 0.55 is also the value
# Chapter 5 defends from the sweep, so a battery run at anything else is not
# measuring the configuration the report describes.
MIN_CONF        = 0.55

# Classes the tilt criterion is allowed to reject a layout over.
#
# WHAT tilt_drop_mm ACTUALLY MEASURES. It fits ONE plane to every depth pixel
# inside a part's mask and reports the fall across the grasp width. That models
# a genuinely flat surface. No module in this population has one: the ultrasonic
# carries two transducer cans ~13 mm proud, the ESP32 a shield can, a USB
# connector and header pins, the Arduino a USB jack and a barrel socket. So the
# number is a PLANARITY RESIDUAL, not a tilt, and it is inflated by anything
# that makes the camera see that structure obliquely.
#
# Measured on a flat bench, 19 Sep, layout E1. Same ultrasonic class, same
# bench, nothing propped up, versus radial distance from the optical axis
# (which meets the bench near robot x=-32, y=758 per handeye.json):
#
#       r =  34 mm -> 1.5 mm      r = 118 mm -> 6.1 mm      r = 171 mm -> 9.4 mm
#
# and an esp read 1.4 mm at one pose and 4.1 mm after being moved and rotated.
# A single global threshold therefore rejects flat parts for being off-centre,
# which is a property of the camera, not of the layout.
#
# AND IT IS NOT REPEATABLE ENOUGH TO GATE ON AT ALL. Five consecutive scans of
# one untouched scene, 19 Sep 13:23-13:38, apparent drop in mm:
#
#     ultrasonic (r=171)   10.2  10.6   9.4  10.7   9.6     range 1.30
#     ultrasonic (r= 34)    1.2   1.3   1.5   1.0   1.5     range 0.50
#     lcd                   1.1   1.1   1.9   3.3   2.2     range 2.20  <-- !
#     arduino               1.7   1.8   1.6   1.7   1.5     range 0.30
#     esp                   3.5   4.1   3.5   4.4           range 0.90
#
# The lcd - the only class this was still scoped to - moves 2.2 mm across
# repeats of a board nobody touched, against a 3.0 mm threshold. Consecutive
# gates on that scene returned FAIL then PASS. A criterion whose repeatability
# is comparable to its own threshold does not classify anything; it samples.
#
# The fit is least-squares over a mask whose depth already carries the part's
# components, so gradient noise is depth noise amplified by the lever arm of
# the mask - which is why the pooled 0.39 mm z noise floor becomes millimetres
# of apparent slope, and why the worst offender is also the most eccentric part.
#
# So tilt is RECORDED on every gate and JUDGED on none. The tilted lcd is kept
# out of the battery by identity - the operator knows which board it is - and
# is characterised in its own probe. Re-enable by naming classes here if a
# repeatable tilt estimate ever exists: multi-frame median, or a fit that
# models the components instead of averaging over them. Both are future work.
TILT_GATED_CLASSES = ()


def classify(picks, require, want_parts, min_conf):
    """Return (passed, reasons, per_part). Pure: no camera, no robot.

    `reasons` lists every criterion that FAILED, so a rebuild is directed
    rather than guessed at. An empty list means the layout qualifies.
    """
    reasons, rows = [], []
    for p in picks:
        top = p.get("z_med", p.get("z"))
        zl = p.get("z_land")
        rows.append({
            "label": p.get("label"),
            "x": p.get("x"), "y": p.get("y"),
            "conf": p.get("conf"),
            "crowding": p.get("crowding"),
            "occlusion": p.get("occlusion"),
            "tilt_drop_mm": p.get("tilt_drop_mm"),
            "jaw_blocked": (zl is not None and top is not None
                            and zl >= top - JAW_MARGIN_MM),
        })

    # --- checks common to every condition -------------------------------
    if want_parts and len(picks) != want_parts:
        reasons.append(f"detected {len(picks)} parts, expected {want_parts} "
                       f"— a layout the vision cannot fully see is not a "
                       f"pick trial, it is a detection failure")
    low = [r for r in rows if (r["conf"] or 0) < min_conf]
    if low:
        reasons.append(f"{len(low)} part(s) below conf {min_conf}: "
                       + ", ".join(f"{r['label']} {r['conf']}" for r in low))

    def val(r, k):
        v = r.get(k)
        return 0.0 if v is None else float(v)

    def tilted(rs):
        """Only the classes the criterion is scoped to — see TILT_GATED_CLASSES."""
        return [r for r in rs if r["label"] in TILT_GATED_CLASSES
                and val(r, "tilt_drop_mm") >= TILT_DROP_MM]

    if require == "easy":
        crowded = [r for r in rows if val(r, "crowding") > 0]
        if crowded:
            reasons.append("not isolated: " + ", ".join(
                f"{r['label']} crowding {val(r,'crowding'):.3f}" for r in crowded))
        occ = [r for r in rows if val(r, "occlusion") > 0]
        if occ:
            reasons.append("shadowed: " + ", ".join(
                f"{r['label']} occlusion {val(r,'occlusion'):.3f}" for r in occ))
        blocked = [r for r in rows if r["jaw_blocked"]]
        if blocked:
            reasons.append("jaws would land on something: "
                           + ", ".join(r["label"] for r in blocked))
        bad = tilted(rows)
        if bad:
            reasons.append("tilted board(s) in an easy layout: " + ", ".join(
                f"{r['label']} drops {val(r,'tilt_drop_mm'):.1f}mm" for r in bad)
                + " — the tilted board is excluded from the battery by design "
                  "and runs as its own 'tilted' probe")

    elif require == "hard":
        crowded = [r for r in rows if val(r, "crowding") >= HARD_CROWDING]
        if len(crowded) < MIN_HARD_PARTS:
            reasons.append(
                f"only {len(crowded)} part(s) at crowding >= {HARD_CROWDING}, "
                f"need {MIN_HARD_PARTS} — this layout is not measurably harder "
                f"than an easy one, and running it would repeat the 16 Sep "
                f"mislabelling")
        bad = tilted(rows)
        if bad:
            reasons.append("tilted board(s) present: " + ", ".join(
                r["label"] for r in bad)
                + " — keep the tilted board out so crowding stays the only "
                  "axis separating easy from hard")

    elif require == "tilted":
        # Identity, not measurement. The drop estimate is not repeatable enough
        # to confirm which board this is (see TILT_GATED_CLASSES), so the probe
        # gates only on "one part, clearly seen" and the operator asserts that
        # it is the header-propped lcd. The measured drop is still archived, so
        # the claim can be checked against the record afterwards.
        if len(rows) == 1 and rows[0]["label"] not in ("lcd",):
            reasons.append(f"the tilted probe is for the lcd with the rear "
                           f"header; this is a {rows[0]['label']}")

    else:
        reasons.append(f"unknown condition {require!r}")

    return (not reasons), reasons, rows


def show(rows, require):
    print(f"\n  {'label':11} {'conf':>5} {'X':>8} {'Y':>8} {'crowd':>7} "
          f"{'occl':>7} {'drop':>6}  flags")
    for r in rows:
        flags = []
        if r["jaw_blocked"]:
            flags.append("JAW-BLOCKED")
        d = r.get("tilt_drop_mm")
        if d is not None and float(d) >= TILT_DROP_MM:
            # Same number, different meaning by class: a slope on a flat board
            # is tilt; on a module with tall components it is the module's own
            # structure, which the gate reports but does not judge.
            flags.append(f"TILTED {float(d):.1f}mm"
                         if r["label"] in TILT_GATED_CLASSES
                         else f"(non-planar {float(d):.1f}mm, not tilt)")
        c = r.get("crowding")
        if c is not None and float(c) >= HARD_CROWDING:
            flags.append("CROWDED")

        def f(v, w, p=3):
            return f"{float(v):>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
        print(f"  {str(r['label']):11} {str(r['conf']):>5} "
              f"{f(r['x'],8,1)} {f(r['y'],8,1)} {f(r['crowding'],7)} "
              f"{f(r['occlusion'],7)} {f(r.get('tilt_drop_mm'),6,1)}  "
              + " ".join(flags))


def record(path, entry):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf8") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require", choices=["easy", "hard", "tilted"], required=True)
    ap.add_argument("--layout", default="", help="layout ID, e.g. E1 / H1")
    ap.add_argument("--parts", type=int, default=DEFAULT_PARTS,
                    help=f"expected part count (default {DEFAULT_PARTS}, 0 = any)")
    ap.add_argument("--conf", type=float, default=MIN_CONF)
    ap.add_argument("--db3", default=None, help="score a recording instead of the camera")
    args = ap.parse_args()

    from scan import scan                     # late: needs the model and camera
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.require}"
    if args.layout:
        tag += f"_{args.layout}"
    os.makedirs(GATES_DIR, exist_ok=True)
    overlay = os.path.join(GATES_DIR, tag + ".png")
    picks, _ = scan(args.db3, conf=args.conf, save_as=overlay)

    passed, reasons, rows = classify(picks, args.require, args.parts, args.conf)
    print(f"\nGATE: {args.require.upper()}"
          + (f"  layout {args.layout}" if args.layout else "")
          + f"  ({len(picks)} parts)  -> gates/{tag}.png")
    show(rows, args.require)

    if passed:
        print(f"\n  PASS — this layout qualifies as '{args.require}'. Run it.\n")
    else:
        print(f"\n  FAIL — do NOT run this as '{args.require}':")
        for r in reasons:
            print(f"    - {r}")
        print("  Rebuild the layout and re-gate.\n")

    record(os.path.join(GATES_DIR, "gates.jsonl"), {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layout": args.layout, "require": args.require,
        "passed": passed, "reasons": reasons,
        "conf": args.conf, "n_parts": len(picks),
        "overlay": os.path.relpath(overlay, HERE),
        "parts": rows,
        "thresholds": {"hard_crowding": HARD_CROWDING,
                       "min_hard_parts": MIN_HARD_PARTS,
                       "tilt_drop_mm": TILT_DROP_MM,
                       "tilt_applies_to": list(TILT_GATED_CLASSES),
                       "jaw_margin_mm": JAW_MARGIN_MM,
                       "min_conf": MIN_CONF},
    })
    return 0 if passed else 1


# ----------------------------------------------------------------- self-test
def self_test():
    """Score the gate against the 16 Sep layouts it was written to catch."""
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, wanted {want!r}")

    def part(label, x, y, crowd=0.0, occl=0.0, drop=0.0, zmed=250.0,
             zland=210.0, conf=0.9):
        return {"label": label, "x": x, "y": y, "conf": conf, "crowding": crowd,
                "occlusion": occl, "tilt_drop_mm": drop, "z_med": zmed,
                "z_land": zland}

    # --- a genuinely easy layout: run_20260916_203449, all crowding 0.00 ----
    easy = [part("esp", -23.9, 819.2), part("lcd", 60.5, 748.3),
            part("ultrasonic", 53.4, 589.7), part("ultrasonic", -126.4, 773.4),
            part("arduino", -93.8, 651.5)]
    ok, why, _ = classify(easy, "easy", 5, 0.55)
    check("a real easy layout passes", (ok, why), (True, []))

    # The same layout must NOT be accepted as hard. This is the 205719 defect:
    # four parts, every crowding 0.00, labelled 'hard' by eye and run anyway.
    ok, why, _ = classify(easy, "hard", 5, 0.55)
    check("an easy layout is refused as hard", ok, False)
    check("and the reason names crowding", "crowding" in " ".join(why), True)

    # --- a genuinely hard layout: run_20260916_210328 crowding values -------
    hard = [part("lcd", -93.3, 698.3, crowd=0.435, zland=269.0, zmed=250.0),
            part("ultrasonic", -125.9, 708.5, crowd=0.203),
            part("esp", 48.3, 726.6, crowd=0.470),
            part("lcd", -114.3, 704.0, crowd=0.117),
            part("arduino", 29.0, 748.7, crowd=0.0)]
    ok, why, _ = classify(hard, "hard", 5, 0.55)
    check("a real hard layout passes", (ok, why), (True, []))
    # ...and its jaw-blocked part is allowed, because refusing it is correct
    # behaviour that the run logs as 'skipped'.
    ok2, _, rows = classify(hard, "hard", 5, 0.55)
    check("jaw-blocked part detected", sum(r["jaw_blocked"] for r in rows), 1)
    check("jaw-blocked part does not fail a hard gate", ok2, True)
    # the same layout must be refused as easy
    ok, _, _ = classify(hard, "easy", 5, 0.55)
    check("a hard layout is refused as easy", ok, False)

    # The tilted lcd is kept out of the battery by IDENTITY, not by the drop
    # estimate, so a steep reading no longer changes any verdict.
    with_tilt = easy[:4] + [part("lcd", 37.3, 610.5, drop=4.2)]
    ok, why, _ = classify(with_tilt, "easy", 5, 0.55)
    check("a steep drop no longer decides an easy gate", (ok, why), (True, []))

    # --- surface drop is RECORDED, NEVER JUDGED ---------------------------
    # Layout E1 as actually scanned on 19 Sep. Every one of these parts was
    # flat on the bench; the drops are viewing geometry and frame noise.
    e1 = [part("esp", -90.2, 673.1, drop=1.4, conf=0.62),
          part("ultrasonic", -147.2, 783.6, drop=6.1, conf=0.985),
          part("arduino", 32.5, 662.7, drop=2.8, conf=0.969),
          part("lcd", -44.9, 836.3, drop=1.4, conf=0.978),
          part("ultrasonic", 60.5, 819.0, drop=2.9, conf=0.944)]
    ok, why, _ = classify(e1, "easy", 5, 0.55)
    check("a non-planar ultrasonic does not fail an easy gate", (ok, why), (True, []))

    e1b = [part("ultrasonic", -126.8, 616.6, drop=9.4, conf=0.888),
           part("esp", 45.1, 819.4, drop=4.1, conf=0.75),
           part("ultrasonic", -39.2, 725.2, drop=1.5, conf=0.984),
           part("lcd", -105.2, 800.4, drop=1.9, conf=0.976),
           part("arduino", 44.0, 622.1, drop=1.6, conf=0.963)]
    ok, why, _ = classify(e1b, "easy", 5, 0.55)
    check("an off-axis ultrasonic and a rotated esp both pass",
          (ok, why), (True, []))

    # THE ONE THAT MATTERS. The same untouched lcd sampled 1.1, 1.1, 1.9, 3.3
    # and 2.2 mm across five consecutive scans. Under the old 3.0 mm rule the
    # gate returned FAIL then PASS on a board nobody had touched. Every one of
    # those samples must now give the same verdict.
    verdicts = set()
    for d in (1.1, 1.1, 1.9, 3.3, 2.2):
        scene = e1b[:3] + [part("lcd", -105.2, 800.4, drop=d, conf=0.975),
                           part("arduino", 44.0, 622.1, drop=1.6, conf=0.963)]
        verdicts.add(classify(scene, "easy", 5, 0.55)[0])
    check("repeat scans of one untouched scene give ONE verdict",
          verdicts, {True})

    # The tilted probe now gates on identity, not on the drop estimate.
    ok, why, _ = classify([part("lcd", 37.3, 610.5, drop=1.1)], "tilted", 1, 0.55)
    check("the tilted probe accepts the lcd whatever the drop reads",
          (ok, why), (True, []))
    ok, _, _ = classify([part("ultrasonic", -126.8, 616.6, drop=9.4)],
                        "tilted", 1, 0.55)
    check("the tilted probe rejects a non-lcd however steep it reads", ok, False)

    # --- the gate must look at the same confidence the RUN uses ------------
    check("gate confidence matches the operating threshold", MIN_CONF, 0.55)

    # --- an undetected part is a detection failure, not a pick trial -------
    ok, why, _ = classify(easy[:4], "easy", 5, 0.55)
    check("a missing part fails the gate", ok, False)
    check("and is named as such", "expected 5" in " ".join(why), True)

    # --- a low-confidence detection fails ----------------------------------
    shaky = easy[:4] + [part("arduino", -93.8, 651.5, conf=0.41)]
    ok, why, _ = classify(shaky, "easy", 5, 0.55)
    check("a sub-threshold detection fails the gate", ok, False)

    # --- one crowded part is an incident, not a hard condition -------------
    almost = easy[:4] + [part("arduino", 29.0, 748.7, crowd=0.30)]
    ok, _, _ = classify(almost, "hard", 5, 0.55)
    check("one crowded part is not enough for hard", ok, False)

    for f in fails:
        print("FAIL " + f)
    if fails:
        return 1
    print("pnp_gate self-test passed — the gate accepts the 16 Sep layouts that "
          "were genuinely easy and genuinely hard, and refuses the one that was "
          "labelled hard with every crowding at zero")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(main())
