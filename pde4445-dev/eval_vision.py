"""
OFFLINE VISION BATTERY — score the detector over every recording, with no robot.

This is the "capture once, evaluate offline" half of the evaluation. Every number
the report makes about perception comes from here, and it can be re-run after any
model or threshold change without booking the lab.

    python pde4445-dev/eval_vision.py                  # everything
    python pde4445-dev/eval_vision.py --limit 4        # quick smoke run
    python pde4445-dev/eval_vision.py --sweep          # + confidence sweep

WHAT IT MEASURES, AND WHAT IT DOES NOT

Ground truth comes from the `note` column of captures/manifest.csv, which records
what was on the bench: "2 arduino, 1 lcd, 1 esp, 1 ultrasonic". That gives COUNTS
per class per scene, so this computes count-based recall and over-detection.

It is NOT precision/recall in the detection-benchmark sense. Without per-instance
boxes there is no IoU, so a scene can score a perfect count while every mask is on
the wrong object. Two guards on that: the per-scene overlays are written out so
the counts can be spot-checked visually, and the pose-repeatability section below
would blow up if masks were landing on different objects between repeats. Say so
in the report - a count-based metric honestly labelled beats an IoU number that
was never computed.

FOUR THINGS IT PRODUCES

1. Detection accuracy per CLASS and per CLUTTER CONDITION (scattered / regular /
   bunched / adversarial). The condition axis is the point: the thesis claims
   clutter is what makes this hard, so the metric has to be resolved by clutter.
2. A confidence sweep, so 0.55 is defended by a curve rather than by one
   afternoon's impression.
3. Pose repeatability. Each condition was recorded ~10 times seconds apart with
   the parts untouched, so the spread of a given part's (x, y, z) across those
   repeats is the perception noise floor - the precision that any pick accuracy
   figure has to be read against.
4. ORDERING DIVERGENCE. How often does topmost-first actually produce a different
   sequence from the naive raster baseline? If they mostly agree, the Pillar-1
   experiment has no power and Saturday's trials cannot show anything, whatever
   the success rates. Better to learn that at a desk on Thursday.

OUTPUTS -> pde4445-dev/vision_eval/
    per_scene.csv        one row per recording
    per_class.csv        pooled by class
    per_condition.csv    pooled by clutter condition
    sweep.csv            confidence curve (with --sweep)
    repeatability.csv    pose spread within each condition
    ordering.csv         topmost vs naive, per scene
    overlays/            annotated frame per scene, for spot-checking
    summary.md           the tables, ready to paste into the report
"""
import os, sys, csv, json, time, argparse, re, statistics as stats
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MANIFEST = os.path.join(HERE, "captures", "manifest.csv")
OUTDIR   = os.path.join(HERE, "vision_eval")
CLASSES  = ["arduino", "esp", "lcd", "ultrasonic"]

# The manifest was written on the lab PC, so its db3 paths are Windows absolute
# paths. Re-root them against wherever this file actually lives, so the battery
# runs from any checkout.
def reroot(p):
    if os.path.exists(p):
        return p
    tail = re.split(r"[\\/]pde4445-dev[\\/]", p)
    if len(tail) == 2:
        cand = os.path.join(HERE, tail[1].replace("\\", os.sep))
        if os.path.exists(cand):
            return cand
    return p


# --------------------------------------------------------------- ground truth
def parse_note(note, baseline):
    """'2 arduino, 1 lcd, 1 esp, 1 ultrasonic' -> {'arduino':2, ...}.

    'same 5 - grid layout' inherits the baseline: the notes were written by a
    human describing a repeat, and re-typing the counts every time is exactly how
    ground truth drifts out of step with the bench.
    """
    counts = {c: 0 for c in CLASSES}
    found = False
    for n, name in re.findall(r"(\d+)\s*(arduino|esp32|esp|lcd|ultrasonic)", note.lower()):
        counts["esp" if name == "esp32" else name] += int(n)
        found = True
    if not found:
        counts = dict(baseline)
    distractors = bool(re.search(r"unseen|distract|foreign|novel", note.lower()))
    return counts, distractors


def load_manifest(path, limit=None):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    baseline = {c: 0 for c in CLASSES}
    for r in rows:                                  # first fully-specified note wins
        c, _ = parse_note(r["note"], baseline)
        if sum(c.values()):
            baseline = c
            break
    scenes = []
    for r in rows:
        counts, distractors = parse_note(r["note"], baseline)
        scenes.append({
            "scene": os.path.splitext(os.path.basename(r["db3"].replace("\\", "/")))[0],
            "condition": r["condition"],
            "db3": reroot(r["db3"].replace("\\", "/") if os.sep == "/" else r["db3"]),
            "note": r["note"],
            "truth": counts,
            "distractors": distractors,
        })
    return scenes[:limit] if limit else scenes


# ------------------------------------------------------------------- scoring
def score(truth, picks):
    """Count-based agreement for one scene."""
    got = {c: 0 for c in CLASSES}
    for p in picks:
        got[p["label"]] = got.get(p["label"], 0) + 1
    hit  = {c: min(truth[c], got.get(c, 0)) for c in CLASSES}
    miss = {c: truth[c] - hit[c] for c in CLASSES}
    over = {c: max(0, got.get(c, 0) - truth[c]) for c in CLASSES}
    return got, hit, miss, over


def kendall_tau(a, b):
    """Rank correlation between two orderings of the same items. 1.0 = identical."""
    idx = {v: i for i, v in enumerate(b)}
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            # a[i] comes before a[j] in a. Concordant means it also comes before
            # in b, i.e. its index in b is SMALLER. Getting this sign backwards
            # reports every identical pair of orderings as perfectly reversed.
            s = idx[a[i]] - idx[a[j]]
            conc += s < 0
            disc += s > 0
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def orderings(picks):
    """topmost-first vs the depth-agnostic raster baseline, over the same picks."""
    keyed = [(i, p) for i, p in enumerate(picks)]
    topmost = [i for i, p in sorted(keyed, key=lambda kp: -kp[1]["z"])]
    naive   = [i for i, p in sorted(keyed, key=lambda kp: (kp[1].get("v", 0),
                                                          kp[1].get("u", 0)))]
    return topmost, naive


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Offline vision battery.")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--conf", type=float, default=0.55,
                    help="operating threshold to report as the headline (default 0.55)")
    ap.add_argument("--base-conf", type=float, default=0.20,
                    help="run the model once at this low threshold and filter upward. "
                         "Equivalent to re-running per threshold, ~6x faster.")
    ap.add_argument("--sweep", action="store_true", help="also emit the confidence curve")
    ap.add_argument("--sweep-points", default="0.25,0.35,0.45,0.55,0.65,0.75")
    ap.add_argument("--exclude", nargs="*", default=["static"],
                    help="conditions to leave OUT of the detection benchmark. "
                         "'static' is captured for pose repeatability, not recall: it "
                         "is one scene shot repeatedly and its manifest note carries no "
                         "counts, so the ground truth falls back to the 5-part baseline "
                         "and every missing part is phantom. Scoring it as detection "
                         "invented 4 arduino misses that were never on the bench.")
    ap.add_argument("--limit", type=int, default=None, help="first N recordings only")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--stack-mm", type=float, default=12.0,
                    help="z spread above which a scene counts as genuinely stacked "
                         "rather than five flat parts of differing heights (default 12)")
    ap.add_argument("--static-mm", type=float, default=20.0,
                    help="if a class moves more than this between consecutive "
                         "recordings, they are independent layouts and repeatability "
                         "is not measurable from them (default 20)")
    ap.add_argument("--match-mm", type=float, default=25.0,
                    help="repeatability: how far a detection may sit from a track's "
                         "mean and still be the same part (default 25)")
    ap.add_argument("--from-cache", action="store_true",
                    help="recompute every report from detections.json without running "
                         "the model. Metric definitions change more often than the "
                         "detector does; this makes that free.")
    args = ap.parse_args()

    if args.from_cache:
        cache = os.path.join(args.outdir, "detections.json")
        if not os.path.exists(cache):
            raise SystemExit(f"No cache at {cache} — run once without --from-cache first.")
        results = json.load(open(cache, encoding="utf-8"))
        if args.exclude:
            n0 = len(results)
            results = [r for r in results if r["condition"] not in args.exclude]
            if n0 != len(results):
                print(f"excluding {n0 - len(results)} cached recording(s) from "
                      f"{args.exclude} — not detection benchmarks")
        for s in results:
            at = [p for p in s["picks"] if p["conf"] >= args.conf]
            g, h, m, o = score(s["truth"], at)
            s.update(got=g, hit=h, miss=m, over=o, n_at_conf=len(at))
        print(f"{len(results)} scenes from cache @ conf {args.conf}")
        emit(args, results)
        return

    if not os.path.exists(args.manifest):
        raise SystemExit(f"No manifest at {args.manifest}")
    scenes = load_manifest(args.manifest, args.limit)
    if args.exclude:
        before = len(scenes)
        scenes = [s_ for s_ in scenes if s_["condition"] not in args.exclude]
        if before != len(scenes):
            print(f"excluding {before - len(scenes)} recording(s) from conditions "
                  f"{args.exclude} — not detection benchmarks")
    os.makedirs(os.path.join(args.outdir, "overlays"), exist_ok=True)
    print(f"{len(scenes)} recordings from {args.manifest}")
    print(f"ground truth per scene: "
          + ", ".join(f"{k}={v}" for k, v in scenes[0]["truth"].items() if v))

    import cv2
    from scan import frame_db3, picks_from_frame
    from pose_seg import load_handeye, HANDEYE
    R, t = load_handeye(HANDEYE if os.path.isabs(HANDEYE) else os.path.join(HERE, "handeye.json"))

    results = []
    for i, s in enumerate(scenes, 1):
        if not os.path.exists(s["db3"]):
            print(f"  [{i}/{len(scenes)}] MISSING {s['db3']}")
            continue
        try:
            t0 = time.time()
            color, depth, intr = frame_db3(s["db3"])
            t_read = time.time() - t0
            t0 = time.time()
            picks, vis = picks_from_frame(color, depth, intr, R, t, conf=args.base_conf)
            t_infer = time.time() - t0
        except Exception as e:
            print(f"  [{i}/{len(scenes)}] FAILED {s['scene']}: {type(e).__name__}: {e}")
            continue

        cv2.imwrite(os.path.join(args.outdir, "overlays", s["scene"] + ".png"), vis)
        s.update(picks=picks, t_read=t_read, t_infer=t_infer)
        at_conf = [p for p in picks if p["conf"] >= args.conf]
        got, hit, miss, over = score(s["truth"], at_conf)
        s.update(got=got, hit=hit, miss=miss, over=over, n_at_conf=len(at_conf))
        results.append(s)
        print(f"  [{i}/{len(scenes)}] {s['scene']:<38} {s['condition']:<12} "
              f"found {len(at_conf)}/{sum(s['truth'].values())}  "
              f"miss {sum(miss.values())}  extra {sum(over.values())}  "
              f"{t_infer*1000:.0f}ms")

    if not results:
        raise SystemExit("nothing scored — check the db3 paths in the manifest")

    # Cache the detections so no future metric change costs another 35 inferences.
    json.dump([{k: v for k, v in s.items()
                if k in ("scene", "condition", "note", "truth", "distractors",
                         "picks", "t_read", "t_infer")} for s in results],
              open(os.path.join(args.outdir, "detections.json"), "w"), indent=1)
    emit(args, results)


def emit(args, results):
    write_per_scene(args, results)
    write_per_class(args, results)
    write_per_condition(args, results)
    write_ordering(args, results)
    write_repeatability(args, results)
    sweep = write_sweep(args, results) if args.sweep else None
    write_summary(args, results, sweep)
    print(f"\nWrote {args.outdir}. Start with summary.md, then spot-check overlays/.")


# ------------------------------------------------------------------ reporting
def infer_times(res):
    """Inference times with the warm-up excluded.

    The first inference of a session pays for weight loading and CUDA/JIT setup -
    measured here at 6.3 s against a steady state of ~120 ms. Averaging it in
    inflated the first condition's mean to 745 ms and the overall mean to 299 ms,
    neither of which describes what the robot experiences on its second pick.
    Reported as a MEDIAN as well, so one stall cannot move the headline.
    """
    t = [s["t_infer"] for s in res]
    return t[1:] if len(t) > 2 else t


def write_per_scene(args, res):
    path = os.path.join(args.outdir, "per_scene.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "condition", "expected", "detected", "missed", "extra",
                    "infer_ms"] + [f"miss_{c}" for c in CLASSES]
                   + [f"extra_{c}" for c in CLASSES])
        for s in res:
            w.writerow([s["scene"], s["condition"], sum(s["truth"].values()),
                        s["n_at_conf"], sum(s["miss"].values()), sum(s["over"].values()),
                        round(s["t_infer"] * 1000)]
                       + [s["miss"][c] for c in CLASSES]
                       + [s["over"][c] for c in CLASSES])


def write_per_class(args, res):
    path = os.path.join(args.outdir, "per_class.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "expected", "detected", "recall_pct", "over_detections"])
        for c in CLASSES:
            exp = sum(s["truth"][c] for s in res)
            hit = sum(s["hit"][c] for s in res)
            over = sum(s["over"][c] for s in res)
            w.writerow([c, exp, hit, round(100 * hit / exp, 1) if exp else "", over])


def write_per_condition(args, res):
    path = os.path.join(args.outdir, "per_condition.csv")
    by = defaultdict(list)
    for s in res:
        by[s["condition"]].append(s)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "scenes", "expected", "detected", "recall_pct",
                    "over_detections", "median_infer_ms"]
                   + [f"miss_{c}" for c in CLASSES])
        for cond, ss in by.items():
            exp = sum(sum(s["truth"].values()) for s in ss)
            hit = sum(sum(s["hit"].values()) for s in ss)
            w.writerow([cond, len(ss), exp, hit,
                        round(100 * hit / exp, 1) if exp else "",
                        sum(sum(s["over"].values()) for s in ss),
                        round(1000 * stats.median(infer_times(ss)))]
                       + [sum(s["miss"][c] for s in ss) for c in CLASSES])


def write_ordering(args, res):
    """Does topmost-first actually differ from the naive baseline in these scenes?"""
    path = os.path.join(args.outdir, "ordering.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "condition", "n_parts", "identical_order",
                    "kendall_tau", "z_spread_mm", "stacked"])
        for s in res:
            picks = [p for p in s["picks"] if p["conf"] >= args.conf]
            if len(picks) < 2:
                continue
            top, naive = orderings(picks)
            zs = [p["z"] for p in picks]
            spread = max(zs) - min(zs)
            # A scene only tests the CLAIM if something is physically resting on
            # something else. Five different modules lying flat still span several
            # mm, because an ultrasonic sensor is taller than an LCD - reordering
            # by that is sorting parts by height, not reasoning about occlusion.
            # One board on another lifts it by a board thickness plus components,
            # which is where --stack-mm sits.
            w.writerow([s["scene"], s["condition"], len(picks),
                        int(top == naive), round(kendall_tau(top, naive), 3),
                        round(spread, 1), int(spread >= args.stack_mm)])


def yaw_spread(vals):
    """Spread of a set of yaw readings, modulo 180 degrees.

    PCA returns an UNDIRECTED long axis: the eigenvector's sign is arbitrary, so
    the same board can read as +85 in one frame and -95 in the next. Those are the
    same physical orientation. Taking a plain standard deviation over raw yaw
    counts every sign flip as a ~180 deg error and reports tens of degrees of
    'noise' for a part that never moved. Wrap to the nearest equivalent first.

    This is harmless for picking - a parallel jaw is symmetric, so yaw mod 180 is
    all the arm needs - but it is fatal to the metric.
    """
    a0 = vals[0]
    return stats.pstdev([((v - a0 + 90) % 180) - 90 for v in vals])


def rearrangement(ss):
    """Median distance a class moves between consecutive recordings, in mm.

    Distinguishes 'repeats of one static scene' from 'independent layouts'. The
    35 recordings of 8 Sep turn out to be the latter - the bench was re-arranged
    between every capture, median displacement 72-150 mm - so there is no repeated
    observation of a fixed scene anywhere in the set and the noise floor simply
    cannot be extracted from it. Better to detect that and say so than to emit a
    number computed by matching one physical object to a different one.
    """
    disp = []
    for a, b in zip(ss, ss[1:]):
        for p in a["picks"]:
            cand = [q for q in b["picks"] if q["label"] == p["label"]]
            if cand:
                disp.append(min(((p["x"] - q["x"]) ** 2 +
                                 (p["y"] - q["y"]) ** 2) ** .5 for q in cand))
    return stats.median(disp) if disp else None


def write_repeatability(args, res):
    """Spread of the SAME part's pose across repeats of one condition.

    Only meaningful if the condition really is repeated observations of a static
    bench. Where it is, the spread of one part's pose across those repeats is the
    perception noise floor.

    Matching detections to tracks is where this goes wrong if done casually.
    Three rules, each of which the first version broke:
      * match only WITHIN A CLASS - otherwise an lcd can join an arduino's track
      * match ONE-TO-ONE, cheapest pair first - otherwise two detections in one
        frame both join the same track and the spread measures the gap between
        two different objects
      * anchor on the track's running mean, not on frame 0, so the gate does not
        drift

    A track is only reported if it was found in most repeats; a track that keeps
    vanishing is a detection problem, and averaging its pose spread would quietly
    mix that in with sensor noise.
    """
    path = os.path.join(args.outdir, "repeatability.csv")
    by = defaultdict(list)
    for s in res:
        by[s["condition"]].append(s)
    rows = []
    for cond, ss in by.items():
        if len(ss) < 3:
            continue
        moved = rearrangement([{"picks": [p for p in s["picks"]
                                          if p["conf"] >= args.conf]} for s in ss])
        if moved is not None and moved > args.static_mm:
            rows.append([cond, 0, 0, "", "", "", "", "",
                         f"NOT REPEATS - parts move a median {moved:.0f} mm between "
                         f"recordings - these are independent layouts"])
            continue
        ref = [p for p in ss[0]["picks"] if p["conf"] >= args.conf]
        tracks = [{"label": p["label"], "pts": [p]} for p in ref]

        for s in ss[1:]:
            dets = [p for p in s["picks"] if p["conf"] >= args.conf]
            cand = []
            for ti, tr in enumerate(tracks):
                cx = stats.mean(q["x"] for q in tr["pts"])
                cy = stats.mean(q["y"] for q in tr["pts"])
                for di, p in enumerate(dets):
                    if p["label"] != tr["label"]:
                        continue
                    d2 = (cx - p["x"]) ** 2 + (cy - p["y"]) ** 2
                    if d2 <= args.match_mm ** 2:
                        cand.append((d2, ti, di))
            cand.sort()
            ut, ud = set(), set()
            for d2, ti, di in cand:
                if ti in ut or di in ud:
                    continue
                tracks[ti]["pts"].append(dets[di])
                ut.add(ti); ud.add(di)

        need = max(3, int(0.8 * len(ss)))
        kept = [tr for tr in tracks if len(tr["pts"]) >= need]
        if not kept:
            rows.append([cond, 0, len(tracks), "", "", "", "", "",
                         f"no part was tracked through {need} of {len(ss)} recordings"])
            continue

        def sd(key):
            return round(stats.mean(stats.pstdev([p[key] for p in tr["pts"]])
                                    for tr in kept), 2)

        # Largest frame-to-frame jump in xy. Sensor noise is small and unbiased;
        # a part that was NUDGED between recordings shows one big step. This is
        # what tells the two apart, so it is reported rather than inferred.
        jump = 0.0
        for tr in kept:
            pts = tr["pts"]
            for a, b in zip(pts, pts[1:]):
                jump = max(jump, ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** .5)
        rows.append([cond, len(kept), len(tracks), sd("x"), sd("y"), sd("z"),
                     round(stats.mean(yaw_spread([p["yaw"] for p in tr["pts"]])
                                      for tr in kept), 2),
                     round(jump, 1), ""])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "tracks_kept", "tracks_seen", "sd_x_mm", "sd_y_mm",
                    "sd_z_mm", "sd_yaw_deg", "max_xy_jump_mm", "note"])
        w.writerows(rows)


def write_sweep(args, res):
    """Recall and over-detection against the confidence threshold."""
    path = os.path.join(args.outdir, "sweep.csv")
    points = [float(v) for v in args.sweep_points.split(",")]
    rows = []
    for c in points:
        exp = hit = over = 0
        for s in res:
            picks = [p for p in s["picks"] if p["conf"] >= c]
            g, h, m, o = score(s["truth"], picks)
            exp += sum(s["truth"].values()); hit += sum(h.values()); over += sum(o.values())
        rows.append([c, exp, hit, round(100 * hit / exp, 1) if exp else "", over])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf", "expected", "detected", "recall_pct", "over_detections"])
        w.writerows(rows)
    return rows


def write_summary(args, res, sweep):
    exp = sum(sum(s["truth"].values()) for s in res)
    hit = sum(sum(s["hit"].values()) for s in res)
    over = sum(sum(s["over"].values()) for s in res)
    infer = [t * 1000 for t in infer_times(res)]
    warm = res[0]["t_infer"] * 1000 if len(res) > 2 else None

    L = [f"# Vision battery — {len(res)} recordings @ conf {args.conf}", ""]
    L += [f"- objects expected: **{exp}**, detected: **{hit}** "
          f"(**{100*hit/exp:.1f}%** recall)",
          f"- over-detections: **{over}**",
          f"- inference: median **{stats.median(infer):.0f} ms**, "
          f"mean {stats.mean(infer):.0f} ms, max {max(infer):.0f} ms"
          + (f" (first inference excluded as warm-up: {warm:.0f} ms)" if warm else ""),
          "",
          "Counts, not IoU: a scene can score full marks with masks on the wrong",
          "objects. Spot-check `overlays/` before quoting these.", ""]

    def table(path, title):
        if not os.path.exists(path):
            return []
        rows = list(csv.reader(open(path, encoding="utf-8")))
        if len(rows) < 2:
            return []
        out = [f"## {title}", "", "| " + " | ".join(rows[0]) + " |",
               "|" + "---|" * len(rows[0]) + ""]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
        return out + [""]

    L += table(os.path.join(args.outdir, "per_class.csv"), "By class")
    L += table(os.path.join(args.outdir, "per_condition.csv"), "By clutter condition")
    L += table(os.path.join(args.outdir, "repeatability.csv"),
               "Pose repeatability (perception noise floor)")
    rpath = os.path.join(args.outdir, "repeatability.csv")
    if os.path.exists(rpath):
        rr = list(csv.DictReader(open(rpath, encoding="utf-8")))
        if rr and all("NOT REPEATS" in r.get("note", "") for r in rr):
            L += ["**The noise floor is NOT measured by this dataset.** Every "
                  "condition was re-arranged between captures, so no part is ever "
                  "observed twice in the same place. That makes the 35 recordings a "
                  "STRONGER detection benchmark - 35 independent layouts rather than "
                  "4 layouts seen 10 times each - but it leaves pose precision "
                  "unmeasured.", "",
                  "To fix it, 5 minutes with the camera alone: set one scene, record "
                  "~10 clips WITHOUT touching anything, then re-run this battery over "
                  "that folder. No arm, no gripper.", ""]
    if sweep:
        L += table(os.path.join(args.outdir, "sweep.csv"), "Confidence sweep")

    # the finding that changes Saturday
    opath = os.path.join(args.outdir, "ordering.csv")
    if os.path.exists(opath):
        rows = list(csv.DictReader(open(opath, encoding="utf-8")))
        if rows:
            same = sum(int(r["identical_order"]) for r in rows)
            stacked = [r for r in rows if int(r["stacked"])]
            L += ["## Ordering divergence — does the experiment have any power?", "",
                  f"- scenes where topmost-first and naive give the SAME sequence: "
                  f"**{same}/{len(rows)}**",
                  f"- mean Kendall tau between the two orderings: "
                  f"**{stats.mean(float(r['kendall_tau']) for r in rows):+.2f}** "
                  f"(0 = uncorrelated)",
                  "",
                  "**Divergence alone proves nothing.** Two orderings of a FLAT scene",
                  "differ because the parts have different component heights, not",
                  "because anything is on top of anything. Reordering there is a",
                  "permutation, not depth reasoning, so a trial on such a scene",
                  "compares raster against shuffle and can only measure noise.",
                  "",
                  f"Scenes with real stacking (z spread >= {args.stack_mm:.0f} mm, "
                  f"about one board plus components):", ""]
            by = defaultdict(lambda: [0, 0, []])
            for r in rows:
                b = by[r["condition"]]
                b[0] += int(r["stacked"]); b[1] += 1
                b[2].append(float(r["z_spread_mm"]))
            for cond, (st, n_, sp) in by.items():
                L.append(f"  - {cond}: **{st}/{n_}** stacked   "
                         f"(z spread {min(sp):.1f}–{max(sp):.1f} mm)")
            L += ["",
                  f"**{len(stacked)}/{len(rows)} recorded scenes actually test the claim.**",
                  "",
                  "Acceptance criterion for Saturday: build each layout, run",
                  f"`scan.py --conf 0.55`, and only keep it if all parts are detected",
                  f"AND the z spread is >= {args.stack_mm:.0f} mm. That is a checkable",
                  "gate, unlike 'make them look piled'.", ""]

    path = os.path.join(args.outdir, "summary.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:14]))


# ----------------------------------------------------------------- self-test
def self_test(outdir):
    """Exercise everything except the model, on a synthetic result set.

    The point is narrow but important: this battery runs for minutes before it
    writes anything, so a typo in a report writer would only surface after all 35
    recordings had been processed. This proves the scoring and the six output
    files work first, using the REAL manifest for ground truth.
    """
    import random
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    print("parse_note")
    base = {c: 0 for c in CLASSES}
    c1, d1 = parse_note("2 arduino, 1 lcd, 1 esp, 1 ultrasonic - well separated", base)
    ck("counts a fully-specified note", c1 == {"arduino": 2, "esp": 1, "lcd": 1,
                                               "ultrasonic": 1}, str(c1))
    ck("no distractors flagged", d1 is False)
    c2, _ = parse_note("same 5 - grid layout", c1)
    ck("'same 5' inherits the baseline", c2 == c1, str(c2))
    c3, d3 = parse_note("same 5 + unseen objects", c1)
    ck("unseen objects flagged as distractors", d3 is True)
    ck("and the known counts are unchanged", c3 == c1, str(c3))
    c4, _ = parse_note("1 esp32 only", base)
    ck("esp32 is normalised to esp", c4["esp"] == 1, str(c4))

    print("\nmanifest")
    if os.path.exists(MANIFEST):
        scenes = load_manifest(MANIFEST)
        import csv as _csv
        n = len(list(_csv.DictReader(open(MANIFEST, encoding="utf-8"))))
        ck("loads every row", len(scenes) == n, f"({len(scenes)} of {n})")
        ck("every eval scene has ground truth",
           all(sum(s["truth"].values()) == 5 for s in scenes
               if s["condition"] in ("scattered", "regular", "bunched", "adversarial")),
           str({s["scene"]: sum(s["truth"].values()) for s in scenes
                if sum(s["truth"].values()) != 5}))
        ck("db3 paths resolve on this machine",
           all(os.path.exists(s["db3"]) for s in scenes),
           str([s["db3"] for s in scenes if not os.path.exists(s["db3"])][:2]))
        ck("adversarial scenes carry the distractor flag",
           all(s["distractors"] for s in scenes if s["condition"] == "adversarial"))
    else:
        print(f"  SKIP  no manifest at {MANIFEST}")
        scenes = []

    print("\nscoring")
    truth = {"arduino": 2, "esp": 1, "lcd": 1, "ultrasonic": 1}
    perfect = [{"label": l, "conf": 0.9} for l in
               ["arduino", "arduino", "esp", "lcd", "ultrasonic"]]
    g, h, m, o = score(truth, perfect)
    ck("a perfect scene scores 5 hits, 0 miss, 0 extra",
       (sum(h.values()), sum(m.values()), sum(o.values())) == (5, 0, 0))
    # 4 arduinos + 1 esp against a truth of 2/1/1/1: the two surplus arduinos are
    # over-detections and must NOT be allowed to compensate for the absent lcd and
    # ultrasonic. Total hits stay at 3.
    g, h, m, o = score(truth, perfect[:3] + [{"label": "arduino", "conf": .9}] * 2)
    ck("surplus arduinos cannot substitute for missing classes",
       (sum(h.values()), sum(m.values()), sum(o.values())) == (3, 2, 2),
       f"(hit={sum(h.values())} miss={sum(m.values())} over={sum(o.values())})")

    print("\nordering")
    ck("identical orderings give tau 1.0", kendall_tau([0, 1, 2], [0, 1, 2]) == 1.0)
    ck("reversed orderings give tau -1.0", kendall_tau([0, 1, 2], [2, 1, 0]) == -1.0)
    flat = [{"z": 250, "u": 10, "v": 10}, {"z": 250, "u": 90, "v": 90}]
    top, nai = orderings(flat)
    ck("a flat scene cannot separate the policies", top == nai, f"{top} vs {nai}")
    piled = [{"z": 250, "u": 10, "v": 10}, {"z": 280, "u": 90, "v": 90}]
    top, nai = orderings(piled)
    ck("a piled scene does separate them", top != nai, f"{top} vs {nai}")

    print("\nyaw wrapping")
    ck("a PCA sign flip is not 180 degrees of noise",
       yaw_spread([85.0, -95.0, 85.0, -95.0]) < 0.01,
       f"(got {yaw_spread([85.0, -95.0, 85.0, -95.0]):.2f})")
    ck("but real rotation still registers",
       yaw_spread([0.0, 20.0, 40.0]) > 10.0)

    print("\nreport writers (synthetic results)")
    rnd = random.Random(0)
    res = []
    for cond in ("scattered", "regular", "bunched", "adversarial"):
        for k in range(5):
            # Two arduinos only ~11 mm apart, and the PCA axis flipping sign on
            # alternate frames: the exact conditions that made the first version
            # report 13 mm and 52 deg of "noise" for a bench nobody touched.
            flip = 180.0 * (k % 2)
            picks = [{"label": l, "conf": 0.9,
                      "x": 10.0 * j + rnd.gauss(0, .3), "y": 5.0 * j + rnd.gauss(0, .3),
                      "z": 250.0 + 9 * j + rnd.gauss(0, .3), "yaw": 10.0 * j + flip,
                      "u": 100 * j, "v": 300 - 40 * j}
                     for j, l in enumerate(["arduino", "arduino", "esp", "lcd",
                                            "ultrasonic"])]
            at = picks
            g, h, m, o = score(truth, at)
            res.append(dict(scene=f"{cond}_{k}", condition=cond, truth=truth,
                            picks=picks, t_read=0.1, t_infer=0.05, got=g, hit=h,
                            miss=m, over=o, n_at_conf=len(at)))
    a = argparse.Namespace(outdir=outdir, conf=0.55, sweep_points="0.25,0.55,0.75",
                           stack_mm=12.0, match_mm=25.0, static_mm=20.0)
    os.makedirs(outdir, exist_ok=True)
    write_per_scene(a, res); write_per_class(a, res); write_per_condition(a, res)
    write_ordering(a, res); write_repeatability(a, res)
    sweep = write_sweep(a, res)
    write_summary(a, res, sweep)
    for name in ("per_scene", "per_class", "per_condition", "ordering",
                 "repeatability", "sweep"):
        p = os.path.join(outdir, name + ".csv")
        n = len(list(csv.reader(open(p, encoding="utf-8")))) if os.path.exists(p) else 0
        ck(f"{name}.csv has rows", n > 1, f"({n} lines)")
    md = open(os.path.join(outdir, "summary.md"), encoding="utf-8").read()
    ck("summary.md reports recall", "recall" in md)
    ck("summary.md reports ordering divergence", "Ordering divergence" in md)

    rep = list(csv.DictReader(open(os.path.join(outdir, "repeatability.csv"),
                                   encoding="utf-8")))
    ck("every part is tracked through every repeat",
       all(int(r["tracks_kept"]) == 5 for r in rep),
       str([(r["condition"], r["tracks_kept"]) for r in rep]))
    ck("two same-class parts 11mm apart do not contaminate each other's track",
       all(float(r["sd_x_mm"]) < 1.0 for r in rep),
       str([r["sd_x_mm"] for r in rep]))
    ck("yaw spread survives the sign flips",
       all(float(r["sd_yaw_deg"]) < 1.0 for r in rep),
       str([r["sd_yaw_deg"] for r in rep]))
    ck("no spurious jump is reported for a static scene",
       all(float(r["max_xy_jump_mm"]) < 3.0 for r in rep),
       str([r["max_xy_jump_mm"] for r in rep]))

    print("\n" + "=" * 56)
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("self-test passed — the battery will not fall over after 35 inferences")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        import tempfile
        sys.exit(self_test(os.path.join(tempfile.gettempdir(), "eval_selftest")))
    main()
