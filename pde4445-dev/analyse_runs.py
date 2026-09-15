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
    """Rows where the arm actually tried to pick something."""
    return [r for r in rows if r["outcome"] not in ("skipped",) + NON_PICK]


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
