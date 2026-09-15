"""
BILL OF QUANTITIES — turn "fetch me these parts" into a pick plan.

This is the use case the whole system is framed around: an operator asks for a
set of modules by class and quantity, and the robot retrieves exactly those from
a cluttered bench, leaving the rest alone.

    python pde4445-dev/boq_fetch.py                          # plan from the last scan
    python pde4445-dev/boq_fetch.py --db3 captures/.../x.db3  # plan from a recording
    python pde4445-dev/boq_fetch.py --live                   # scan the bench now
    python pde4445-dev/boq_fetch.py --run --grasp-dz 25      # plan, then execute it

BOQ FORMAT — pde4445-dev/boq.txt, one line per requirement, '#' comments:

    arduino: 1
    esp32:   1
    lcd:     1

WHAT THIS ADDS OVER "clear the whole bench"
Clearing is indiscriminate: every part is a target, so selection is never tested.
A BOQ makes the system commit to a decision it can get WRONG - fetching the wrong
class, fetching too many, or reporting success when a requirement was never met.
Those are the errors an operator actually cares about, so they are what gets
reported here: fulfilled, short, and unfulfillable, each with a reason.

SELECTION IS STILL TOPMOST-FIRST
Within the parts that satisfy a requirement, the deepest-buried instance is the
worst one to take: it is under the others. So candidates are ordered by base Z
exactly as in the clearing loop, and the requirement is filled from the top down.
The BOQ narrows WHICH parts are eligible; it does not change the sequencing
claim being tested.

HONESTY ABOUT WHAT 'FULFILLED' MEANS
Only classes with a measured thickness signature can be confirmed from the servo.
For the rest the plan is marked vision-verified, and the run is only complete when
the verification re-scan shows the part gone. See clear_scene.verification_mode.
"""
import os, sys, json, argparse, subprocess, re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pick_one import PROPRIOCEPTIVE_CLASSES                       # noqa: E402
from pose_seg import NAMES                                        # noqa: E402
NAMES_KNOWN = set(NAMES.values())

BOQ_FILE  = os.path.join(HERE, "boq.txt")
PICK_LIST = os.path.join(HERE, "pick_list.json")

# Operators write what is printed on the board; the model has its own class names.
ALIASES = {"esp32": "esp", "esp-32": "esp", "nodemcu": "esp",
           "uno": "arduino", "arduino uno": "arduino",
           "lcd1602": "lcd", "display": "lcd",
           "hcsr04": "ultrasonic", "hc-sr04": "ultrasonic", "sonar": "ultrasonic"}

DEFAULT_BOQ = """# Bill of quantities - what the operator asked for.
# One 'class: quantity' per line. Names are matched loosely (esp32 -> esp).
arduino: 1
esp32:   1
lcd:     1
"""


def canon(name):
    n = name.strip().lower().replace("_", " ")
    return ALIASES.get(n, n)


def load_boq(path):
    """Parse the BOQ. Creates a starter file if none exists rather than failing."""
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").write(DEFAULT_BOQ)
        print(f"No BOQ found — wrote a starter one at {path}. Edit it and re-run.")
    want, order = {}, []
    for lineno, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^(.+?)\s*[:=]\s*(\d+)\s*$", line)
        if not m:
            m2 = re.match(r"^(\d+)\s*[xX]?\s+(.+?)$", line)       # "2 arduino"
            if not m2:
                print(f"  [boq] line {lineno} not understood, skipped: {raw.strip()!r}")
                continue
            cls, qty = canon(m2.group(2)), int(m2.group(1))
        else:
            cls, qty = canon(m.group(1)), int(m.group(2))
        if cls not in want:
            order.append(cls)
        want[cls] = want.get(cls, 0) + qty
    return want, order


def parse_request(text):
    """Parse one typed line of wants: 'arduino=1 esp 2, lcd:1' -> dict + order.

    Deliberately forgiving about separators and word order. The operator is
    standing at a bench reading a screen, not writing a config file, and a demo
    that rejects 'arduino 1, esp 1' because it wanted '=' is a bad demo.
    """
    want, order = {}, []
    for chunk in re.split(r"[,;]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = (re.match(r"^(.+?)\s*[:=]\s*(\d+)$", chunk)
             or re.match(r"^(.+?)\s+(\d+)$", chunk)
             or re.match(r"^(\d+)\s*[xX]?\s+(.+)$", chunk))
        if not m:
            cls, qty = canon(chunk), 1           # bare class name means one
        elif m.group(1).isdigit():
            cls, qty = canon(m.group(2)), int(m.group(1))
        else:
            cls, qty = canon(m.group(1)), int(m.group(2))
        if qty <= 0:
            continue
        if cls not in want:
            order.append(cls)
        want[cls] = want.get(cls, 0) + qty
    return want, order


def show_inventory(picks, conf):
    """What the robot can see, before anyone says what they want."""
    by = {}
    for p in picks:
        if p.get("conf", 1) >= conf:
            by.setdefault(p["label"], []).append(p)
    print("\n--- ON THE BENCH, AS THE ROBOT SEES IT ---")
    if not by:
        print("  nothing detected above conf %.2f" % conf)
        return by
    print(f"  {'class':12} {'qty':>4}  {'confidence':>10}   heights (mm)")
    for cls in sorted(by, key=lambda c: -len(by[c])):
        ps = sorted(by[cls], key=lambda p: -p["z"])
        cf = ", ".join(f"{p.get('conf', 0):.2f}" for p in ps)
        hz = ", ".join(f"{p['z']:.0f}" for p in ps)
        print(f"  {cls:12} {len(ps):>4}  {cf:>10}   {hz}")
    print("\n  Anything on the bench that is NOT listed here is invisible to the")
    print("  robot - either an unknown object, or a detection it declined.")
    return by


def ask_request(available):
    """Prompt the operator for a bill, checked against what is actually visible."""
    print("\n--- WHAT DO YOU NEED? ---")
    print("  e.g.  arduino 1, esp 1, lcd 1        (blank line to cancel)")
    while True:
        raw = input("  > ").strip()
        if not raw:
            return None, None
        want, order = parse_request(raw)
        if not want:
            print("  could not read that - try 'arduino 1, esp 1'")
            continue
        unknown = [c for c in want if c not in NAMES_KNOWN]
        if unknown:
            print(f"  not a class this system knows: {', '.join(unknown)}")
            print(f"  known classes: {', '.join(sorted(NAMES_KNOWN))}")
            continue
        short = {c: (n, len(available.get(c, [])))
                 for c, n in want.items() if len(available.get(c, [])) < n}
        if short:
            for c, (asked, got) in short.items():
                print(f"  ** {c}: you asked for {asked}, the robot can see {got}")
            print("  Proceed anyway? It will fetch what it can and report the shortfall.")
            if not input("  [y/N] ").strip().lower().startswith("y"):
                continue
        return want, order


def plan(want, order, picks, conf):
    """Choose which detections satisfy the BOQ. Returns (chosen, report)."""
    usable = [p for p in picks if p.get("conf", 1) >= conf]
    by = {}
    for p in usable:
        by.setdefault(p["label"], []).append(p)
    for v in by.values():
        v.sort(key=lambda p: -p["z"])            # topmost-first within the class

    chosen, report = [], []
    for cls in order:
        need = want[cls]
        avail = by.get(cls, [])
        take = avail[:need]
        chosen += take
        verify = "servo" if cls in PROPRIOCEPTIVE_CLASSES else "re-scan"
        if len(take) == need:
            status = "fulfilled"
        elif take:
            status = "short"
        else:
            status = "unavailable"
        report.append({"class": cls, "requested": need, "available": len(avail),
                       "selected": len(take), "status": status, "verify": verify})

    # Anything detected that the BOQ did NOT ask for must be left alone. Reporting
    # it is the point: a fetch task is as much about what you do not touch.
    leave = [p for p in usable if p not in chosen]
    return chosen, report, leave


def show(report, chosen, leave, want):
    print("\n--- BILL OF QUANTITIES ---")
    print(f"  {'class':12} {'want':>5} {'seen':>5} {'take':>5}  {'status':12} verified by")
    for r in report:
        print(f"  {r['class']:12} {r['requested']:>5} {r['available']:>5} "
              f"{r['selected']:>5}  {r['status']:12} {r['verify']}")

    print("\n--- PICK PLAN (topmost-first within each requirement) ---")
    if not chosen:
        print("  nothing to pick")
    for i, p in enumerate(chosen, 1):
        print(f"  {i}. {p['label']:10} conf {p.get('conf','?'):<5} "
              f"({p['x']:>7.1f}, {p['y']:>7.1f}, {p['z']:>6.1f})"
              + (f"  z_land {p['z_land']:.1f}" if p.get("z_land") is not None else ""))

    if leave:
        print(f"\n--- LEAVE ON THE BENCH ({len(leave)}) ---")
        for p in leave:
            print(f"     {p['label']:10} ({p['x']:>7.1f}, {p['y']:>7.1f})")

    blocked = [p for p in chosen if p.get("under")]
    if blocked:
        print("\n--- BLOCKED, WILL BE UNCOVERED FIRST ---")
        for p in blocked:
            print(f"     {p['label']} is under: {', '.join(p['under'])}"
                  f"   (occlusion {p.get('occlusion', 0):.2f})")
        print("  Those parts are NOT part of the order. They will be moved out of")
        print("  the way, logged as blocker_cleared, and not counted as delivered.")

    short = [r for r in report if r["status"] != "fulfilled"]
    if short:
        print("\n  ** BOQ CANNOT BE FULFILLED FROM THIS SCENE **")
        for r in short:
            print(f"     {r['class']}: asked {r['requested']}, only "
                  f"{r['available']} detected")
        print("  A missing part is not necessarily absent — it may be undetected.")
        print("  Detection recall drops to 74% when parts touch (see vision_eval),")
        print("  and same-class neighbours are the usual cause. Re-scan after the")
        print("  first few picks: removing one part often reveals the next.")
    else:
        print("\n  BOQ fully satisfiable from the current scene.")


def main():
    ap = argparse.ArgumentParser(description="Plan a BOQ fetch from the current scene.")
    ap.add_argument("--boq", default=BOQ_FILE)
    ap.add_argument("--db3", default=None, help="plan from a recording")
    ap.add_argument("--live", action="store_true", help="scan the bench now")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="scan first, show the operator what the robot can see, then "
                         "ask what is needed. The bill is written against the actual "
                         "inventory instead of a file that may not match the bench.")
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--out", default=os.path.join(HERE, "boq_plan.json"))
    ap.add_argument("--run", action="store_true",
                    help="after planning, hand the plan to clear_scene and execute it")
    ap.add_argument("--grasp-dz", type=float, default=None, help="required with --run")
    ap.add_argument("--min-z", type=float, default=None,
                    help="hard floor passed straight through to clear_scene. Strongly "
                         "recommended: nothing else stops the descent at the bench.")
    ap.add_argument("--dz-class", nargs="*", default=None, metavar="CLASS=MM")
    ap.add_argument("--dz-extra-raised", nargs="*", default=None, metavar="CLASS=MM",
                    help="extra descent for a class when it is resting on something "
                         "rather than on the bench, e.g. esp=4 lcd=2 ultrasonic=2")
    ap.add_argument("--max-z", type=float, default=None,
                    help="ceiling on the hover height; a tall stack otherwise produces "
                         "a pose the arm cannot reach and the controller faults")
    ap.add_argument("--place-xy", default=None, metavar="X,Y")
    ap.add_argument("--max-attempts", type=int, default=12,
                    help="safety cap. Must exceed the BOQ size: a failed pick is "
                         "retried after the next re-scan, so attempts > parts.")
    ap.add_argument("--descent-mode", default="fixed",
                    help="'fixed' is the only mode with live hours on it (12 Sep). "
                         "'clearance' is implemented and unit-tested but has never "
                         "moved the arm - do not debut it in a demo.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # SCAN FIRST when the operator is going to choose. The robot reports what it
    # can see, and the bill is written against that - so a request for a part the
    # system cannot detect is caught before the arm moves, not after.
    if not args.interactive:
        want, order = load_boq(args.boq)
        if not want:
            raise SystemExit(f"{args.boq} lists no requirements.")
        print("BOQ: " + ", ".join(f"{c}x{want[c]}" for c in order))

    if args.live or args.db3:
        from scan import scan
        picks, _ = scan(args.db3, conf=args.conf,
                        save_as=os.path.join(HERE, "boq_scan.png"))
        json.dump(picks, open(PICK_LIST, "w"), indent=2)
        print(f"  scanned {len(picks)} parts -> boq_scan.png")
    else:
        if not os.path.exists(PICK_LIST):
            raise SystemExit("No pick_list.json — run with --live or --db3, "
                             "or run scan.py first.")
        picks = json.load(open(PICK_LIST))
        print(f"  using the last scan ({len(picks)} parts from pick_list.json)")

    if args.interactive:
        avail = show_inventory(picks, args.conf)
        want, order = ask_request(avail)
        if not want:
            raise SystemExit("\n  cancelled - nothing was requested.")
        print("\n  requested: " + ", ".join(f"{c} x{want[c]}" for c in order))

    chosen, report, leave = plan(want, order, picks, args.conf)
    show(report, chosen, leave, want)

    json.dump({"boq": want, "report": report, "plan": chosen,
               "leave": [p["label"] for p in leave]},
              open(args.out, "w"), indent=2)
    print(f"\n  plan -> {args.out}")

    if not args.run:
        print("  (add --run --grasp-dz <mm> to execute it)")
        return

    if args.grasp_dz is None:
        raise SystemExit("--run needs --grasp-dz")
    if args.interactive:
        print("\n  The arm will now fetch the parts listed above.")
        if not input("  Hand on the e-stop. Execute? [y/N] ").strip().lower().startswith("y"):
            raise SystemExit("  cancelled.")
    if not chosen:
        raise SystemExit("nothing to execute")

    # Hand the SELECTED parts to the existing clearing loop rather than
    # reimplementing the pick cycle: one definition of how a part is picked.
    sel = os.path.join(HERE, "boq_pick_list.json")
    json.dump(chosen, open(sel, "w"), indent=2)
    # Hand the QUANTITIES to clear_scene, not just the class list. "--only arduino
    # esp lcd --max 3" on a bench holding two arduinos can return two arduinos and
    # an esp and stop, having fulfilled nothing. --want counts per class, against
    # parts confirmed off the bench.
    #
    # --rescan per-pick, not end: measured 12 Sep, re-planning after every pick is
    # worth 50 percentage points of clearance (4/4 vs 2/4). A BOQ fetch that gives
    # up at 50% is not a demonstration.
    #
    # --min-z is passed through because without it nothing stops the descent at the
    # bench, and there is one finger pair left.
    cmd = [sys.executable, os.path.join(HERE, "clear_scene.py"),
           "--grasp-dz", str(args.grasp_dz),
           "--descent-mode", args.descent_mode,
           "--order", "topmost",
           "--max", str(args.max_attempts),
           "--condition", "boq",
           "--rescan", "per-pick",
           "--clear-blockers",
           # THE PLANNER AND THE EXECUTOR MUST SEE THE SAME SCENE. Without this
           # the plan was built at --conf 0.55 and the run re-scanned at
           # clear_scene's own default, so a part could appear in the bill and be
           # invisible to the arm. Observed 15 Sep: an arduino under an lcd
           # planned at conf 0.569 and was simply absent from the run's scan.
           "--conf", str(args.conf),
           "--min-conf", str(min(args.conf, 0.5)),
           "--want"] + [f"{c}={n}" for c, n in want.items()]
    if args.min_z is not None:
        cmd += ["--min-z", str(args.min_z)]
    if args.dz_class:
        cmd += ["--dz-class"] + list(args.dz_class)
    if args.dz_extra_raised:
        cmd += ["--dz-extra-raised"] + list(args.dz_extra_raised)
    if args.max_z is not None:
        cmd += ["--max-z", str(args.max_z)]
    if args.place_xy:
        cmd += ["--place-xy", args.place_xy]
    if args.dry_run:
        cmd.append("--dry-run")
    env = dict(os.environ, PICKER_PICK_LIST=sel)
    print("\n  executing: " + " ".join(cmd[1:]) + "\n")
    # Flush before handing over. Python block-buffers stdout when it is a file
    # rather than a terminal, so without this the whole BOQ plan lands in the log
    # AFTER the execution trace it was supposed to explain - and redirecting to a
    # log is exactly what gets done in the lab.
    sys.stdout.flush()
    sys.exit(subprocess.call(cmd, env=env))


# ------------------------------------------------------------------ self-test
def self_test():
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    import tempfile
    p = os.path.join(tempfile.gettempdir(), "boq_test.txt")
    open(p, "w").write("# comment\narduino: 2\nesp32: 1\n\n1 x lcd\nnonsense line\n")
    want, order = load_boq(p)
    ck("parses 'class: n'", want.get("arduino") == 2, str(want))
    ck("normalises esp32 -> esp", want.get("esp") == 1, str(want))
    ck("parses '1 x lcd'", want.get("lcd") == 1, str(want))
    ck("keeps the operator's order", order == ["arduino", "esp", "lcd"], str(order))

    scene = [
        {"label": "arduino", "conf": .9, "x": 0, "y": 600, "z": 250.0},
        {"label": "arduino", "conf": .9, "x": 40, "y": 600, "z": 268.0},   # on top
        {"label": "esp",     "conf": .9, "x": 80, "y": 600, "z": 246.0},
        {"label": "ultrasonic", "conf": .9, "x": 120, "y": 600, "z": 247.0},
        {"label": "lcd",     "conf": .3, "x": 160, "y": 600, "z": 246.0},  # below conf
    ]
    chosen, report, leave = plan(want, order, scene, 0.55)
    ck("takes the requested count", len([c for c in chosen if c["label"] == "arduino"]) == 2)
    ck("takes the TOPMOST instance first",
       chosen[0]["z"] == 268.0, f"(first z={chosen[0]['z']})")
    ck("an undetected lcd is reported unavailable, not silently dropped",
       [r for r in report if r["class"] == "lcd"][0]["status"] == "unavailable")
    ck("the ultrasonic nobody asked for is left alone",
       [p_["label"] for p_ in leave] == ["ultrasonic"], str([p_["label"] for p_ in leave]))
    ck("arduino is servo-verified", [r for r in report
                                     if r["class"] == "arduino"][0]["verify"] == "servo")
    # Since 12 Sep every class has a measured stall signature, so all four are
    # servo-verified. The re-scan path stays in the code as the ablation arm and
    # for any class that has never been characterised.
    ck("esp is servo-verified too, since 12 Sep",
       [r for r in report if r["class"] == "esp"][0]["verify"] == "servo")
    ck("an uncharacterised class falls back to the re-scan",
       plan({"spoon": 1}, ["spoon"], scene, 0.55)[1][0]["verify"] == "re-scan")

    want2, order2 = {"arduino": 3}, ["arduino"]
    _, rep2, _ = plan(want2, order2, scene, 0.55)
    ck("asking for more than exist reports 'short'", rep2[0]["status"] == "short",
       str(rep2[0]))

    show(report, chosen, leave, want)
    print("\n" + "=" * 56)
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("boq_fetch self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    main()
