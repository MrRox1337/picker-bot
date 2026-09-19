"""
RUN LOGGER — one CSV row per pick attempt, plus a per-session snapshot of the
configuration that produced it.

WHY THIS EXISTS
  * Un-logged runs are lost data. GRASP_DZ_MM was measured successfully on 2 Sep,
    never written down, and had to be re-measured. That must not happen again.
  * The supervisor's evaluation asks for accuracy, time and false positives. None
    of those can be reconstructed from memory after a lab session.
  * Every row also records the CONFIG that produced it, so a result stays
    interpretable after thresholds or calibration change.

USE INSIDE A PICK SCRIPT (3 lines of integration)

    from runlog import RunLogger

    log = RunLogger(condition="regular",
                    notes="5 spaced parts, PLA fingers + eraser padding",
                    config={"grip_current": GRIP_CURRENT,
                            "hold_threshold": HOLD_THRESHOLD,
                            "grasp_dz_mm": GRASP_DZ_MM,
                            "yaw_offset_deg": -88.8})

    for i, p in enumerate(picks):
        pick = log.start_pick(i, p)                  # stamps the clock
        ...
        pick.record(commanded_u_deg=u, grip_pos_after_close=pos, holding=holding)
        ...
        pick.finish("placed")                        # writes the row immediately

    log.close()

OUTCOMES (keep to this vocabulary so the analysis can count them)
    placed          picked, lifted, carried, PROBED at both checkpoints, released
    no_grasp        closed but the hold check said nothing was between the fingers
    lost_on_lift    the post-lift probe found the fingers free
    lost_in_transit the lift probe passed but the probe at the place pose did not
    release_failed  reached the place pose but the fingers never reopened
    arm_fault       the controller refused or failed to reply
    skipped         deliberately not attempted (e.g. thin class, low confidence)
    abandoned       hit the per-part attempt cap and was left alone
    aborted         operator stopped it

SKIPPED vs ABANDONED vs a failure. All three leave the part on the bench, and
they must not share a denominator. "skipped" is the system correctly refusing an
unsafe or ineligible pick; "abandoned" is the system giving up after the cap;
only the named failure outcomes are the pick going wrong. Report attempt success
over attempts (skips and abandonments excluded) AND part fulfilment over parts
present (both counted against), because either number alone can be made to say
whatever is wanted.

CAUTION ON "placed". Every class has a measured stall signature, but they are not
equally separable: a class whose separation from the empty-close position is small
relative to its own spread (the ultrasonic on this rig) can return holding=True on
a marginal grip. A "placed" row for such a class means "nothing contradicted
success", not "success observed", and only the verification re-scan settles it.
Read grip_profiles.json for the current separation-to-spread ratio per class.

Rows are flushed the moment they are written, so a crash or an abort never costs
data already captured.

SUMMARY OF EVERYTHING RECORDED SO FAR
    python pde4445-dev/runlog.py
"""
import os, csv, json, time, glob
from datetime import datetime

HERE     = os.path.dirname(os.path.abspath(__file__))
# PICKER_RUNS_DIR keeps regression-test output out of the real run log. Rehearsal
# rows in runs/ are worse than useless: they dilute every rate computed from it.
RUNS_DIR = os.environ.get("PICKER_RUNS_DIR", os.path.join(HERE, "runs"))

FIELDS = [
    "run_id", "condition", "pick_index",
    "label", "conf", "aspect",
    "x_mm", "y_mm", "z_mm", "vision_yaw_deg", "commanded_u_deg",
    # descent policy: which rule chose the grasp height, what it saw, what it did
    "descent_mode", "z_med", "z_land", "width_mm", "effective_dz_mm",
    "aperture_ticks",
    # Scene difficulty AT THE MOMENT OF THE ATTEMPT. Recorded on every row,
    # including skips, so success rate can be plotted against how buried the
    # part was rather than reported as one pooled number.
    "occlusion", "crowding",
    # which evidence was allowed to settle this grasp: 'proprioceptive' (the servo)
    # or 'vision' (the re-scan). Thin classes have no thickness signature, so a
    # 'vision' row's holding=False is NOT a failure - see clear_scene.verification_mode.
    "verify_mode",
    "grip_current_raw", "grip_pos_after_close", "holding",
    "grip_pos_after_lift", "pos_drop_ticks",
    # ACTIVE probe results - the only trustworthy grasp evidence under the
    # bounded freeze. Recorded at both checkpoints because they fail differently.
    "probe_travel_lift", "probe_holding_lift",
    "probe_travel_place", "probe_holding_place",
    "outcome", "cycle_time_s", "notes",
]

OUTCOMES = ("placed", "blocker_cleared", "no_grasp", "lost_on_lift", "lost_in_transit",
            "gripper_fault",
            "release_failed", "arm_fault", "skipped", "abandoned", "aborted")


class _Pick:
    """One pick attempt. Created by RunLogger.start_pick(); writes on finish()."""

    def __init__(self, logger, index, part):
        self.logger = logger
        self.t0 = time.time()
        self.row = {f: "" for f in FIELDS}
        part = part or {}
        self.row.update(
            run_id=logger.run_id,
            condition=logger.condition,
            pick_index=index,
            label=part.get("label", ""),
            conf=part.get("conf", ""),
            aspect=part.get("aspect", ""),
            x_mm=part.get("x", ""),
            y_mm=part.get("y", ""),
            z_mm=part.get("z", ""),
            vision_yaw_deg=part.get("yaw", ""),
        )

    def record(self, **kw):
        """Fill in fields as they become known. Unknown keys go to notes."""
        for k, v in kw.items():
            if k in self.row:
                self.row[k] = v
            else:
                self.row["notes"] = f"{self.row['notes']} {k}={v}".strip()
        return self

    def finish(self, outcome, **kw):
        self.record(**kw)
        if outcome not in OUTCOMES:
            print(f"[runlog] WARNING: '{outcome}' is not a known outcome {OUTCOMES}")
        self.row["outcome"] = outcome
        self.row["cycle_time_s"] = round(time.time() - self.t0, 2)
        # derive the lift drop if both positions are present
        a, b = self.row["grip_pos_after_close"], self.row["grip_pos_after_lift"]
        if a != "" and b != "" and self.row["pos_drop_ticks"] == "":
            try:
                self.row["pos_drop_ticks"] = int(a) - int(b)
            except (TypeError, ValueError):
                pass
        self.logger._write(self.row)
        return self.row


class RunLogger:
    def __init__(self, condition="unspecified", notes="", config=None, runs_dir=RUNS_DIR):
        os.makedirs(runs_dir, exist_ok=True)
        self.runs_dir = runs_dir
        self.condition = condition
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{condition}"
        self.csv_path = os.path.join(runs_dir, f"run_{self.run_id}.csv")

        self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._w.writeheader()
        self._f.flush()
        self.n = 0

        # session record: the config snapshot that makes these rows interpretable
        with open(os.path.join(runs_dir, "sessions.jsonl"), "a", encoding="utf-8") as s:
            s.write(json.dumps({
                "run_id": self.run_id,
                "started": datetime.now().isoformat(timespec="seconds"),
                "condition": condition,
                "notes": notes,
                "config": config or {},
            }) + "\n")

        print(f"[runlog] logging to {self.csv_path}")

    def start_pick(self, index, part):
        return _Pick(self, index, part)

    def event(self, outcome, note="", **kw):
        """Log something that is not a pick attempt (e.g. an aborted run)."""
        p = _Pick(self, "", {})
        p.record(notes=note, **kw)
        return p.finish(outcome)

    def _write(self, row):
        self._w.writerow(row)
        self._f.flush()                      # never lose a row to a crash
        self.n += 1
        print(f"[runlog] #{self.n} {row['outcome']:<12} {row['label']:<10} "
              f"{row['cycle_time_s']}s")

    def finding(self, **kw):
        """Attach a run-level result to the session record.

        The per-attempt CSV cannot hold the verification scan: it is one answer
        for the whole run, and it is the answer that matters most. On 12 Sep a
        naive-ordering run logged 6 placed while the scan found 4 parts gone -
        analysis that reads only the outcome column therefore OVERSTATES exactly
        the arm the experiment is trying to criticise.
        """
        path = os.path.join(self.runs_dir, "sessions.jsonl")
        lines = []
        if os.path.exists(path):
            lines = open(path, encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("run_id") == self.run_id:
                d.setdefault("result", {}).update(kw)
                lines[i] = json.dumps(d)
                break
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    def close(self):
        self._f.close()
        self.summarise_file(self.csv_path)


    # ------------------------------ reporting ------------------------------
    @staticmethod
    def summarise_file(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"[runlog] {os.path.basename(path)}: no rows")
            return
        counts, times = {}, []
        for r in rows:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
            try:
                times.append(float(r["cycle_time_s"]))
            except (TypeError, ValueError):
                pass
        placed = counts.get("placed", 0)
        print(f"\n[runlog] {os.path.basename(path)}  —  {len(rows)} attempts")
        print("         " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"         success rate {placed}/{len(rows)} = {100*placed/len(rows):.0f}%")
        if times:
            print(f"         cycle time mean {sum(times)/len(times):.1f}s  "
                  f"min {min(times):.1f}s  max {max(times):.1f}s")


def main():
    files = sorted(glob.glob(os.path.join(RUNS_DIR, "run_*.csv")))
    if not files:
        print(f"No runs yet in {RUNS_DIR}")
        return
    for p in files:
        RunLogger.summarise_file(p)

    # pooled totals across every run
    allrows = []
    for p in files:
        with open(p, newline="", encoding="utf-8") as f:
            allrows += list(csv.DictReader(f))
    counts = {}
    for r in allrows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    placed = counts.get("placed", 0)
    print(f"\n==== ALL RUNS: {len(allrows)} attempts across {len(files)} sessions ====")
    print("     " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if allrows:
        print(f"     overall success {placed}/{len(allrows)} = {100*placed/len(allrows):.0f}%")


if __name__ == "__main__":
    main()
