# D-Day workspace

Everything in this folder exists to make the demonstration go well. Nothing in
it may change a number in the report.

**v1 of the thesis is frozen and committed.** The report quotes measurements
taken under one configuration: `GRIP_CURRENT 100`, `HOLD_THRESHOLD 830`,
`EMPTY_STALL_TICKS 708`, per-class stall profiles 901/1143/1237/1859, aperture
25.983 ticks/mm, calibration `max_open 3541`, `--min-z 215`, `--grasp-dz 35`.
If any of those changes, the demonstrated system is no longer the system the
report describes, and that is a question an examiner can ask.

## The rule

> Work here is ADDITIVE and READ-ONLY with respect to the pick pipeline.
> Nothing in `pde4445-dev/*.py` changes behaviour. If a change cannot be made
> without altering a constant the report quotes, it does not get made.

Two permitted exceptions, both harmless to the numbers:

1. **Pure additions** — a new module here that observes, records or displays,
   and never feeds a decision back into the pick.
2. **Flags that default to today's behaviour** — a new option is fine if
   omitting it reproduces v1 exactly.

## Provenance

Weighting: report 40%, demonstration 20%. The report is submitted before the
next lab slot, so there is no route by which work here can improve it --- only
routes by which it could invalidate it.

## Layout

    dday/
      README.md      this file
      NOTES.md       running notes, decisions and their reasons

Created 22 Sep 2026, after the v1 commit.
