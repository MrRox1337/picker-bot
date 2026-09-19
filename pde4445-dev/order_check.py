#!/usr/bin/env python3
"""order_check.py — does this layout have the POWER to test the ordering claim?

Scan once, print the two sequences side by side, and say whether the raster
baseline actually makes a mistake on this layout. It moves nothing.

WHY
---
A paired ordering trial only measures something if the baseline's order is
WRONG here: if raster happens to reach every supporting part after the part
resting on it, both policies do the same thing and the pair is a null by
construction. That is not hypothetical - on 16 Sep layout L3 lost its power
exactly this way. Raster order happened to be a valid top-down order, both arms
scored 4/4, and the pair measured nothing. Worse, the dry run beforehand looked
fine: the rebuild BETWEEN the two arms changed which order raster produced.

So check the layout before each arm, not once at the start.

THE TEST
--------
`occlusion_pass` already reports, for every part, which detections sit above it
(`under`, `under_xy`). That is a support relation. Raster order is wrong here if
it reaches a SUPPORTING part before the part sitting on top of it - pulling the
support out from under its neighbour, which is the penalty topmost-first exists
to avoid.

    POWER    raster violates at least one support relation -> run the pair
    NO POWER raster happens to be a valid top-down order -> rebuild

Exit status: 0 = has power, 1 = no power, so it can gate a script.

USAGE
    python pde4445-dev/order_check.py
    python pde4445-dev/order_check.py --conf 0.55 --db3 some.db3
    python pde4445-dev/order_check.py --self-test
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAME_MM = 25.0


def orders(picks):
    """(topmost, naive) as lists of indices into picks, mirroring clear_scene."""
    top = sorted(range(len(picks)),
                 key=lambda i: -(picks[i].get("z_med") or picks[i]["z"]))
    nai = sorted(range(len(picks)),
                 key=lambda i: (picks[i].get("v", 0), picks[i].get("u", 0)))
    return top, nai


def support_pairs(picks):
    """[(supporter_index, resting_index)] from the measured `under_xy` lists.

    `under_xy` on part R lists the positions of detections sitting ABOVE R, so
    each entry is a part resting on R and R is the supporter.
    """
    pairs = []
    for ri, r in enumerate(picks):
        for bx, by in (r.get("under_xy") or []):
            for si, s in enumerate(picks):
                if si == ri:
                    continue
                if ((s["x"] - bx) ** 2 + (s["y"] - by) ** 2) ** .5 <= SAME_MM:
                    pairs.append((ri, si))       # ri supports si
    return sorted(set(pairs))


def violations(order, pairs):
    """Support relations this order gets WRONG: supporter taken before its load."""
    pos = {idx: k for k, idx in enumerate(order)}
    return [(sup, load) for sup, load in pairs
            if pos.get(sup, 1e9) < pos.get(load, 1e9)]


def verdict(picks):
    top, nai = orders(picks)
    pairs = support_pairs(picks)
    vt, vn = violations(top, pairs), violations(nai, pairs)
    lines = []
    add = lines.append
    name = lambda i: f"{picks[i]['label']}@({picks[i]['x']:.0f},{picks[i]['y']:.0f})"

    add(f"\n  {len(picks)} parts, {len(pairs)} measured support relation(s)")
    for sup, load in pairs:
        add(f"    {name(load)}  rests on  {name(sup)}")
    add("")
    add("  topmost : " + " -> ".join(picks[i]["label"] for i in top))
    add("  naive   : " + " -> ".join(picks[i]["label"] for i in nai))
    add("")
    if not pairs:
        add("  NO POWER — nothing rests on anything, so the two policies cannot")
        add("  differ in any way that matters. Stack something and re-check.")
        return False, lines
    if vt:
        add("  WARNING — topmost-first also violates a support relation here:")
        for sup, load in vt:
            add(f"    takes {name(sup)} before {name(load)} which rests on it")
        add("  That usually means a depth reading is wrong. Re-scan before trusting it.")
    if not vn:
        add("  NO POWER — raster order happens to be a valid top-down order, so")
        add("  it has no mistake available to make. Both arms will score the")
        add("  same and the pair will measure nothing. REBUILD the layout.")
        return False, lines
    add("  HAS POWER — raster takes a supporting part before its load:")
    for sup, load in vn:
        add(f"    {name(sup)} before {name(load)}, which is resting on it")
    add("")
    add("  Run the pair. Re-check after the rebuild between arms — the rebuild")
    add("  is what silently removed the power on 16 Sep.")
    return True, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--db3", default=None)
    ap.add_argument("--layout", default="")
    args = ap.parse_args()

    from scan import scan
    out = os.path.join(HERE, "scans")
    os.makedirs(out, exist_ok=True)
    import time
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_ordercheck" + (
        f"_{args.layout}" if args.layout else "")
    picks, _ = scan(args.db3, conf=args.conf,
                    save_as=os.path.join(out, tag + ".png"))
    ok, lines = verdict(picks)
    print(f"\nORDER CHECK" + (f"  layout {args.layout}" if args.layout else "")
          + f"  -> scans/{tag}.png")
    print("\n".join(lines))
    return 0 if ok else 1


# ----------------------------------------------------------------- self-test
def self_test():
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    def p(lbl, x, y, z, u, v, under=None):
        return {"label": lbl, "x": x, "y": y, "z": z, "z_med": z,
                "u": u, "v": v, "under_xy": under or []}

    # An esp resting on an lcd. Raster (sorted by v then u) reaches the lcd
    # FIRST because it is higher up the image -> it pulls the support out.
    lcd = p("lcd", 0, 700, 250, 300, 100)
    esp = p("esp", 5, 702, 262, 305, 300, under=[])
    lcd["under_xy"] = [[esp["x"], esp["y"]]]          # the esp sits on the lcd
    ok, lines = verdict([lcd, esp])
    ck("a support relation is recovered from under_xy",
       len(support_pairs([lcd, esp])) == 1)
    ck("raster taking the support first is POWER", ok is True)
    ck("topmost takes the esp first",
       orders([lcd, esp])[0][0] == 1)
    ck("the violation is named",
       any("resting on it" in l for l in lines), str(lines[-4:]))

    # Same stack, but the esp is higher up the image, so raster happens to take
    # it first and is accidentally correct. No power.
    lcd2 = p("lcd", 0, 700, 250, 300, 300)
    esp2 = p("esp", 5, 702, 262, 305, 100)
    lcd2["under_xy"] = [[esp2["x"], esp2["y"]]]
    ok2, lines2 = verdict([lcd2, esp2])
    ck("raster accidentally correct is NO POWER", ok2 is False)
    ck("and it says to rebuild", any("REBUILD" in l for l in lines2))

    # A flat scene has no support relations at all.
    flat = [p("lcd", 0, 700, 250, 100, 100), p("esp", 200, 700, 251, 400, 400)]
    ok3, lines3 = verdict(flat)
    ck("a flat scene has no power", ok3 is False)
    ck("and says why", any("nothing rests on anything" in l for l in lines3))

    # Topmost must never violate a relation it can see.
    ck("topmost violates nothing on the powered layout",
       violations(orders([lcd, esp])[0], support_pairs([lcd, esp])) == [])

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("order_check self-test passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(main())
