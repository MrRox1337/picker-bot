"""
Locate Aman's GripSense repo and make its Lib importable.

Single-laptop setup: picker-bot and GripSense sit SIDE BY SIDE, so a script in
pde4445-dev is not inside the GripSense tree and cannot find Config/ by walking
up its parents. Search known locations instead.

Override anytime with:
    set GRIPSENSE_REPO=C:\\Users\\10463\\MB_faseeh\\REPO\\GripSense
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # ...\picker-bot\pde4445-dev
_REPOS = os.path.dirname(os.path.dirname(_HERE))            # ...\MB_faseeh\REPO

CANDIDATES = [
    os.path.join(_REPOS, "GripSense"),                      # sibling of picker-bot
    os.path.join(_REPOS, "GripSense-main"),
    os.path.join(_REPOS, "GripSense-main", "GripSense-main"),
    r"C:\Users\10463\MB_faseeh\REPO\GripSense",
]


def find_gripsense():
    cands = []
    env = os.environ.get("GRIPSENSE_REPO")
    if env:
        cands.append(env)
    cands += CANDIDATES
    for c in cands:
        if (c and os.path.exists(os.path.join(c, "Config", "gripper_config.yaml"))
                and os.path.isdir(os.path.join(c, "Lib"))):
            return c
    raise SystemExit(
        "Could not find Aman's GripSense repo.\n"
        "Looked in:\n  " + "\n  ".join(str(c) for c in cands) + "\n"
        "Set it explicitly:\n"
        r"    set GRIPSENSE_REPO=C:\Users\10463\MB_faseeh\REPO\GripSense"
    )


def load_settings():
    """Return (gripper_settings module, repo path), with Lib/ on sys.path."""
    repo = find_gripsense()
    lib = os.path.join(repo, "Lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import gripper_settings as settings
    return settings, repo
