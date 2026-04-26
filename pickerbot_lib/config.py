import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_ROOT, "config.json"), "r") as _f:
    CONFIG = json.load(_f)


def resolve(path):
    """Resolve a config-relative path against the project root."""
    return os.path.join(PROJECT_ROOT, path)
