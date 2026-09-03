from pickerbot_lib.config import CONFIG, PROJECT_ROOT, resolve
from pickerbot_lib.sender import connect, disconnect, epsonPickAll, epsonPick, epsonMove, epsonGo, epsonJump, epsonStandby

# detection/calibration pull in torch + a model file. Load them lazily so that
# lightweight tasks (e.g. just sending arm commands) don't need the ML stack or
# a model present. Access still works: `from pickerbot_lib import detect_and_annotate`.
_LAZY = {
    "detect_and_annotate": "pickerbot_lib.detection",
    "load_calibration_data": "pickerbot_lib.calibration",
    "calculate_homography": "pickerbot_lib.calibration",
    "pixel_to_world": "pickerbot_lib.calibration",
    "run_calibration_gui": "pickerbot_lib.calibration",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module:
        import importlib
        return getattr(importlib.import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
