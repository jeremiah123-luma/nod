"""Which settings the dashboard may change, their allowed ranges, and how to save them.

Only keys listed in SCHEMA can be changed from the dashboard. Everything is range-checked, so a
typo or a hostile web page can't write nonsense into config.json.
"""
import copy
import json
import os
import tempfile

from nod.config import CONFIG_PATH

# kind: "float" | "int" | "bool" | "choice".  `live` = takes effect immediately (no restart).
SCHEMA = [
    # ---- Pointer ----
    dict(key="pointer.gain_x", group="Pointer", label="Cursor speed, left/right", kind="float", min=20, max=250, step=5,
         unit="px per head unit", help="Bigger = the cursor travels farther for the same head turn."),
    dict(key="pointer.gain_y", group="Pointer", label="Cursor speed, up/down", kind="float", min=20, max=250, step=5,
         unit="px per head unit", help="Bigger = the cursor travels farther for the same head tilt."),
    dict(key="pointer.deadzone_px", group="Pointer", label="Steadiness lock", kind="float", min=0, max=40, step=1,
         unit="px", help="Wobbles smaller than this don't move the cursor at all. Raise it if the cursor shakes when you hold still."),
    dict(key="pointer.filter_min_cutoff", group="Pointer", label="Smoothing when still", kind="float", min=0.05, max=2.0, step=0.05,
         unit="", help="Lower = smoother when you hold still (a little slower to start). Higher = more responsive."),
    dict(key="pointer.filter_beta", group="Pointer", label="Smoothing when moving", kind="float", min=0.0, max=1.5, step=0.05,
         unit="", help="Higher = the cursor keeps up better when you move fast."),
    dict(key="pointer.acceleration", group="Pointer", label="Fast-move boost", kind="float", min=0.0, max=1.0, step=0.05,
         unit="", help="Extra reach when you move your head quickly."),
    dict(key="pointer.invert_x", group="Pointer", label="Flip left/right", kind="bool", help="Use if the cursor goes the wrong way sideways."),
    dict(key="pointer.invert_y", group="Pointer", label="Flip up/down", kind="bool", help="Use if the cursor goes the wrong way vertically."),
    # ---- Click ----
    dict(key="click.threshold", group="Click", label="Mouth-open amount", kind="float", min=0.1, max=0.9, step=0.05,
         unit="0-1", help="How wide to open your mouth to click. Watch the Jaw meter and set this between 'relaxed' and 'open'."),
    dict(key="click.hold_ms", group="Click", label="Hold time", kind="int", min=100, max=800, step=25,
         unit="ms", help="How long to hold your mouth open. Longer = fewer accidental clicks when talking."),
    dict(key="click.cooldown_ms", group="Click", label="Pause between clicks", kind="int", min=200, max=1500, step=50, unit="ms",
         help="After a click, wait this long before the next one."),
    dict(key="dwell.enabled", group="Click", label="Dwell click (hold still to click)", kind="bool",
         help="Clicks when the cursor rests in one spot."),
    dict(key="dwell.dwell_ms", group="Click", label="Dwell time", kind="int", min=300, max=1500, step=50, unit="ms",
         help="How long to rest before a dwell click."),
    # ---- Scroll ----
    dict(key="scroll.toggle_gesture", group="Scroll", label="Scroll-mode gesture", kind="choice", options=["brows", "pucker"],
         help="Raise your eyebrows, or pucker your lips, to switch between pointing and scrolling."),
    dict(key="scroll.gestures.brows.delta", group="Scroll", label="Eyebrow raise needed", kind="float", min=0.1, max=0.8, step=0.05,
         unit="above relaxed", help="How far past your relaxed brows counts as a raise."),
    dict(key="scroll.gestures.pucker.delta", group="Scroll", label="Lip pucker needed", kind="float", min=0.1, max=0.8, step=0.05,
         unit="above relaxed", help="How far past your relaxed lips counts as a pucker."),
    dict(key="scroll.gate_pitch_window", group="Scroll", label="Level-head tolerance", kind="float", min=2, max=15, step=1,
         unit="head units", help="The toggle only works when your head is this close to level. Stops tilting-up from toggling."),
    dict(key="scroll.gain", group="Scroll", label="Scroll speed", kind="float", min=0.2, max=4.0, step=0.1,
         unit="lines/s per unit", help="How fast the page scrolls as you tilt your head."),
    dict(key="scroll.deadzone", group="Scroll", label="Scroll dead zone", kind="float", min=1, max=10, step=0.5,
         unit="head units", help="How far you must tilt before scrolling starts."),
    dict(key="scroll.max_clicks_per_s", group="Scroll", label="Top scroll speed", kind="int", min=5, max=80, step=5, unit="lines/s",
         help="Speed limit."),
    dict(key="scroll.invert", group="Scroll", label="Reverse scroll direction", kind="bool", help="Head down scrolls up instead."),
    # ---- Eye jump ----
    dict(key="gaze.enabled", group="Eye jump", label="Eye jump on", kind="bool", help="Look at a far spot and the cursor jumps there."),
    dict(key="gaze.min_jump_fraction", group="Eye jump", label="Minimum jump distance", kind="float", min=0.1, max=0.5, step=0.01,
         unit="of screen", help="Looks closer than this to the cursor are ignored (the head handles those)."),
    dict(key="gaze.dwell_ms", group="Eye jump", label="Look time", kind="int", min=200, max=1200, step=50, unit="ms",
         help="How long to keep looking before the jump."),
    dict(key="gaze.stable_fraction", group="Eye jump", label="Steadiness needed", kind="float", min=0.4, max=0.95, step=0.05,
         unit="0-1", help="How steady your gaze must be during the look. Lower = jumps more easily."),
    dict(key="gaze.cooldown_ms", group="Eye jump", label="Pause after a jump", kind="int", min=300, max=3000, step=100, unit="ms",
         help="Wait this long before the next jump."),
]
BY_KEY = {s["key"]: s for s in SCHEMA}


def get_path(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def set_path(cfg, dotted, value):
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def current_values(cfg):
    return {s["key"]: get_path(cfg, s["key"]) for s in SCHEMA}


def validate(key, value):
    """Returns the cleaned value, or raises ValueError with a readable message."""
    spec = BY_KEY.get(key)
    if spec is None:
        raise ValueError(f"unknown setting: {key}")
    kind = spec["kind"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    if kind == "choice":
        if value not in spec["options"]:
            raise ValueError(f"{key} must be one of {spec['options']}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{key} must be a finite number")
    if not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{key} must be between {spec['min']} and {spec['max']}")
    return int(round(value)) if kind == "int" else float(value)


def apply_changes(cfg, changes):
    """Returns a NEW config with the changes applied. All-or-nothing: one bad value rejects everything."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError("no changes given")
    cleaned = {k: validate(k, v) for k, v in changes.items()}
    new = copy.deepcopy(cfg)
    for k, v in cleaned.items():
        set_path(new, k, v)
    return new


def save_config(cfg, path=None):
    """Writes config.json atomically (a crash mid-write can't leave a half-written file)."""
    path = str(path or CONFIG_PATH)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def schema_for_ui(cfg):
    """The schema plus current values, shaped for the dashboard."""
    out = []
    for s in SCHEMA:
        item = dict(s)
        item["value"] = get_path(cfg, s["key"])
        out.append(item)
    return out
