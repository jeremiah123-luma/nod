"""Global hotkeys: work even when another app is in front.

Needs macOS permission: Input Monitoring (and Accessibility) for the app running Python.
The callbacks run on a background thread, so they only flip flags / drop messages in a queue.
"""
import queue

from nod import actions

events = queue.Queue()  # main loop reads things like "recenter", "quit"


def build_bindings(cfg):
    keys = cfg["hotkeys"]
    bindings = {
        keys["pause"]: actions.toggle_pause,           # takes effect instantly, on this thread
        keys["recenter"]: lambda: events.put("recenter"),
        keys["quit"]: lambda: events.put("quit"),
    }
    if "dwell_toggle" in keys:
        bindings[keys["dwell_toggle"]] = lambda: events.put("dwell_toggle")
    if "voice_toggle" in keys:
        bindings[keys["voice_toggle"]] = lambda: events.put("voice_toggle")
    if "gaze_toggle" in keys:
        bindings[keys["gaze_toggle"]] = lambda: events.put("gaze_toggle")
    return bindings


def start(cfg):
    """Starts listening. Returns the listener (call .stop() on exit), or None if it can't start."""
    try:
        from pynput import keyboard
        listener = keyboard.GlobalHotKeys(build_bindings(cfg))
        listener.start()
        return listener
    except Exception as e:
        print(f"WARNING: global hotkeys unavailable ({e}). Check Input Monitoring permission.")
        return None
