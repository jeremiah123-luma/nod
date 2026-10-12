"""Reads the macOS permissions Nod needs, so the dashboard can show what's missing.

Each function returns True / False, or None if it can't tell (not macOS, library missing).
These only READ the current status; they never pop up a permission prompt.
"""


def accessibility():
    """Needed to move the mouse, click, scroll and type."""
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        return None


def input_monitoring():
    """Needed for the global hotkeys (ctrl+alt+p and friends)."""
    try:
        import Quartz
        return bool(Quartz.CGPreflightListenEventAccess())
    except Exception:
        return None


def status():
    return {"accessibility": accessibility(), "input_monitoring": input_monitoring()}
