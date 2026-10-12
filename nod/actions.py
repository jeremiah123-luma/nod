"""The ONLY file that touches the real mouse. Everything here respects pause.

Paused means zero mouse actions: every function below checks the pause flag first.
"""
import threading

_paused = False
_dry_run = False          # True = print what we WOULD do, never touch the mouse
_lock = threading.Lock()  # the pause hotkey flips the flag from another thread
_pg_module = None


class FailSafeTriggered(Exception):
    """Raised when the physical mouse is slammed into the top-left corner (pyautogui's panic stop)."""


def _pg():
    """Import pyautogui lazily so unit tests never need a screen."""
    global _pg_module
    if _pg_module is None:
        import pyautogui
        pyautogui.PAUSE = 0           # no built-in delay after each call (speed)
        pyautogui.FAILSAFE = True     # mouse in the top-left corner = emergency stop
        pyautogui.MINIMUM_DURATION = 0
        _pg_module = pyautogui
    return _pg_module


def _do(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # pyautogui.FailSafeException
        if type(e).__name__ == "FailSafeException":
            set_paused(True)
            raise FailSafeTriggered() from e
        raise


# ---- state ----
def set_paused(value):
    global _paused
    with _lock:
        _paused = bool(value)


def toggle_pause():
    global _paused
    with _lock:
        _paused = not _paused
        return _paused


def is_paused():
    return _paused


def set_dry_run(value):
    global _dry_run
    _dry_run = bool(value)


# ---- read-only helpers (safe while paused) ----
def screen_size():
    return tuple(_pg().size())


def position():
    return tuple(_pg().position())


# ---- things that change the screen: all blocked while paused ----
def move_to(x, y):
    if _paused:
        return False
    if _dry_run:
        print(f"[dry-run] move_to({x:.0f}, {y:.0f})")
        return True
    _do(_pg().moveTo, x, y, _pause=False)
    return True


def click():
    if _paused:
        return False
    if _dry_run:
        print("[dry-run] click")
        return True
    _do(_pg().click, _pause=False)
    return True


def scroll(clicks):
    """clicks > 0 scrolls up, < 0 scrolls down."""
    if _paused or clicks == 0:
        return False
    if _dry_run:
        print(f"[dry-run] scroll({clicks})")
        return True
    _do(_pg().scroll, int(clicks), _pause=False)
    return True
