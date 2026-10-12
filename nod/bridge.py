"""Carries commands from the dashboard (web server threads) into the main loop safely.

Pausing is instant (it only flips a flag, safe from any thread). Everything else is queued and
done by the main loop, so the controller is only ever touched from one thread.
"""
import queue


class Bridge:
    def __init__(self, actions):
        self.actions = actions
        self._q = queue.Queue()

    # ---- called from server threads ----
    def on_command(self, action):
        if action == "pause":
            self.actions.set_paused(True)
        elif action == "resume":
            self.actions.set_paused(False)
        elif action == "toggle_pause":
            self.actions.set_paused(not self.actions.is_paused())
        else:
            self._q.put(action)
            return {"queued": True}
        return {"paused": self.actions.is_paused()}

    def on_settings(self, cfg):
        self._q.put(("settings", cfg))

    # ---- called from the main loop, once per frame ----
    def drain(self):
        items = []
        try:
            while True:
                items.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return items
