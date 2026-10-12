"""Preview the Command Center without a camera or a mouse:  .venv/bin/python -m nod.demo

Runs the real controller on invented, animated face data, and shows it on the real dashboard.
It NEVER touches your mouse, camera, or config.json (settings changes go to a throwaway file).
"""
import argparse
import math
import os
import random
import tempfile
import time
import webbrowser
from types import SimpleNamespace

from nod import mesh as meshmod
from nod import settings
from nod.bridge import Bridge
from nod.config import load_config
from nod.controller import Controller
from nod.dashboard import Dashboard
from nod.face import FaceData
from nod.gaze import calibration_targets, fit_best

SCREEN = (1512, 982)


class DemoActions:
    """Same interface as nod.actions, but moves nothing."""

    def __init__(self):
        self.paused = True
        self.pos = (SCREEN[0] // 2, SCREEN[1] // 2)

    screen_size = staticmethod(lambda: SCREEN)
    position = lambda self: self.pos
    is_paused = lambda self: self.paused
    set_paused = lambda self, v: setattr(self, "paused", bool(v))
    move_to = lambda self, x, y: (setattr(self, "pos", (x, y)) or not self.paused)
    click = lambda self: None
    scroll = lambda self, n: None
    type_text = lambda self, t: None


# ---------- a made-up face, good enough to look like one ----------
def _chains(pairs):
    """Splits connection pairs into ordered point chains (loops or paths)."""
    adj = {}
    for a, b in pairs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, chains = set(), []
    starts = sorted(adj, key=lambda n: (len(adj[n]) != 1, n))     # path ends first, then anything
    for s in starts:
        if s in seen:
            continue
        chain, cur, prev = [s], s, None
        seen.add(s)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            seen.add(cur)
        chains.append(chain)
    return sorted(chains, key=len, reverse=True)


_CHAINS = {name: _chains(pairs) for name, pairs in meshmod.GROUPS.items()}


def demo_points(t, yaw=0.0, pitch=0.0, jaw=0.0, brow=0.0, look=(0.0, 0.0)):
    """Flat [x,y,...] in 0..1000 for meshmod.POINT_IDS, a stylised face that moves with the numbers."""
    pts = [[500.0, 500.0] for _ in meshmod.POINT_IDS]
    cx, cy = 0.5 + yaw * 0.004, 0.50 + pitch * 0.004

    def ellipse(chain, ex, ey, rx, ry, closed=True):
        n = len(chain)
        for i, p in enumerate(chain):
            a = 2 * math.pi * i / n if closed else math.pi * i / max(1, n - 1)
            pts[p][0] = (ex + rx * math.cos(a)) * 1000
            pts[p][1] = (ey - ry * math.sin(a)) * 1000

    ellipse(_CHAINS["oval"][0], cx, cy, 0.17, 0.30)
    for i, chain in enumerate(_CHAINS["eyes"][:2]):
        ellipse(chain, cx + (-0.075 if i == 0 else 0.075), cy - 0.08, 0.04, 0.018)
    for i, chain in enumerate(_CHAINS["brows"][:2]):
        ellipse(chain, cx + (-0.08 if i == 0 else 0.08), cy - 0.145 - brow * 0.04, 0.05, 0.01)
    for i, chain in enumerate(_CHAINS["irises"][:2]):
        ellipse(chain, cx + (-0.075 if i == 0 else 0.075) + look[0] * 0.012, cy - 0.08 + look[1] * 0.008, 0.009, 0.009)
    lips = _CHAINS["lips"]
    ellipse(lips[0], cx, cy + 0.15, 0.065, 0.016 + jaw * 0.05)
    if len(lips) > 1:
        ellipse(lips[1], cx, cy + 0.15, 0.045, 0.004 + jaw * 0.045)
    nose = _CHAINS["nose"][0]
    for i, p in enumerate(nose):
        f = i / max(1, len(nose) - 1)
        pts[p] = [(cx + (-0.02 + 0.04 * f) + yaw * 0.002) * 1000, (cy - 0.04 + 0.12 * (1 - abs(f - 0.5) * 0.4) * 0.8 + 0.0) * 1000]
    return [round(v) for xy in pts for v in xy]


def demo_face(t, rng):
    """Head drifting around, a mouth-open every few seconds, brows now and then, eyes wandering."""
    hx = 5 * math.sin(t * 0.55) + 1.5 * math.sin(t * 1.7) + rng.gauss(0, 0.14)
    hy = 2.5 * math.sin(t * 0.37 + 1) + rng.gauss(0, 0.25)
    jaw = 0.75 if (t % 9.0) < 0.55 and t > 9 else 0.05 + 0.03 * rng.random()
    brow = 0.7 if (t % 17.0) > 15.2 and t > 17 else 0.1 + 0.03 * rng.random()
    spot = int(t // 4) % 3
    tx, ty = [(0.25, 0.3), (0.85, 0.7), (0.55, 0.5)][spot]
    gh = (tx - 0.5) * 0.30 + rng.gauss(0, 0.004)
    gv = (ty - 0.5) * 0.20 + rng.gauss(0, 0.004)
    face = FaceData(head_x=hx, head_y=hy, blend={"jawOpen": jaw, "browInnerUp": brow, "mouthPucker": 0.05},
                    gaze_h=gh, gaze_v=gv)
    return face, demo_points(t, hx * 1.5, hy * 1.5, jaw, brow, ((tx - 0.5) * 2, (ty - 0.5) * 2))


def demo_mapper():
    dots = []
    for x, y in calibration_targets(*SCREEN) * 2:
        dots.append((SimpleNamespace(gaze_h=(x / SCREEN[0] - 0.5) * 0.30, gaze_v=(y / SCREEN[1] - 0.5) * 0.20,
                                     head_x=0.0, head_y=0.0), (x, y)))
    return fit_best(dots, SCREEN, 0.1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--uncalibrated", action="store_true", help="show the 'eyes not calibrated' state")
    args = ap.parse_args()

    cfg = load_config()
    actions = DemoActions()
    controller = Controller(cfg, actions, None if args.uncalibrated else demo_mapper())
    bridge = Bridge(actions)
    box = {"cfg": cfg}
    scratch = os.path.join(tempfile.mkdtemp(prefix="nod-demo-"), "config.json")      # demo never writes your real config
    dash = Dashboard(lambda: box["cfg"], bridge.on_command, bridge.on_settings,
                     meta_extra=lambda: {"mesh": meshmod.meta(), "demo": True}, port=args.port, config_path=scratch)
    url = dash.start()
    print(f"Nod demo (no camera, no mouse): {url}")
    if not args.no_open:
        webbrowser.open(url)

    rng, t0, last_pub = random.Random(5), time.monotonic(), 0.0
    try:
        while True:
            now = time.monotonic() - t0
            face, points = demo_face(now, rng)
            for item in bridge.drain():
                if item == "recenter":
                    controller.recenter()
                elif item == "toggle_dwell":
                    controller.toggle_dwell()
                elif item == "toggle_gaze":
                    controller.toggle_gaze()
                elif isinstance(item, tuple) and item[0] == "settings":
                    box["cfg"] = item[1]
                    controller.reload(item[1])
            controller.update(face, now)
            if now - last_pub > 1 / 15:
                last_pub = now
                snap = controller.snapshot()
                snap.update(fps=30, mesh=points, perms={"accessibility": True, "input_monitoring": True})
                dash.publish(snap)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop()


if __name__ == "__main__":
    main()
