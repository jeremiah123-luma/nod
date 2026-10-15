# Nod

**🚧 In development.** The head pointer works on a real webcam and is being tuned for jitter. Everything past that is coded and unit tested but not yet proven on a real face. See the [roadmap](#roadmap).

Nod is a macOS app that turns an ordinary webcam into a hands-free mouse. It is for people with motor disabilities, and for anyone who can't or doesn't want to use their hands in the moment. No extra hardware: head movement steers the cursor and a simple face gesture clicks. Everything runs locally on the Mac. Camera frames are never saved or sent anywhere.

> Screenshot / GIF coming soon.
>
> ![Nod demo placeholder](docs/demo.gif)

## Why I built this

Most of us never think twice about using a mouse, but for a lot of people it isn't that simple. Some people can't use their hands at all, and some of us just can't, or don't want to, in the moment, like at the end of a long night when you're too tired to even reach for the trackpad. I wanted to build something for both of them: an app that turns the webcam you already have into a mouse, so you can move the cursor with your head and click with a simple face gesture, without buying any special hardware. That's what Nod is.

## How it works

1. **Face tracking.** [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) runs on each webcam frame and returns 478 face landmarks plus blendshapes (named face expressions like `jawOpen` and `browInnerUp`). `nod/face.py`
2. **Head pointer.** Nod measures where the nose tip sits relative to the line between the eyes. That ratio is designed to stay stable across distance from the camera and head roll. Head movement drives the cursor like a trackpad (relative movement times a gain), smoothed by a [One Euro filter](https://gery.casiez.net/1euro/) and a lock-and-release deadzone. `nod/pointer.py`, `nod/filters.py`
3. **Gestures (coded, being tested).** A mouth-open gesture held for a short time clicks once (with hysteresis and a cooldown so a yawn can't double-click). An eyebrow raise toggles scroll mode, where tilting the head scrolls instead of moving the cursor. Mouth pucker is an alternate toggle if eyebrows misfire. `nod/gestures.py`, `nod/controller.py`
4. **Actions and safety.** `nod/actions.py` is the only file that touches the mouse (via PyAutoGUI). The app starts paused, a global hotkey pauses it from any app, and paused means zero mouse actions. `nod/hotkeys.py`
5. **Feedback.** A preview window shows the face mesh, the current mode (POINT / SCROLL / PAUSED), and click progress. A local browser dashboard (127.0.0.1 only, token protected) shows raw vs smoothed head signals, gesture meters that explain why a gesture did or didn't fire, and sliders for every tunable number. `nod/hud.py`, `nod/dashboard.py`, `nod/ui/`

Every threshold, gain, and hold time lives in `config.json`, not in code, so tuning never means editing Python.

### Problems I ran into

- **Pointer jitter.** MediaPipe landmarks wobble by a pixel or two every frame even when your head is still, and a gain of 60 to 70 turns that into a cursor that never settles. The One Euro filter alone wasn't enough. The fix in `pointer.py` is a lock-and-release deadzone: once the head has stayed inside a small circle for 300 ms the cursor locks and doesn't move at all, and it only releases when the head travels a deliberate distance. Jitter stays inside the circle; a real move leaves it immediately. Still being tuned on a real face (see roadmap).
- **Raising your eyebrows moves your head.** The scroll toggle was firing when I tilted my head up, because tilting up also lifts the brows. The toggle now only counts when the head is level, still, and has been steady for a moment (`controller.py`).
- **Opening your mouth moves your head too.** A mouth-open click nudges the cursor off the target right as it clicks. That is why the next milestone switches the click gesture to tongue-out, with mouth-open kept as a backup setting.
- **Webcam eye tracking is rough.** A gaze-based cursor jump is in the code behind a toggle, with a dot calibration screen that reports its own error, but webcam gaze is only accurate to roughly which part of the screen. It is being rethought rather than promised.

## Tech stack

- **Python 3.11 / 3.12**
- **MediaPipe** (Face Landmarker, Tasks API) for landmarks and blendshapes
- **OpenCV** for webcam capture and the preview window
- **PyAutoGUI** to move, click, and scroll
- **pynput** for global hotkeys
- **faster-whisper** and **sounddevice** for the local voice-command prototype (off by default)
- Standard library only for the dashboard server; the dashboard page loads no CDNs or external fonts
- `unittest` for the test suite (244 unit tests that run on recorded/fake landmark data, so no camera or mouse is needed)

Built with Claude Code as a pair programmer.

## Run it

Requires macOS and Python 3.11 or 3.12 (MediaPipe lags behind the newest Python releases).

```bash
git clone https://github.com/jeremiah123-luma/nod.git
cd nod
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# the face model (about 4 MB), not checked in
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**macOS permissions.** Open System Settings, then Privacy & Security, and turn on the app you run Python from (Terminal, iTerm, VS Code) under each of these. Then fully quit and reopen that app.

| Permission | Needed for |
|---|---|
| Camera | reading the webcam |
| Accessibility | moving the mouse, clicking, scrolling |
| Input Monitoring | the global hotkeys |
| Microphone | only if you turn voice on |

If the cursor does nothing, it is almost always Accessibility. If the hotkeys do nothing, it is Input Monitoring.

```bash
.venv/bin/python -m nod.main --dry-run   # safe: prints what it WOULD do, never touches the mouse
.venv/bin/python -m nod.main             # the real thing
.venv/bin/python -m nod.demo             # the dashboard with no camera and no mouse
.venv/bin/python -m unittest discover -s tests -t .   # tests
```

Nod starts paused. Press `ctrl+alt+p` (or Start control in the dashboard) when you're ready.

| Hotkey (works from any app) | What |
|---|---|
| `ctrl+alt+p` | pause / resume (paused = zero mouse actions) |
| `ctrl+alt+c` | recenter: re-learn your relaxed face and level head |
| `ctrl+alt+d` | dwell click on / off |
| `ctrl+alt+q` | quit |

Emergency stop: slam the real mouse into the top-left corner of the screen (PyAutoGUI's failsafe).

## Roadmap

| Milestone | Status | Notes |
|---|---|---|
| M0: face tracking + mesh preview | ✅ Done | Webcam preview with landmarks and FPS |
| M1: head pointer | 🔧 In progress | Works on a real webcam. Fixing jitter: One Euro filter plus a lock-and-release deadzone, still tuning the numbers against a real face |
| M2: click gesture | 🔧 In progress | Mouth-open click works but moves the head. Switching to tongue-out so clicking doesn't move the cursor; mouth-open stays as a backup setting. Tongue-out is not coded yet |
| M3: scroll mode | 🔧 In progress | Coded and unit tested, not yet verified live. Tilt head up/down to scroll, toggled by an eyebrow raise (or mouth pucker if eyebrows misfire), with the mode shown on screen |
| Later: natural-language voice commands | 📋 Planned | e.g. "scroll up", "type Romans 12:2". A basic local voice prototype (faster-whisper, a few fixed commands) exists behind an off-by-default setting |
| Under consideration: eye tracking | 🤔 Rethinking | Look at a far part of the screen and the cursor jumps there, then the head fine-aims. Coded with a calibration screen, but webcam eye tracking is imprecise, so it may not stay |

Ideas that are written down but not built are in `IDEAS.md`.

## Privacy

No camera frames are ever saved or sent anywhere. The dashboard shows only numbers and a point wireframe of your face, and it listens on 127.0.0.1 only. Eye-tracking calibration data stays on your machine and is excluded from git.

## License

MIT. See [LICENSE](LICENSE).
