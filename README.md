# Nod

Hands-free Mac control with your webcam (and later your voice). Everything runs on your
computer. Camera frames are never saved or sent anywhere.

- **Head** moves the cursor (like a trackpad)
- **Eyes** make big jumps: look at a far part of the screen and the cursor jumps there
- **Mouth open** = click
- **Eyebrows up** = switch between POINT mode and SCROLL mode
- **Voice** (optional) = "scroll down", "click", "type romans twelve two"

## Setup (once)
```
cd ~/nod
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
You also need the face model at `models/face_landmarker.task` (already downloaded).

## Run
```
cd ~/nod
.venv/bin/python -m nod.main --dry-run     # safe: prints what it WOULD do, never touches the mouse
.venv/bin/python -m nod.main               # the real thing
```
Nod **starts paused**. Press **ctrl+alt+p** when you're ready.

## Controls
| Keys (work from any app) | What |
|---|---|
| ctrl+alt+p | pause / resume (paused = zero mouse actions) |
| ctrl+alt+c | recenter cursor to the middle of the screen |
| ctrl+alt+d | dwell click on/off (hold still ~0.6 s = click) |
| ctrl+alt+g | eye jump on/off |
| ctrl+alt+v | voice: start / stop talking (if enabled) |
| ctrl+alt+q | quit |

In the preview window: `q` quit, `c` recenter, `d` dwell, `g` eye jump, `k` calibrate eyes.
Emergency stop: slam the **real mouse** into the **top-left corner** of the screen.

## Tests (no camera, no mouse needed)
```
cd ~/nod
.venv/bin/python -m unittest discover -s tests -t .
```

## macOS permissions
System Settings → Privacy & Security, turn on the app you run Python from (Terminal, iTerm, VS Code...), then fully quit and reopen it:
- **Camera** (needed)
- **Accessibility** (to move the mouse / click / type)
- **Input Monitoring** (for the global hotkeys)
- **Microphone** (only for voice)

## Tuning
Every number you might want to change is in `config.json`. See MORNING_REPORT.md for the list of the ones you'll probably touch first.
