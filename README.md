# Nod

Hands-free Mac control with your webcam. Everything runs locally.

## Setup (once)
```
cd ~/nod
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
You also need `models/face_landmarker.task` (see CLAUDE.md / M0 notes).

## Run
```
cd ~/nod
.venv/bin/python -m nod.main
```
Press `q` in the preview window to quit.

## macOS permissions
System Settings → Privacy & Security → **Camera**: turn on your terminal app.
(Accessibility and Input Monitoring are needed later, from M1.)
