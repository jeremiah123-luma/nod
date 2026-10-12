"""Nod: hands-free computer control. Run with:  .venv/bin/python -m nod.main

Add --dry-run to print what the mouse WOULD do without touching it.
Global hotkeys (work from any app): ctrl+alt+p pause/resume, ctrl+alt+c recenter, ctrl+alt+d dwell click on/off, ctrl+alt+g eye jump on/off, ctrl+alt+v voice (if enabled), k = calibrate eye jump, ctrl+alt+q quit.
"""
import argparse
import queue
import time

import cv2

from nod import actions, hotkeys
from nod.camera import Camera
from nod.config import load_config
from nod.controller import Controller
from nod.face import FaceTracker
from nod.gaze import GazeMapper
from nod.voice import apply_command, parse_command
from nod.gaze_calibration import CAL_PATH, run_calibration
from nod.hud import draw_hud, draw_mesh


def _gaze_label(state):
    return "not calibrated (press k first)" if state is None else ("ON" if state else "OFF")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print mouse actions instead of doing them")
    args = parser.parse_args()

    cfg = load_config()
    actions.set_dry_run(args.dry_run)
    camera = Camera(cfg["camera"]["index"], cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"])
    aspect = cfg["camera"]["width"] / cfg["camera"]["height"]
    tracker = FaceTracker(cfg)
    mapper = GazeMapper.load(CAL_PATH, actions.screen_size())
    controller = Controller(cfg, actions, mapper)
    print("Eye jump:", f"calibrated (error ~{mapper.error_px:.0f}px)" if mapper else "not calibrated yet. Press k in the preview window.")
    voice = None
    if cfg["voice"]["enabled"]:
        from nod.voice_input import VoiceInput
        voice = VoiceInput(cfg["voice"], lambda text: hotkeys.events.put(("voice_text", text)))
        print("Voice is ON: ctrl+alt+v to start/stop talking. Say: scroll up/down, click, pause, resume, type ...")
    listener = hotkeys.start(cfg)
    window = cfg["hud"]["window_name"]

    print("Nod is running. ctrl+alt+p = pause/resume, ctrl+alt+c = recenter, ctrl+alt+q = quit.")
    print("Emergency stop: slam the real mouse into the top-left corner of the screen.")
    actions.set_paused(True)  # start paused so nothing moves until you press ctrl+alt+p
    print("Started PAUSED. Press ctrl+alt+p when you're ready.")

    fps, last = 0.0, time.monotonic()
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            now = time.monotonic()
            result = tracker.process(frame)
            face = tracker.analyze(result, aspect)

            try:
                controller.update(face, now)
            except actions.FailSafeTriggered:
                print("Failsafe triggered (mouse in the corner). Stopping.")
                break

            if face is not None:
                draw_mesh(frame, face.landmarks)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            info = controller.hud_info()
            if voice:
                info["lines"].append(voice.status)
            draw_hud(frame, info, fps, face is not None)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                controller.recenter()
            if key == ord("k"):
                actions.set_paused(True)   # no mouse control while the dots are showing
                new_mapper = run_calibration(camera, tracker, aspect, cfg)
                if new_mapper:
                    controller.set_gaze_mapper(new_mapper)
                print("Still PAUSED. Press ctrl+alt+p to resume.")
            if key == ord("g"):
                print("eye jump:", _gaze_label(controller.toggle_gaze()))
            if key == ord("d"):
                print("dwell click:", "ON" if controller.toggle_dwell() else "OFF")
            try:
                while True:
                    event = hotkeys.events.get_nowait()
                    if event == "quit":
                        raise KeyboardInterrupt
                    if event == "recenter":
                        controller.recenter()
                    if event == "voice_toggle" and voice:
                        voice.toggle()
                    if isinstance(event, tuple) and event[0] == "voice_text":
                        cmd = parse_command(event[1])
                        print(f'voice: "{event[1]}" ->', cmd if cmd else "not understood, ignored")
                        if cmd:
                            apply_command(cmd, actions, cfg["voice"])
                    if event == "gaze_toggle":
                        print("eye jump:", _gaze_label(controller.toggle_gaze()))
                    if event == "dwell_toggle":
                        print("dwell click:", "ON" if controller.toggle_dwell() else "OFF")
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        if voice:
            voice.close()
        if listener:
            listener.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
