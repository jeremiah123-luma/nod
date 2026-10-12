"""Nod: hands-free computer control. Run with:  .venv/bin/python -m nod.main

Add --dry-run to print what the mouse WOULD do without touching it.
Add --no-dashboard to skip the browser Command Center, --no-open to not open the browser automatically.
Global hotkeys (work from any app): ctrl+alt+p pause/resume, ctrl+alt+c recenter, ctrl+alt+d dwell click on/off,
ctrl+alt+g eye jump on/off, ctrl+alt+k calibrate eye jump, ctrl+alt+v voice (if enabled), ctrl+alt+q quit.
"""
import argparse
import queue
import time
import webbrowser

import cv2

from nod import actions, hotkeys, mesh, permissions
from nod.bridge import Bridge
from nod.camera import Camera
from nod.config import load_config
from nod.controller import Controller
from nod.dashboard import Dashboard
from nod.face import FaceTracker
from nod.gaze import GazeMapper
from nod.gaze_calibration import CAL_PATH, run_calibration, signal_tag
from nod.hud import draw_hud, draw_mesh
from nod.voice import apply_command, parse_command

PUBLISH_EVERY = 1 / 15      # dashboard refresh rate (seconds)


def _gaze_label(state):
    return "not calibrated (press ctrl+alt+k first)" if state is None else ("ON" if state else "OFF")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print mouse actions instead of doing them")
    parser.add_argument("--no-dashboard", action="store_true", help="don't start the browser Command Center")
    parser.add_argument("--no-open", action="store_true", help="start the Command Center but don't open the browser")
    args = parser.parse_args()

    cfg = load_config()
    box = {"cfg": cfg}                                     # the live settings (the dashboard can change them)
    actions.set_dry_run(args.dry_run)
    camera = Camera(cfg["camera"]["index"], cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"])
    aspect = cfg["camera"]["width"] / cfg["camera"]["height"]
    tracker = FaceTracker(cfg)
    mapper = GazeMapper.load(CAL_PATH, actions.screen_size(), signal_tag(cfg))
    controller = Controller(cfg, actions, mapper)
    if mapper:
        print(f"Eye jump: calibrated ({mapper.label}, error ~{mapper.error_px:.0f}px). Recalibrate any time with ctrl+alt+k.")
    else:
        print("Eye jump: NOT calibrated (or the old calibration is out of date). Press ctrl+alt+k, then look at each dot.")

    voice = None
    if cfg["voice"]["enabled"]:
        from nod.voice_input import VoiceInput
        voice = VoiceInput(cfg["voice"], lambda text: hotkeys.events.put(("voice_text", text)))
        print("Voice is ON: ctrl+alt+v to start/stop talking. Say: scroll up/down, click, pause, resume, type ...")
    listener = hotkeys.start(cfg)

    bridge = Bridge(actions)
    dash = None
    if cfg["ui"]["dashboard"] and not args.no_dashboard:
        try:
            dash = Dashboard(lambda: box["cfg"], bridge.on_command, bridge.on_settings,
                             meta_extra=lambda: {"mesh": mesh.meta()}, port=cfg["ui"]["port"])
            url = dash.start()
            print(f"Command Center: {url}")
            if cfg["ui"]["auto_open"] and not args.no_open:
                webbrowser.open(url)
        except OSError as e:
            print(f"WARNING: couldn't start the dashboard ({e}). Nod still works without it.")
            dash = None

    show_preview = cfg["ui"]["preview_window"]
    window = cfg["hud"]["window_name"]
    print("Nod is running. ctrl+alt+p = pause/resume, ctrl+alt+c = recenter, ctrl+alt+q = quit.")
    print("Emergency stop: slam the real mouse into the top-left corner of the screen.")
    actions.set_paused(True)  # start paused so nothing moves until you press ctrl+alt+p (or Start control)
    print("Started PAUSED. Press ctrl+alt+p (or click Start control in the Command Center) when you're ready.")

    fps_box = [30.0]

    def calibrate():
        """Runs the dot screen. The mouse is never touched by it, and Nod's pause state is left alone."""
        print("Eye calibration: look at each dot until it moves. ESC cancels. (~35 seconds)")
        shown = [0.0]

        def tick(step, total, text):
            """Keeps the dashboard informed while this window has the main loop busy."""
            if dash and time.monotonic() - shown[0] > 0.3:
                shown[0] = time.monotonic()
                snap = controller.snapshot()
                snap.update(fps=round(fps_box[0], 1), perms=permissions.status(), busy=f"Calibrating eyes: {text}")
                dash.publish(snap)

        tick(0, 0, "starting")
        new_mapper = run_calibration(camera, tracker, aspect, box["cfg"], on_tick=tick)
        if new_mapper:
            controller.set_gaze_mapper(new_mapper)
            print("Eye jump is now ON. Look at a far part of the screen and hold steady to jump.")

    def handle(item):
        """One thing to do, from a hotkey, the preview window, or the dashboard. Returns True to quit."""
        if item == "quit":
            return True
        if item == "recenter":
            controller.recenter()
        elif item == "calibrate":
            calibrate()
        elif item in ("gaze_toggle", "toggle_gaze"):
            print("eye jump:", _gaze_label(controller.toggle_gaze()))
        elif item in ("dwell_toggle", "toggle_dwell"):
            print("dwell click:", "ON" if controller.toggle_dwell() else "OFF")
        elif item == "voice_toggle" and voice:
            voice.toggle()
        elif isinstance(item, tuple) and item[0] == "voice_text":
            cmd = parse_command(item[1])
            print(f'voice: "{item[1]}" ->', cmd if cmd else "not understood, ignored")
            controller.note("voice", f'Heard "{item[1]}"' + (f" → {cmd.kind}" if cmd else " (not understood)"))
            if cmd:
                apply_command(cmd, actions, box["cfg"]["voice"])
        elif isinstance(item, tuple) and item[0] == "settings":
            box["cfg"] = item[1]
            controller.reload(item[1])
            print("settings updated from the dashboard")
        return False

    fps, last, last_pub, perms, perms_at = 0.0, time.monotonic(), 0.0, {}, -99.0
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
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            fps_box[0] = fps
            last = now

            # ---- the browser dashboard ----
            if dash and now - last_pub >= PUBLISH_EVERY:
                last_pub = now
                if now - perms_at > 2.0:                  # permissions rarely change; don't ask every frame
                    perms, perms_at = permissions.status(), now
                snap = controller.snapshot()
                snap.update(fps=round(fps, 1), perms=perms, hotkeys_on=listener is not None,
                            mesh=mesh.compact(face.landmarks) if face is not None else None)
                if voice:
                    snap["voice"] = voice.status
                dash.publish(snap)

            # ---- the preview window ----
            quit_now = False
            if show_preview:
                if face is not None:
                    draw_mesh(frame, face.landmarks)
                info = controller.hud_info()
                if voice:
                    info["lines"].append(voice.status)
                draw_hud(frame, info, fps, face is not None)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                keymap = {ord("c"): "recenter", ord("k"): "calibrate", ord("g"): "toggle_gaze", ord("d"): "toggle_dwell"}
                if key == ord("q"):
                    quit_now = True
                elif key in keymap:
                    quit_now = handle(keymap[key])

            # ---- hotkeys + dashboard commands ----
            items = bridge.drain()
            try:
                while True:
                    items.append(hotkeys.events.get_nowait())
            except queue.Empty:
                pass
            for item in items:
                quit_now = handle(item) or quit_now
            if quit_now:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if dash:
            dash.stop()
        if voice:
            voice.close()
        if listener:
            listener.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
