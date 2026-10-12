"""Microphone -> text, all on this computer (faster-whisper). Audio is kept in memory only.

Toggle-to-talk: press the hotkey once to start listening, press again to stop (or it stops by
itself after `max_seconds`). Then the audio is transcribed on a background thread and the text
is handed to `on_text`.

Needs the macOS Microphone permission for the app running Python.
The first time you use it, faster-whisper downloads the speech model once (needs internet
for that single download; after that everything is offline).
"""
import threading


class VoiceInput:
    def __init__(self, vcfg, on_text):
        self.cfg = vcfg
        self.on_text = on_text
        self.status = "voice: idle"
        self._model = None
        self._stream = None
        self._chunks = []
        self._timer = None
        self._lock = threading.Lock()

    @property
    def recording(self):
        return self._stream is not None

    def toggle(self):
        with self._lock:
            if self.recording:
                self._stop_and_transcribe()
            else:
                self._start()

    # ---- recording ----
    def _start(self):
        try:
            import sounddevice as sd
            self._chunks = []
            self._stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                                          callback=lambda data, *_: self._chunks.append(data.copy()))
            self._stream.start()
        except Exception as e:   # no mic, or permission not granted yet
            self._stream = None
            self.status = f"voice error: {e} (check Microphone permission)"
            return
        self.status = "LISTENING... (press the voice hotkey again to stop)"
        self._timer = threading.Timer(self.cfg["max_seconds"], self._auto_stop)
        self._timer.daemon = True
        self._timer.start()

    def _auto_stop(self):
        with self._lock:
            if self.recording:
                self._stop_and_transcribe()

    def _stop_and_transcribe(self):
        if self._timer:
            self._timer.cancel()
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()
        chunks, self._chunks = self._chunks, []
        self.status = "voice: transcribing..."
        threading.Thread(target=self._transcribe, args=(chunks,), daemon=True).start()

    # ---- speech to text ----
    def _transcribe(self, chunks):
        try:
            import numpy as np
            if not chunks:
                self.status = "voice: heard nothing"
                return
            audio = np.concatenate(chunks).flatten()
            if len(audio) < 16000 * 0.3:
                self.status = "voice: too short"
                return
            if self._model is None:
                self.status = "voice: loading speech model (first time downloads it)..."
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.cfg["model"], device="cpu", compute_type="int8")
                self.status = "voice: transcribing..."
            segments, _ = self._model.transcribe(audio, language=self.cfg["language"],
                                                 beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            self.status = f'voice heard: "{text}"' if text else "voice: heard nothing"
            if text:
                self.on_text(text)
        except Exception as e:
            self.status = f"voice error: {e}"

    def close(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
