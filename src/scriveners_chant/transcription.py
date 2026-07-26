from __future__ import annotations

import queue
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PySide6.QtCore import QObject, Signal, Slot

SAMPLE_RATE = 16_000
CHANNELS = 1
AUDIO_BLOCK_SECONDS = 0.1

def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def transcribe_chunk(
    model: WhisperModel,
    audio: np.ndarray,
    language: str,
) -> str:
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

class TranscriptionWorker(QObject):
    transcription_ready = Signal(str)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, args, input_device):
        super().__init__()
        self.args = args
        self.input_device = input_device
        self._stop_requested = threading.Event()

        
    @Slot()
    def request_stop(self) -> None:
        self._stop_requested.set()

    @Slot()
    def run(self) -> None:
        self._stop_requested.clear()

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        warning_queue: queue.Queue[str] = queue.Queue()

        def audio_callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            del frames, time_info

            if status:
                warning_queue.put_nowait(str(status))

            # PortAudio reuses its buffer, so keep our own copy.
            audio_queue.put_nowait(indata[:, 0].copy())

        try:
            self.status_changed.emit(
                f"Loading model: {self.args.model}"
            )

            model = WhisperModel(
                self.args.model,
                device=self.args.device_type,
                compute_type=self.args.compute_type,
            )

            chunk_sample_count = int(SAMPLE_RATE * self.args.chunk_seconds)
            audio_block_size = int(SAMPLE_RATE * AUDIO_BLOCK_SECONDS)

            audio_buffer = np.empty(0, dtype=np.float32)
            processed_sample_count = 0

            transcript_directory = Path("transcripts")
            transcript_directory.mkdir(parents=True, exist_ok=True)

            filename = ("transcription_"f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"".txt")

            transcript_path = transcript_directory / filename

            with transcript_path.open("w", encoding="utf-8") as transcript_file:

                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=audio_block_size,
                    device=self.input_device,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=audio_callback,
                ):
                    self.status_changed.emit(
                        f"Listening — saving to {filename}"
                    )

                    while not self._stop_requested.is_set():
                        try:
                            warning = warning_queue.get_nowait()
                        except queue.Empty:
                            pass
                        else:
                            self.status_changed.emit(
                                f"Audio warning: {warning}"
                            )

                        try:
                            audio_block = audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        audio_buffer = np.concatenate((audio_buffer, audio_block))

                        while (
                            audio_buffer.size >= chunk_sample_count
                            and not self._stop_requested.is_set()
                        ):
                            audio_chunk = audio_buffer[:chunk_sample_count]
                            audio_buffer = audio_buffer[chunk_sample_count:]

                            chunk_start = processed_sample_count / SAMPLE_RATE
                            processed_sample_count += chunk_sample_count
                            chunk_end = processed_sample_count / SAMPLE_RATE

                            text = transcribe_chunk(
                                model=model,
                                audio=audio_chunk,
                                language=self.args.language,
                            )

                            if not text:
                                continue

                            start_label = format_timestamp(chunk_start)
                            end_label = format_timestamp(chunk_end)

                            transcript_line = (f"[{start_label}–{end_label}] "f"{text}")

                            transcript_file.write(transcript_line + "\n")
                            transcript_file.flush()

                            self.transcription_ready.emit(transcript_line)

            self.status_changed.emit("Transcription stopped")

        except sd.PortAudioError as exc:
            self.error_occurred.emit(f"Audio-device error: {exc}")

        except Exception as exc:
            self.error_occurred.emit(f"Transcription error: {exc}")

        finally:
            self.finished.emit()