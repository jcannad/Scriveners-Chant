#!/usr/bin/env python3
"""Near-real-time microphone transcription prototype.

Audio is held in memory, transcribed in short chunks, and printed to stdout.
No audio or transcript files are created.
"""

from __future__ import annotations

import argparse
import queue
import sys
from collections.abc import Sequence

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16_000
CHANNELS = 1
AUDIO_BLOCK_SECONDS = 0.1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe microphone audio to the console."
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Input-device number. Uses the system default when omitted.",
    )
    parser.add_argument(
        "--model",
        default="base.en",
        help="Faster Whisper model name. Default: base.en",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=5.0,
        help="Seconds of audio processed at a time. Default: 5",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Spoken-language code. Default: en",
    )
    parser.add_argument(
        "--device-type",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Inference device. Default: cpu",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Faster Whisper compute type. Default: int8",
    )

    args = parser.parse_args(argv)

    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be greater than zero")

    return args


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
    """Return transcription text for one 16 kHz mono audio chunk."""

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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_devices:
        print(sd.query_devices())
        return 0

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info

        if status:
            print(f"\nAudio warning: {status}", file=sys.stderr)

        # The callback buffer is reused by PortAudio, so copy its contents.
        audio_queue.put(indata[:, 0].copy())

    print(
        f"Loading Faster Whisper model '{args.model}' "
        f"on {args.device_type}..."
    )

    model = WhisperModel(
        args.model,
        device=args.device_type,
        compute_type=args.compute_type,
    )

    chunk_sample_count = int(SAMPLE_RATE * args.chunk_seconds)
    audio_block_size = int(SAMPLE_RATE * AUDIO_BLOCK_SECONDS)

    audio_buffer = np.empty(0, dtype=np.float32)
    processed_sample_count = 0

    print("Listening. Press Ctrl+C to stop.")
    print(
        f"Input device: "
        f"{args.input_device if args.input_device is not None else 'default'}"
    )
    print(f"Chunk length: {args.chunk_seconds:g} seconds\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=audio_block_size,
            device=args.input_device,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                audio_block = audio_queue.get()
                audio_buffer = np.concatenate((audio_buffer, audio_block))

                while audio_buffer.size >= chunk_sample_count:
                    audio_chunk = audio_buffer[:chunk_sample_count]
                    audio_buffer = audio_buffer[chunk_sample_count:]

                    chunk_start = processed_sample_count / SAMPLE_RATE
                    processed_sample_count += chunk_sample_count
                    chunk_end = processed_sample_count / SAMPLE_RATE

                    text = transcribe_chunk(
                        model=model,
                        audio=audio_chunk,
                        language=args.language,
                    )

                    if text:
                        start_label = format_timestamp(chunk_start)
                        end_label = format_timestamp(chunk_end)
                        print(
                            f"[{start_label}–{end_label}] {text}",
                            flush=True,
                        )

    except KeyboardInterrupt:
        print("\nStopped listening.")
        return 0
    except sd.PortAudioError as exc:
        print(f"\nAudio-device error: {exc}", file=sys.stderr)
        print(
            "Run this script with --list-devices and choose an input "
            "using --input-device NUMBER.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())