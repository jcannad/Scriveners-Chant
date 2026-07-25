from PySide6.QtWidgets import QApplication
from .main_window import MainWindow

from collections.abc import Sequence

import sys
import argparse

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


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.show()


    app.exec()


if __name__ == "__main__":
    raise SystemExit(main())