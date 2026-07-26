from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QComboBox
from PySide6.QtCore import QThread, Slot
from PySide6.QtGui import QCloseEvent
from .transcription import TranscriptionWorker

import sounddevice as sd

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        self.setMinimumSize(QtCore.QSize(400, 300))
        self.setMaximumSize(QtCore.QSize(800, 600))

        self.transcription_thread = None
        self.transcription_worker = None

        self._close_requested = False

        self.build_widgets()
        self.connect_signals()

    def build_widgets(self):
        self.start_button = QtWidgets.QPushButton("Start")

        self.audio_selector_dropdown = QComboBox()
        self.get_audio_inputs()

        self.transcript_display = QtWidgets.QPlainTextEdit()
        self.transcript_display.setReadOnly(True)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.start_button)
        main_layout.addWidget(self.audio_selector_dropdown)
        main_layout.addWidget(self.transcript_display)
        
        main_container = QtWidgets.QWidget()
        main_container.setLayout(main_layout)

        self.setWindowTitle("Scrivener's Chant")
        self.setCentralWidget(main_container)

    def get_audio_inputs(self):
        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
        except sd.PortAudioError as exc:
            self.audio_selector_dropdown.addItem(
                f"Unable to read audio devices: {exc}",
                None,
            )
            self.audio_selector_dropdown.setEnabled(False)
            self.start_button.setEnabled(False)
            return

        input_device_indices = []

        for device_index, device in enumerate(devices):
            if device["max_input_channels"] <= 0:
                continue

            host_api = host_apis[device["hostapi"]]
            sample_rate = int(device["default_samplerate"])

            display_name = (
                f"{device['name']} "
                f"— {host_api['name']} "
                f"— {sample_rate} Hz"
            )

            self.audio_selector_dropdown.addItem(
                display_name,
                device_index,
            )

            input_device_indices.append(device_index)

    def connect_signals(self):
        self.start_button.clicked.connect(self.toggle_transcription)

    def closeEvent(self, event: QCloseEvent) -> None:
        thread_is_running = (
            self.transcription_thread is not None
            and self.transcription_thread.isRunning()
        )

        if not thread_is_running:
            event.accept()
            return

        self._close_requested = True
        self.stop_transcription()

        # Wait for the worker to finish before actually closing.
        event.ignore()

    @Slot()
    def start_transcription(self) -> None:
        input_device = self.audio_selector_dropdown.currentData()

        if (
            self.transcription_thread is not None
            and self.transcription_thread.isRunning()
        ):
            return


        self.transcription_thread = QThread()
        self.transcription_worker = TranscriptionWorker(
            self.args,
            input_device=input_device,
        )

        self.transcription_worker.moveToThread(
            self.transcription_thread
        )

        self.transcription_thread.started.connect(
            self.transcription_worker.run
        )

        self.transcription_worker.transcription_ready.connect(
            self.display_transcription
        )

        self.transcription_worker.status_changed.connect(
            self.display_status
        )

        self.transcription_worker.finished.connect(
            self.transcription_thread.quit
        )

        self.transcription_worker.finished.connect(
            self.transcription_worker.deleteLater
        )

        self.transcription_thread.finished.connect(
            self.transcription_thread.deleteLater
        )

        self.transcription_thread.finished.connect(
            self.transcription_finished
        )

        self.transcription_worker.error_occurred.connect(
            self.display_error
        )

        self.start_button.setText("Stop")
        self.audio_selector_dropdown.setEnabled(False)
        self.transcription_thread.start()


    @Slot()
    def transcription_finished(self) -> None:
        self.transcription_worker = None
        self.transcription_thread = None

        self.start_button.setText("Start")
        self.start_button.setEnabled(True)
        self.audio_selector_dropdown.setEnabled(True)

        if self._close_requested:
            self.close()

    @Slot()
    def toggle_transcription(self) -> None:
        if self.transcription_worker is None:
            self.start_transcription()
        else:
            self.stop_transcription()

    @Slot()
    def stop_transcription(self) -> None:
        if self.transcription_worker is None:
            return

        self.start_button.setEnabled(False)
        self.start_button.setText("Stopping...")

        self.transcription_worker.request_stop()

    @Slot(str)
    def display_transcription(self, text: str) -> None:
        self.transcript_display.appendPlainText(text)

    @Slot(str)
    def display_status(self, status: str) -> None:
        self.transcript_display.appendPlainText(f"Status: {status}")

    @Slot(str)
    def display_error(self, error: str) -> None:
        self.transcript_display.appendPlainText(
            f"Error: {error}"
        )