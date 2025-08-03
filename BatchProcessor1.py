# -*- coding: utf-8 -*-
"""
A PyQt6 GUI application for batch converting audio files into spectrogram images.

This application provides a user-friendly interface for the command-line
batch processor, allowing users to select a directory and process all .wav
files within it.

Features:
- A "Select Directory" button to choose the folder containing .wav files.
- A "Start Processing" button to begin the batch job.
- A progress bar to visualize the overall progress.
- A console log to display real-time status updates for each file.
- Processing is run in a separate thread to keep the GUI responsive.

Required Libraries:
- PyQt6
- soundfile
- matplotlib
- numpy

You can install these with pip:
pip install PyQt6 soundfile matplotlib numpy
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Use a non-interactive backend for Matplotlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLineEdit, QProgressBar, QTextEdit, QLabel
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal


def create_spectrogram(audio_path, output_path):
    """
    Generates and saves a single spectrogram image from an audio file.
    This function is designed to be called from the worker thread.
    """
    try:
        data, sample_rate = sf.read(audio_path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]

        fig, ax = plt.subplots(figsize=(12, 8))
        Pxx, freqs, bins, im = ax.specgram(
            data, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap='inferno'
        )

        if Pxx.size > 0:
            Pxx_dB = 10 * np.log10(Pxx + 1e-12)
            vmax = Pxx_dB.max()
            vmin = vmax - 90
            im.set_clim(vmin, vmax)

        ax.set_title(audio_path.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, sample_rate / 2)
        fig.colorbar(im, ax=ax, format='%+2.0f dB').set_label('Intensity [dB]')
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)

    except Exception as e:
        # This error will be emitted as a signal by the worker
        raise e
    finally:
        plt.close(fig)


class Worker(QObject):
    """
    A worker thread for performing the batch processing task without freezing the GUI.
    """
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)

    def __init__(self, file_list):
        super().__init__()
        self.file_list = file_list

    def run(self):
        """The main processing loop."""
        total_files = len(self.file_list)
        for i, audio_file in enumerate(self.file_list):
            self.log_message.emit(f"Processing file {i + 1}/{total_files}: {audio_file.name}")
            output_filepath = audio_file.with_suffix(".png")
            try:
                create_spectrogram(audio_file, output_filepath)
            except Exception as e:
                self.log_message.emit(f"  [ERROR] Could not process {audio_file.name}: {e}")

            self.progress.emit(int(((i + 1) / total_files) * 100))

        self.log_message.emit("\nBatch processing complete.")
        self.finished.emit()


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch Spectrogram Generator")
        self.setGeometry(100, 100, 600, 400)
        self.worker_thread = None
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        """Sets up all the UI elements."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Directory Selection ---
        dir_layout = QHBoxLayout()
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setReadOnly(True)
        self.dir_path_edit.setPlaceholderText("Select a directory containing .wav files...")
        select_dir_button = QPushButton("Select Directory...")
        select_dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.dir_path_edit)
        dir_layout.addWidget(select_dir_button)
        main_layout.addLayout(dir_layout)

        # --- Start Button ---
        self.start_button = QPushButton("Start Processing")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_processing)
        main_layout.addWidget(self.start_button)

        # --- Progress Bar and Console ---
        main_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        main_layout.addWidget(self.console_log)

    def select_directory(self):
        """Opens a dialog to select the input directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_path_edit.setText(directory)
            self.start_button.setEnabled(True)
            self.console_log.clear()
            self.progress_bar.setValue(0)

    def start_processing(self):
        """Initiates the spectrogram generation in a worker thread."""
        input_dir = self.dir_path_edit.text()
        if not input_dir:
            return

        input_path = Path(input_dir)
        # --- FIX: Make glob pattern case-insensitive to find .wav and .WAV ---
        audio_files = list(input_path.glob("*.[wW][aA][vV]"))

        if not audio_files:
            self.console_log.setText(f"No .wav files found in '{input_path}'.")
            return

        # --- Disable UI elements during processing ---
        self.start_button.setEnabled(False)
        self.console_log.clear()

        # --- Set up and start the worker thread ---
        self.worker_thread = QThread()
        self.worker = Worker(audio_files)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals from the worker to GUI slots
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log_message.connect(self.console_log.append)

        # Re-enable the start button when finished
        self.worker_thread.finished.connect(lambda: self.start_button.setEnabled(True))

        self.worker_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
