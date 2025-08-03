# -*- coding: utf-8 -*-
"""
An advanced, interactive PyQt6 GUI application to analyze audio files.

Features:
- Opens various audio formats using the 'soundfile' library.
- Displays a spectrogram of the audio content.
- Includes Matplotlib's navigation toolbar for interactive zooming and panning.
- Applies a 2D median filter to the spectrogram to reduce noise.
- Toggles an overlay of metadata (timestamp, GPS location) on the plot.

Required Libraries:
- PyQt6: For the graphical user interface.
- soundfile: For robust audio file reading.
- scipy: For the median filter algorithm.
- matplotlib: For plotting.
- numpy: For numerical operations.

You can install these with pip:
pip install PyQt6 soundfile scipy matplotlib numpy
"""
import sys
from datetime import datetime
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout,
    QWidget, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QPushButton,
    QCheckBox, QLineEdit, QLabel
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# --- Import the Navigation Toolbar ---
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class SpectrogramCanvas(FigureCanvas):
    """A Matplotlib canvas for displaying the spectrogram, embedded in a PyQt6 widget."""

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.cbar = None  # To hold the colorbar object
        self.metadata_text_object = None  # To hold the metadata text object
        super().__init__(self.fig)
        self.setParent(parent)
        self.initialize_plot()

    def initialize_plot(self):
        """Sets up the initial empty plot with labels."""
        if self.cbar:
            self.cbar.remove()
            self.cbar = None
        self.axes.clear()
        self.axes.set_title("No file loaded")
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Frequency (Hz)")
        self.fig.tight_layout()
        self.draw()

    def plot_spectrogram(self, Pxx, freqs, bins, filename, metadata_str=None):
        """
        Draws the spectrogram on the canvas from pre-computed data.
        """
        # Preserve the current view limits if they exist (for panning/zooming)
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        is_zoomed = self.axes.get_xlim() != (0.0, 1.0)  # A simple check if view is default

        if self.cbar:
            self.cbar.remove()
            self.cbar = None
        if self.metadata_text_object:
            self.metadata_text_object.remove()
            self.metadata_text_object = None

        self.axes.clear()

        Pxx_dB = 10 * np.log10(Pxx + 1e-12)
        vmax = Pxx_dB.max()
        vmin = vmax - 90

        im = self.axes.pcolormesh(bins, freqs, Pxx_dB, cmap='inferno', shading='gouraud', vmin=vmin, vmax=vmax)

        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Frequency (Hz)")
        self.axes.set_title(f"Spectrogram of {filename.split('/')[-1]}")

        self.cbar = self.fig.colorbar(im, ax=self.axes, format='%+2.0f dB')
        self.cbar.set_label('Intensity [dB]')

        if metadata_str:
            self.metadata_text_object = self.axes.text(
                0.02, 0.98, metadata_str,
                transform=self.axes.transAxes,
                fontsize=10,
                color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.6)
            )

        # Restore the previous view if it was zoomed/panned
        if is_zoomed:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        else:
            self.axes.set_ylim(0, freqs.max())

        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Spectrogram Analyzer")
        self.setGeometry(100, 100, 800, 750)

        # --- Data Storage ---
        self.current_filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.Pxx = None  # Original spectrogram data
        self.freqs = None
        self.bins = None
        self.is_filtered = False

        self._setup_ui()

    def _setup_ui(self):
        """Initializes all UI elements."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open Audio File...", self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        main_layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        # --- Spectrogram Canvas ---
        self.spectrogram_canvas = SpectrogramCanvas(self)

        # --- Create and add the Navigation Toolbar ---
        self.toolbar = NavigationToolbar(self.spectrogram_canvas, self)
        main_layout.addWidget(self.toolbar)

        # Add the canvas itself
        main_layout.addWidget(self.spectrogram_canvas)

        # --- Controls Group Box ---
        controls_group = QGroupBox("Processing Controls")
        main_layout.addWidget(controls_group)
        controls_layout = QHBoxLayout()
        controls_group.setLayout(controls_layout)

        filter_form = QFormLayout()
        self.filter_size_spinbox = QSpinBox()
        self.filter_size_spinbox.setRange(3, 15)
        self.filter_size_spinbox.setSingleStep(2)
        self.filter_size_spinbox.setValue(3)
        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.setCheckable(True)
        self.apply_filter_button.clicked.connect(self.toggle_filter)
        self.apply_filter_button.setEnabled(False)
        filter_form.addRow(QLabel("Median Filter Size:"), self.filter_size_spinbox)
        filter_form.addRow(self.apply_filter_button)

        metadata_form = QFormLayout()
        self.metadata_checkbox = QCheckBox("Show Metadata")
        self.metadata_checkbox.stateChanged.connect(self.redraw_plot)
        self.metadata_checkbox.setEnabled(False)
        self.gps_input = QLineEdit("46°51'35.2\"N 96°45'24.5\"W")
        metadata_form.addRow(self.metadata_checkbox)
        metadata_form.addRow(QLabel("GPS Location:"), self.gps_input)

        controls_layout.addLayout(filter_form)
        controls_layout.addSpacing(20)
        controls_layout.addLayout(metadata_form)

    def open_file_dialog(self):
        """Opens a dialog to select an audio file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav *.flac *.ogg)")
        if file_path:
            self.load_audio_file(file_path)

    def load_audio_file(self, file_path):
        """Loads and processes the selected audio file."""
        try:
            self.current_filepath = file_path
            self.audio_data, self.sample_rate = sf.read(file_path, dtype='float32')
            if self.audio_data.ndim > 1:
                self.audio_data = self.audio_data[:, 0]

            self.Pxx, self.freqs, self.bins, _ = plt.specgram(
                self.audio_data, NFFT=2048, Fs=self.sample_rate, noverlap=1024
            )
            plt.close()

            self.is_filtered = False
            self.apply_filter_button.setChecked(False)
            self.apply_filter_button.setText("Apply Filter")
            self.apply_filter_button.setEnabled(True)
            self.metadata_checkbox.setEnabled(True)

            # Reset the view on the toolbar
            self.toolbar.home()
            self.redraw_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not process file:\n{e}")
            self.spectrogram_canvas.initialize_plot()
            self.apply_filter_button.setEnabled(False)
            self.metadata_checkbox.setEnabled(False)

    def toggle_filter(self):
        """Applies or reverts the median filter."""
        if self.Pxx is not None:
            self.is_filtered = not self.is_filtered
            self.apply_filter_button.setText("Revert" if self.is_filtered else "Apply Filter")
            self.redraw_plot()

    def redraw_plot(self):
        """Redraws the spectrogram with current settings."""
        if self.Pxx is None:
            return

        plot_data = self.Pxx
        if self.is_filtered:
            filter_size = self.filter_size_spinbox.value()
            plot_data = median_filter(self.Pxx, size=filter_size)

        metadata_str = None
        if self.metadata_checkbox.isChecked():
            try:
                filename_stem = self.current_filepath.split('/')[-1].split('.')[0]
                clean_stem = filename_stem.rstrip('T')
                dt_obj = datetime.strptime(clean_stem, '%Y%m%d_%H%M%S')
                formatted_date = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                gps_coords = self.gps_input.text()
                metadata_str = f"{formatted_date}\n{gps_coords}"
            except (ValueError, IndexError):
                metadata_str = "Could not parse timestamp from filename"

        self.spectrogram_canvas.plot_spectrogram(
            plot_data, self.freqs, self.bins, self.current_filepath, metadata_str
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
