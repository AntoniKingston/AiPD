import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, QTabWidget,
                             QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
                             QLabel, QSlider, QSplitter, QMessageBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns
from scipy.io import wavfile  # For WAV file loading
import os.path  # For file path validation
from matplotlib.widgets import SpanSelector

# Import functions from separate file
from functions import *


class AudioAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Signal Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.audio_data = None
        self.selected_region = None  # Will store (start_index, end_index) of selection
        self.sample_rate = None
        self.file_path = None

        # Add span selector objects
        self.time_domain_span_selector = None
        self.window_function_span_selector = None

        # Default spectrogram parameters
        self.spec_window = 'rectangular'
        self.spec_frame_dur = 0.05
        self.spec_overlap = 0.5
        self.max_spec_freq = 2000

        self.init_ui()

    def init_ui(self):
        # Create main menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # Add open file action to menu
        open_action = QAction('Open Audio File (.wav)', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)

        # Add an exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create toolbar and add actions
        toolbar = self.addToolBar('File')
        toolbar.addAction(open_action)

        # Create tab widget for bookmarks
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs for each bookmark
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()
        self.setup_tab5()

        self.tabs.currentChanged.connect(self.handle_tab_changed)

        # Add status bar
        self.statusBar().showMessage('Ready')

        self.show()

    def open_audio_file(self):
        """Open a WAV file and load its data"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "WAV Files (*.wav)")

        if not filename:
            return  # User canceled the dialog

        try:
            # Check if the file exists
            if not os.path.exists(filename):
                QMessageBox.critical(self, "Error", f"File not found: {filename}")
                return

            # Try to load the WAV file
            self.sample_rate, audio_data = wavfile.read(filename)

            # Handle multi-channel audio by taking the first channel
            if len(audio_data.shape) > 1:
                self.audio_data = audio_data[:, 0]  # Take first channel
            else:
                self.audio_data = audio_data

            # Convert data to float for processing if needed
            if self.audio_data.dtype != np.float32 and self.audio_data.dtype != np.float64:
                self.audio_data = self.audio_data.astype(np.float32)

                # Normalize the data between -1.0 and 1.0 if it's in integer format
                max_value = np.iinfo(audio_data.dtype).max
                self.audio_data = self.audio_data / max_value

            # Store the file path and update window title
            self.file_path = filename
            self.setWindowTitle(f"Audio Signal Analyzer - {os.path.basename(filename)}")

            # Show loading info in status bar
            duration = len(self.audio_data) / self.sample_rate
            self.statusBar().showMessage(f"Loaded: {os.path.basename(filename)} | "
                                         f"Sample rate: {self.sample_rate} Hz | "
                                         f"Duration: {duration:.2f} seconds")

            # Reset the selected region
            self.selected_region = None

            # Update all plots with the new data
            self.update_all_plots()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file: {str(e)}")
            self.statusBar().showMessage(f"Error loading {os.path.basename(filename)}: {str(e)}")

    def setup_tab1(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Top section with plot and button
        top_layout = QHBoxLayout()

        # Main signal plot
        main_plot_widget = QWidget()
        main_plot_layout = QVBoxLayout()
        main_figure = Figure(figsize=(10, 4))
        main_canvas = FigureCanvas(main_figure)
        main_toolbar = NavigationToolbar(main_canvas, self)
        main_plot_layout.addWidget(main_toolbar)
        main_plot_layout.addWidget(main_canvas)
        main_plot_widget.setLayout(main_plot_layout)

        # Add main plot to layout
        top_layout.addWidget(main_plot_widget, stretch=4)

        # Add selection button
        select_button = QPushButton("Select Frame")
        select_button.clicked.connect(self.copy_selected_fragment)
        top_layout.addWidget(select_button, stretch=1, alignment=Qt.AlignTop)

        layout.addLayout(top_layout)

        # Selected subsection plot
        subsection_label = QLabel("Selected Fragment:")
        layout.addWidget(subsection_label)

        subsection_figure = Figure(figsize=(10, 2))
        subsection_canvas = FigureCanvas(subsection_figure)
        layout.addWidget(subsection_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Time Domain")

        # Store references to canvas and figures for later updates
        self.time_domain_main_figure = main_figure
        self.time_domain_main_canvas = main_canvas
        self.time_domain_subsection_figure = subsection_figure
        self.time_domain_subsection_canvas = subsection_canvas

    def setup_tab2(self):
        """Set up the acoustic features tab with 6 different plots"""
        tab = QWidget()
        main_layout = QVBoxLayout()

        # Add title
        title_label = QLabel("Acoustic Features Analysis")
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setBold(True)
        font.setPointSize(12)
        title_label.setFont(font)
        main_layout.addWidget(title_label)

        # Create a 3x2 grid layout for the six plots
        grid_layout = QVBoxLayout()

        # First row: Volume and Frequency Centroid
        row1_layout = QHBoxLayout()

        # Volume plot
        volume_widget = QWidget()
        volume_layout = QVBoxLayout()
        volume_label = QLabel("Volume (Vol)")
        volume_layout.addWidget(volume_label)

        volume_figure = Figure(figsize=(5, 2))
        volume_canvas = FigureCanvas(volume_figure)
        volume_layout.addWidget(volume_canvas)
        volume_widget.setLayout(volume_layout)

        # Frequency Centroid plot
        fc_widget = QWidget()
        fc_layout = QVBoxLayout()
        fc_label = QLabel("Frequency Centroid (FC)")
        fc_layout.addWidget(fc_label)

        fc_figure = Figure(figsize=(5, 2))
        fc_canvas = FigureCanvas(fc_figure)
        fc_layout.addWidget(fc_canvas)
        fc_widget.setLayout(fc_layout)

        # Add to first row
        row1_layout.addWidget(volume_widget)
        row1_layout.addWidget(fc_widget)
        grid_layout.addLayout(row1_layout)

        # Second row: Effective Bandwidth and Band Energy Ratio
        row2_layout = QHBoxLayout()

        # Effective Bandwidth plot
        bw_widget = QWidget()
        bw_layout = QVBoxLayout()
        bw_label = QLabel("Effective Bandwidth (BW)")
        bw_layout.addWidget(bw_label)

        bw_figure = Figure(figsize=(5, 2))
        bw_canvas = FigureCanvas(bw_figure)
        bw_layout.addWidget(bw_canvas)
        bw_widget.setLayout(bw_layout)

        # Band Energy Ratio plot
        ber_widget = QWidget()
        ber_layout = QVBoxLayout()
        ber_label = QLabel("Band Energy Ratio (BER)")
        ber_layout.addWidget(ber_label)

        ber_figure = Figure(figsize=(5, 2))
        ber_canvas = FigureCanvas(ber_figure)
        ber_layout.addWidget(ber_canvas)
        ber_widget.setLayout(ber_layout)

        # Add to second row
        row2_layout.addWidget(bw_widget)
        row2_layout.addWidget(ber_widget)
        grid_layout.addLayout(row2_layout)

        # Third row: Spectral Flatness Measure and Spectral Crest Factor
        row3_layout = QHBoxLayout()

        # Spectral Flatness Measure plot
        sfm_widget = QWidget()
        sfm_layout = QVBoxLayout()
        sfm_label = QLabel("Spectral Flatness Measure (SFM)")
        sfm_layout.addWidget(sfm_label)

        sfm_figure = Figure(figsize=(5, 2))
        sfm_canvas = FigureCanvas(sfm_figure)
        sfm_layout.addWidget(sfm_canvas)
        sfm_widget.setLayout(sfm_layout)

        # Spectral Crest Factor plot
        scf_widget = QWidget()
        scf_layout = QVBoxLayout()
        scf_label = QLabel("Spectral Crest Factor (SCF)")
        scf_layout.addWidget(scf_label)

        scf_figure = Figure(figsize=(5, 2))
        scf_canvas = FigureCanvas(scf_figure)
        scf_layout.addWidget(scf_canvas)
        scf_widget.setLayout(scf_layout)

        # Add to third row
        row3_layout.addWidget(sfm_widget)
        row3_layout.addWidget(scf_widget)
        grid_layout.addLayout(row3_layout)

        # Add the grid layout to the main layout
        main_layout.addLayout(grid_layout)

        # Add analysis controls at the bottom (optional)
        controls_layout = QHBoxLayout()

        # Frame size control
        frame_size_label = QLabel("Frame Size (ms):")
        controls_layout.addWidget(frame_size_label)

        frame_size_slider = QSlider(Qt.Horizontal)
        frame_size_slider.setMinimum(10)  # 10ms
        frame_size_slider.setMaximum(100)  # 100ms
        frame_size_slider.setValue(20)  # Default 20ms
        frame_size_slider.setTickPosition(QSlider.TicksBelow)
        frame_size_slider.setTickInterval(10)
        frame_size_slider.valueChanged.connect(self.on_acoustic_frame_size_changed)
        controls_layout.addWidget(frame_size_slider)

        frame_size_value_label = QLabel("20 ms")
        controls_layout.addWidget(frame_size_value_label)

        # Hop size control
        hop_size_label = QLabel("Hop Size (%):")
        controls_layout.addWidget(hop_size_label)

        hop_size_slider = QSlider(Qt.Horizontal)
        hop_size_slider.setMinimum(10)  # 10%
        hop_size_slider.setMaximum(100)  # 100%
        hop_size_slider.setValue(50)  # Default 50%
        hop_size_slider.setTickPosition(QSlider.TicksBelow)
        hop_size_slider.setTickInterval(10)
        hop_size_slider.valueChanged.connect(self.on_acoustic_hop_size_changed)
        controls_layout.addWidget(hop_size_slider)

        hop_size_value_label = QLabel("50%")
        controls_layout.addWidget(hop_size_value_label)

        # Update button
        update_button = QPushButton("Update Analysis")
        update_button.clicked.connect(self.update_acoustic_features)
        controls_layout.addWidget(update_button)

        main_layout.addLayout(controls_layout)

        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "Acoustic Features")

        # Store references to the figures and canvases
        self.acoustic_figures = {
            'volume': volume_figure,
            'fc': fc_figure,
            'bw': bw_figure,
            'ber': ber_figure,
            'sfm': sfm_figure,
            'scf': scf_figure
        }

        self.acoustic_canvases = {
            'volume': volume_canvas,
            'fc': fc_canvas,
            'bw': bw_canvas,
            'ber': ber_canvas,
            'sfm': sfm_canvas,
            'scf': scf_canvas
        }

        # Store references to the sliders and their value labels
        self.frame_size_slider = frame_size_slider
        self.frame_size_value_label = frame_size_value_label
        self.hop_size_slider = hop_size_slider
        self.hop_size_value_label = hop_size_value_label

        # Default values for acoustic analysis parameters
        self.acoustic_frame_size_ms = 20  # 20ms
        self.acoustic_hop_size_percent = 50  # 50%

    def setup_tab3(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Top section with plot and button
        top_layout = QHBoxLayout()

        # Main signal plot (same as tab 1)
        main_plot_widget = QWidget()
        main_plot_layout = QVBoxLayout()
        main_figure = Figure(figsize=(10, 4))
        main_canvas = FigureCanvas(main_figure)
        main_toolbar = NavigationToolbar(main_canvas, self)
        main_plot_layout.addWidget(main_toolbar)
        main_plot_layout.addWidget(main_canvas)
        main_plot_widget.setLayout(main_plot_layout)

        # Add main plot to layout
        top_layout.addWidget(main_plot_widget, stretch=4)

        # Add selection button
        select_button = QPushButton("Select Frame")
        select_button.clicked.connect(self.copy_selected_fragment)
        top_layout.addWidget(select_button, stretch=1, alignment=Qt.AlignTop)

        layout.addLayout(top_layout)

        # Window function selection and windowed signal plot
        window_layout = QHBoxLayout()
        window_label = QLabel("Window Function:")
        window_combo = QComboBox()
        window_combo.addItems(["rectangular", "triangular", "hamming", "hanning", "blackman"])
        window_combo.currentTextChanged.connect(self.update_windowed_plot)
        window_layout.addWidget(window_label)
        window_layout.addWidget(window_combo)
        window_layout.addStretch()

        layout.addLayout(window_layout)

        # Create a splitter for the two bottom plots
        bottom_splitter = QSplitter(Qt.Horizontal)

        # Left side: Time domain windowed plot
        windowed_widget = QWidget()
        windowed_layout = QVBoxLayout()
        windowed_label = QLabel("Windowed Signal (Time Domain):")
        windowed_layout.addWidget(windowed_label)

        windowed_figure = Figure(figsize=(5, 2))
        windowed_canvas = FigureCanvas(windowed_figure)
        windowed_layout.addWidget(windowed_canvas)
        windowed_widget.setLayout(windowed_layout)

        # Right side: Frequency domain windowed plot
        freq_windowed_widget = QWidget()
        freq_windowed_layout = QVBoxLayout()
        freq_windowed_label = QLabel("Windowed Signal (Frequency Domain):")
        freq_windowed_layout.addWidget(freq_windowed_label)

        freq_windowed_figure = Figure(figsize=(5, 2))
        freq_windowed_canvas = FigureCanvas(freq_windowed_figure)
        freq_windowed_layout.addWidget(freq_windowed_canvas)
        freq_windowed_widget.setLayout(freq_windowed_layout)

        # Add both widgets to the splitter
        bottom_splitter.addWidget(windowed_widget)
        bottom_splitter.addWidget(freq_windowed_widget)

        # Add splitter to main layout
        layout.addWidget(bottom_splitter)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Window Function")

        # Store references to canvas and figures for later updates
        self.window_function_main_figure = main_figure
        self.window_function_main_canvas = main_canvas
        self.windowed_figure = windowed_figure
        self.windowed_canvas = windowed_canvas
        self.freq_windowed_figure = freq_windowed_figure
        self.freq_windowed_canvas = freq_windowed_canvas
        self.window_combo = window_combo

    def setup_tab4(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Controls for spectrogram parameters
        controls_layout = QVBoxLayout()

        # Window function selection
        window_layout = QHBoxLayout()
        window_label = QLabel("Window Function:")
        self.spec_window_combo = QComboBox()
        self.spec_window_combo.addItems(["rectangular", "triangular", "hamming", "hanning", "blackman"])
        self.spec_window_combo.currentTextChanged.connect(self.on_spectrogram_param_changed)
        window_layout.addWidget(window_label)
        window_layout.addWidget(self.spec_window_combo)
        window_layout.addStretch()
        controls_layout.addLayout(window_layout)

        # Frame duration slider
        frame_dur_layout = QHBoxLayout()
        frame_dur_label = QLabel("Frame Duration (ms):")
        self.frame_dur_slider = QSlider(Qt.Horizontal)
        self.frame_dur_slider.setMinimum(20)  # 0.02s
        self.frame_dur_slider.setMaximum(200)  # 0.2s
        self.frame_dur_slider.setValue(50)  # Default 0.05s
        self.frame_dur_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_dur_slider.setTickInterval(20)
        self.frame_dur_value_label = QLabel("50 ms")
        self.frame_dur_slider.valueChanged.connect(self.on_frame_dur_changed)

        frame_dur_layout.addWidget(frame_dur_label)
        frame_dur_layout.addWidget(self.frame_dur_slider)
        frame_dur_layout.addWidget(self.frame_dur_value_label)
        controls_layout.addLayout(frame_dur_layout)

        # Overlap slider
        overlap_layout = QHBoxLayout()
        overlap_label = QLabel("Overlap:")
        self.overlap_slider = QSlider(Qt.Horizontal)
        self.overlap_slider.setMinimum(0)  # 0.0
        self.overlap_slider.setMaximum(99)  # 0.99
        self.overlap_slider.setValue(50)  # Default 0.5
        self.overlap_slider.setTickPosition(QSlider.TicksBelow)
        self.overlap_slider.setTickInterval(10)
        self.overlap_value_label = QLabel("0.50")
        self.overlap_slider.valueChanged.connect(self.on_overlap_changed)

        overlap_layout.addWidget(overlap_label)
        overlap_layout.addWidget(self.overlap_slider)
        overlap_layout.addWidget(self.overlap_value_label)
        controls_layout.addLayout(overlap_layout)

        # Max frequency slider
        max_freq_layout = QHBoxLayout()
        max_freq_label = QLabel("Max. Frequency:")
        self.max_freq_slider = QSlider(Qt.Horizontal)
        self.max_freq_slider.setMinimum(500)
        self.max_freq_slider.setMaximum(20000)
        self.max_freq_slider.setValue(2000)
        self.max_freq_slider.setTickPosition(QSlider.TicksBelow)
        self.max_freq_slider.setTickInterval(100)
        self.max_freq_value_label = QLabel("2000Hz")
        self.max_freq_slider.valueChanged.connect(self.on_max_freq_changed)

        max_freq_layout.addWidget(max_freq_label)
        max_freq_layout.addWidget(self.max_freq_slider)
        max_freq_layout.addWidget(self.max_freq_value_label)
        controls_layout.addLayout(max_freq_layout)



        layout.addLayout(controls_layout)




        # Spectrogram plot
        spectrogram_label = QLabel("Spectrogram:")
        layout.addWidget(spectrogram_label)

        self.spectrogram_figure = Figure(figsize=(10, 6))
        self.spectrogram_canvas = FigureCanvas(self.spectrogram_figure)
        self.spectrogram_toolbar = NavigationToolbar(self.spectrogram_canvas, self)
        layout.addWidget(self.spectrogram_toolbar)
        layout.addWidget(self.spectrogram_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Spectrogram")

    def setup_tab5(self):
        tab = QWidget()
        layout = QVBoxLayout()

        fundamental_freq_label = QLabel("Fundamental Frequency Over Time:")
        layout.addWidget(fundamental_freq_label)

        fundamental_freq_figure = Figure(figsize=(10, 6))
        fundamental_freq_canvas = FigureCanvas(fundamental_freq_figure)
        fundamental_freq_toolbar = NavigationToolbar(fundamental_freq_canvas, self)
        layout.addWidget(fundamental_freq_toolbar)
        layout.addWidget(fundamental_freq_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Fundamental Frequency")

        # Store references to canvas and figures for later updates
        self.fundamental_freq_figure = fundamental_freq_figure
        self.fundamental_freq_canvas = fundamental_freq_canvas

    def on_time_domain_click(self, event):
        """Handle click event in time domain plot to start selection"""
        if self.audio_data is None or event.inaxes is None:
            return

        # TODO: Implement region selection like in Audacity
        # This would involve tracking the start point on mouse down
        # and updating the selection on mouse move and mouse up
        print(f"Click at x={event.xdata}, y={event.ydata}")

    def on_window_function_click(self, event):
        """Handle click event in window function plot"""
        if self.audio_data is None or event.inaxes is None:
            return

        # Similar to time domain selection
        print(f"Window function plot click at x={event.xdata}, y={event.ydata}")

    def copy_selected_fragment(self):
        """Copy the selected fragment when the button is clicked"""
        if self.selected_region is None:
            QMessageBox.information(self, "No Selection",
                                    "Please select a region first by clicking and dragging on the plot.")
            return

        start_idx, end_idx = self.selected_region
        start_time = start_idx / self.sample_rate
        end_time = end_idx / self.sample_rate
        duration = end_time - start_time

        QMessageBox.information(
            self,
            "Selection Info",
            f"Selected region from {start_time:.3f}s to {end_time:.3f}s\n"
            f"Duration: {duration:.3f}s\n"
            f"Samples: {start_idx} to {end_idx} ({end_idx - start_idx} samples)"
        )

    def update_windowed_plot(self, window_type):
        """Update the windowed signal plots based on the selected window function"""
        if self.audio_data is None:
            return

        # Even if no region is selected, we can show the entire signal with windowing
        if self.selected_region is None:
            # Use the entire signal
            start_idx = 0
            end_idx = len(self.audio_data)
        else:
            # Use the selected region
            start_idx, end_idx = self.selected_region

        # Get the data to be windowed
        data_to_window = self.audio_data[start_idx:end_idx]

        # Create window function
        n = len(data_to_window)
        if n > 0:
            if window_type == "rectangular":
                window = np.ones(n)
            elif window_type == "triangular":
                window = np.bartlett(n)
            elif window_type == "hamming":
                window = np.hamming(n)
            elif window_type == "hanning":
                window = np.hanning(n)
            elif window_type == "blackman":
                window = np.blackman(n)
            else:
                window = np.ones(n)  # Default to rectangular

            # Apply window function
            windowed_data = data_to_window * window

            # Update time domain windowed plot
            self.windowed_figure.clear()
            ax = self.windowed_figure.add_subplot(111)

            # Plot time domain signal
            time = np.arange(n) / self.sample_rate
            ax.plot(time, windowed_data)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Windowed Signal ({window_type})')

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            # Set y-axis limits with some padding
            max_amp = np.max(np.abs(windowed_data))
            if max_amp > 0:  # Avoid division by zero
                ax.set_ylim([-max_amp * 1.1, max_amp * 1.1])

            self.windowed_figure.tight_layout()
            self.windowed_canvas.draw()

            # Update frequency domain windowed plot
            self.freq_windowed_figure.clear()
            freq_ax = self.freq_windowed_figure.add_subplot(111)

            # Compute FFT of the windowed signal
            yf = np.fft.rfft(windowed_data)
            xf = np.fft.rfftfreq(n, 1 / self.sample_rate)

            # Plot magnitude spectrum in dB
            magnitude = np.abs(yf)
            if np.max(magnitude) > 0:  # Avoid log of zero or division by zero
                magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-10)

                freq_ax.plot(xf, magnitude_db)
                freq_ax.set_xlabel('Frequency (Hz)')
                freq_ax.set_ylabel('Magnitude (dB)')
                freq_ax.set_title(f'Frequency Response ({window_type})')

                # Add grid
                freq_ax.grid(True, linestyle='--', alpha=0.7)

                # Set axis limits
                freq_ax.set_xlim([0, min(self.sample_rate / 2, 5000)])  # Limit to 5kHz for better visibility
                freq_ax.set_ylim([-80, 0])

            self.freq_windowed_figure.tight_layout()
            self.freq_windowed_canvas.draw()

            # Update status bar
            if self.selected_region is None:
                self.statusBar().showMessage(f"Applied {window_type} window to entire signal")
            else:
                selection_duration = (end_idx - start_idx) / self.sample_rate
                self.statusBar().showMessage(
                    f"Applied {window_type} window to selection ({start_idx} to {end_idx}, "
                    f"duration: {selection_duration:.3f}s)"
                )

    def on_frame_dur_changed(self, value):
        """Handle frame duration slider change"""
        # Convert slider value (ms) to seconds for the parameter
        self.spec_frame_dur = value / 1000.0
        self.frame_dur_value_label.setText(f"{value} ms")
        self.update_spectrogram()

    def on_overlap_changed(self, value):
        """Handle overlap slider change"""
        # Convert slider value (0-99) to decimal (0.0-0.99)
        self.spec_overlap = value / 100.0
        self.overlap_value_label.setText(f"{self.spec_overlap:.2f}")
        self.update_spectrogram()

    def on_spectrogram_param_changed(self, window_type):
        """Handle window type change for spectrogram"""
        self.spec_window = window_type
        self.update_spectrogram()

    def on_max_freq_changed(self, max_freq):
        self.max_spec_freq = max_freq
        self.max_freq_value_label.setText(f"{max_freq}Hz")
        self.update_spectrogram()

    def update_spectrogram(self):
        """Update the spectrogram plot"""
        # Only proceed if we have audio data
        if self.audio_data is None or self.sample_rate is None:
            return

        try:
            # Call the imported plot_spectrogram_in_figure function with current parameters
            plot_spectrogram(
                self.spectrogram_figure,
                self.sample_rate,
                self.audio_data,
                overlap=self.spec_overlap,
                min_frame_dur=self.spec_frame_dur,
                window=self.spec_window,
                max_freq=self.max_spec_freq
            )
            # Update canvas
            self.spectrogram_canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "Spectrogram Error", f"Error updating spectrogram: {str(e)}")
            print(f"Error updating spectrogram: {e}")

    def update_all_plots(self):
        """Update all plots in the application"""
        if self.audio_data is None:
            return

        # Update time domain plots (Tab 1)
        self.update_time_domain_plots()

        # Update frequency domain plots (Tab 2)
        self.update_frequency_domain_plots()

        # Update window function plots (Tab 3)
        self.update_window_function_plots()

        # Update spectrogram (Tab 4)
        self.update_spectrogram()

        # Update fundamental frequency plot (Tab 5)
        self.update_fundamental_frequency_plot()

    def update_time_domain_plots(self):
        """Update the time domain plots in tab 1"""
        if self.audio_data is None or self.sample_rate is None:
            return

        # Clear the figure
        self.time_domain_main_figure.clear()
        ax = self.time_domain_main_figure.add_subplot(111)

        # Plot time domain signal
        time = np.arange(len(self.audio_data)) / self.sample_rate
        ax.plot(time, self.audio_data)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Time Domain Signal')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set y-axis limits with some padding
        max_amp = np.max(np.abs(self.audio_data))
        ax.set_ylim([-max_amp * 1.1, max_amp * 1.1])

        self.time_domain_main_figure.tight_layout()
        self.time_domain_main_canvas.draw()

        # Update subsection plot if we have a selection
        if self.selected_region is not None:
            self.update_time_domain_subsection_plot()


        # Set up span selector again to ensure it's connected to the new plot
        self.setup_span_selectors()


    def update_time_domain_subsection_plot(self):
        """Update the subsection plot in time domain tab"""
        if self.audio_data is None or self.selected_region is None:
            return

        start_idx, end_idx = self.selected_region

        # Clear the figure
        self.time_domain_subsection_figure.clear()
        ax = self.time_domain_subsection_figure.add_subplot(111)

        # Get the selected portion of the signal
        selected_data = self.audio_data[start_idx:end_idx]
        time = np.arange(len(selected_data)) / self.sample_rate

        # Plot time domain signal
        ax.plot(time, selected_data)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Selected Region')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set y-axis limits with some padding
        max_amp = np.max(np.abs(selected_data))
        ax.set_ylim([-max_amp * 1.1, max_amp * 1.1])

        self.time_domain_subsection_figure.tight_layout()
        self.time_domain_subsection_canvas.draw()

    def update_frequency_domain_plots(self):
        """Update the acoustic features in tab 2"""
        if self.audio_data is None:
            return

        # Only calculate if the tab is visible to improve performance
        if self.tabs.currentIndex() == 1:  # Tab 2 (Acoustic Features)
            self.update_acoustic_features()
        else:
            # Just clear the flag so it will update when the tab becomes visible
            pass

    

    def update_window_function_plots(self):
        """Update the plots in the window function tab"""
        if self.audio_data is None or self.sample_rate is None:
            return

        # Update main plot (same as time domain)
        self.window_function_main_figure.clear()
        ax = self.window_function_main_figure.add_subplot(111)

        time = np.arange(len(self.audio_data)) / self.sample_rate
        ax.plot(time, self.audio_data)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Time Domain Signal (for windowing)')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set y-axis limits with some padding
        max_amp = np.max(np.abs(self.audio_data))
        ax.set_ylim([-max_amp * 1.1, max_amp * 1.1])

        self.window_function_main_figure.tight_layout()
        self.window_function_main_canvas.draw()

        # Update windowed plots with current window type
        window_type = self.window_combo.currentText()
        self.update_windowed_plot(window_type)

    def update_windowed_subsection_plot(self):
        """Update the windowed subsection plot"""
        if self.audio_data is None:
            return

        # Get current window type
        window_type = self.window_combo.currentText()

        # Call the main update method
        self.update_windowed_plot(window_type)

    def update_fundamental_frequency_plot(self):
        """Update the fundamental frequency plot"""
        # Only proceed if we have audio data
        if self.audio_data is None or self.sample_rate is None:
            return

        try:
            # Call the imported plot_spectrogram_in_figure function with current parameters
            plot_f0_from_cepstrum(
                self.fundamental_freq_figure,
                self.audio_data,
                self.sample_rate
            )
            # Update canvas
            self.fundamental_freq_canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "F0 plot Error", f"Error updating f0_plot: {str(e)}")
            print(f"Error updating f0 plot: {e}")

    def setup_span_selectors(self):
        """Set up span selectors for interactive region selection"""
        # Get the current axes from the figures
        time_ax = self.time_domain_main_figure.gca()
        window_ax = self.window_function_main_figure.gca()

        # First tab span selector
        if time_ax is not None:
            # If a previous span selector exists, disconnect it
            if hasattr(self, 'time_domain_span_selector') and self.time_domain_span_selector is not None:
                self.time_domain_span_selector.disconnect_events()

            # Create a new span selector with clear visual feedback
            self.time_domain_span_selector = SpanSelector(
                time_ax,
                self.on_time_domain_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='blue'),
                interactive=True,
                drag_from_anywhere=True,
                button=1  # Left mouse button
            )
            # Let the user know they can now select
            self.statusBar().showMessage("Click and drag to select a region on the plot", 3000)

        # Window function tab span selector
        if window_ax is not None:
            # If a previous span selector exists, disconnect it
            if hasattr(self, 'window_function_span_selector') and self.window_function_span_selector is not None:
                self.window_function_span_selector.disconnect_events()

            # Create new span selector
            self.window_function_span_selector = SpanSelector(
                window_ax,
                self.on_window_function_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='green'),
                interactive=True,
                drag_from_anywhere=True,
                button=1  # Left mouse button
            )

        # Add selection callback methods

    # Add selection callback methods
    def on_time_domain_select(self, xmin, xmax):
        """Handle selection in time domain plot"""
        if self.audio_data is None or self.sample_rate is None:
            return

        # Convert time values to sample indices
        start_idx = max(0, int(xmin * self.sample_rate))
        end_idx = min(len(self.audio_data) - 1, int(xmax * self.sample_rate))

        if start_idx >= end_idx:
            return  # Invalid selection

        # Store the selected region
        self.selected_region = (start_idx, end_idx)

        # Update the subsection plots
        self.update_time_domain_subsection_plot()


        # Update status bar with selection info
        selection_duration = (end_idx - start_idx) / self.sample_rate
        self.statusBar().showMessage(
            f"Selected region: {xmin:.3f}s to {xmax:.3f}s (Duration: {selection_duration:.3f}s)")

        # If we're in the window function tab, also update that plot
        current_tab = self.tabs.currentIndex()
        if current_tab == 2:  # Window Function tab
            # Get current window type and update the windowed plots
            window_type = self.window_combo.currentText()
            self.update_windowed_plot(window_type)

    def on_window_function_select(self, xmin, xmax):
        """Handle selection in window function plot"""
        # Call the same handler as time domain since the functionality is identical
        self.on_time_domain_select(xmin, xmax)

    def on_acoustic_frame_size_changed(self, value):
        """Handle changes to the frame size slider"""
        self.acoustic_frame_size_ms = value
        self.frame_size_value_label.setText(f"{value} ms")
        # No automatic update to avoid performance issues with large files

    def on_acoustic_hop_size_changed(self, value):
        """Handle changes to the hop size slider"""
        self.acoustic_hop_size_percent = value
        self.hop_size_value_label.setText(f"{value}%")
        # No automatic update to avoid performance issues with large files

    def update_acoustic_features(self):
        """Update all acoustic feature plots based on current settings"""
        if self.audio_data is None or self.sample_rate is None:
            QMessageBox.warning(self, "No Data", "Please load an audio file first.")
            return

        try:
            # Calculate frame size in samples
            frame_size_samples = int((self.acoustic_frame_size_ms / 1000.0) * self.sample_rate)

            # Calculate hop size in samples
            hop_size_samples = int(frame_size_samples * (self.acoustic_hop_size_percent / 100.0))

            # Update status
            self.statusBar().showMessage(
                f"Calculating acoustic features (Frame: {self.acoustic_frame_size_ms}ms, Hop: {self.acoustic_hop_size_percent}%)...")

            # Update each feature plot
            self.update_volume_plot(frame_size_samples, hop_size_samples)
            self.update_frequency_centroid_plot(frame_size_samples, hop_size_samples)
            self.update_bandwidth_plot(frame_size_samples, hop_size_samples)
            self.update_band_energy_ratio_plot(frame_size_samples, hop_size_samples)
            self.update_spectral_flatness_plot(frame_size_samples, hop_size_samples)
            self.update_spectral_crest_plot(frame_size_samples, hop_size_samples)

            # Update status
            self.statusBar().showMessage("Acoustic features analysis complete.")

        except Exception as e:
            QMessageBox.warning(self, "Analysis Error", f"Error calculating acoustic features: {str(e)}")
            print(f"Error in acoustic features analysis: {e}")

    # Add supporting methods for the acoustic features tab
    def on_acoustic_frame_size_changed(self, value):
        """Handle changes to the frame size slider"""
        self.acoustic_frame_size_ms = value
        self.frame_size_value_label.setText(f"{value} ms")
        # No automatic update to avoid performance issues with large files

    def on_acoustic_hop_size_changed(self, value):
        """Handle changes to the hop size slider"""
        self.acoustic_hop_size_percent = value
        self.hop_size_value_label.setText(f"{value}%")
        # No automatic update to avoid performance issues with large files

    # Methods for updating individual feature plots
    def update_volume_plot(self, frame_size, hop_size):
        """Update the volume plot"""
        if self.audio_data is None:
            return

        # Clear the figure
        self.acoustic_figures['volume'].clear()
        ax = self.acoustic_figures['volume'].add_subplot(111)

        # Here you would calculate and plot the volume
        # Instead, we'll just set up the axes
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Volume (dB)')
        ax.set_title('Volume')
        ax.grid(True, linestyle='--', alpha=0.7)

        # YOU WILL IMPLEMENT THE ACTUAL PLOTTING HERE

        self.acoustic_figures['volume'].tight_layout()
        self.acoustic_canvases['volume'].draw()

    def update_frequency_centroid_plot(self, frame_size, hop_size):
        """Update the frequency centroid plot"""
        if self.audio_data is None:
            return

        # Clear the figure
        self.acoustic_figures['fc'].clear()
        ax = self.acoustic_figures['fc'].add_subplot(111)

        # Here you would calculate and plot the frequency centroid
        # Instead, we'll just set up the axes
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Frequency Centroid')
        ax.grid(True, linestyle='--', alpha=0.7)

        # YOU WILL IMPLEMENT THE ACTUAL PLOTTING HERE

        self.acoustic_figures['fc'].tight_layout()
        self.acoustic_canvases['fc'].draw()

    def update_bandwidth_plot(self, frame_size, hop_size):
        """Update the effective bandwidth plot"""
        if self.audio_data is None:
            return

        # Clear the figure
        self.acoustic_figures['bw'].clear()
        ax = self.acoustic_figures['bw'].add_subplot(111)

        # Here you would calculate and plot the bandwidth
        # Instead, we'll just set up the axes
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Bandwidth (Hz)')
        ax.set_title('Effective Bandwidth')
        ax.grid(True, linestyle='--', alpha=0.7)

        # YOU WILL IMPLEMENT THE ACTUAL PLOTTING HERE

        self.acoustic_figures['bw'].tight_layout()
        self.acoustic_canvases['bw'].draw()

    def update_band_energy_ratio_plot(self, frame_size, hop_size):
        """Update the band energy ratio plot"""
        if self.audio_data is None:
            return

        # Clear the figure
        self.acoustic_figures['ber'].clear()
        ax = self.acoustic_figures['ber'].add_subplot(111)

        # Here you would calculate and plot the band energy ratio
        # Instead, we'll just set up the axes
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('BER')
        ax.set_title('Band Energy Ratio')
        ax.grid(True, linestyle='--', alpha=0.7)

        # YOU WILL IMPLEMENT THE ACTUAL PLOTTING HERE

        self.acoustic_figures['ber'].tight_layout()
        self.acoustic_canvases['ber'].draw()

    def update_spectral_flatness_plot(self, frame_size, hop_size):
        """Update the spectral flatness measure plot"""
        if self.audio_data is None:
            return

        # Clear the figure
        self.acoustic_figures['sfm'].clear()
        ax = self.acoustic_figures['sfm'].add_subplot(111)

        # Here you would calculate and plot the spectral flatness
        # Instead, we'll just set up the axes
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SFM')
        ax.set_title('Spectral Flatness Measure')
        ax.grid(True, linestyle='--', alpha=0.7)

        # YOU WILL IMPLEMENT THE ACTUAL PLOTTING HERE

        self.acoustic_figures['sfm'].tight_layout()
        self.acoustic_canvases['sfm'].draw()

    def update_spectral_crest_plot(self, frame_size, hop_size):
        """Update the spectral crest factor plot"""
        if self.audio_data is None:
            return

        # Clear the figure
        self.acoustic_figures['scf'].clear()
        ax = self.acoustic_figures['scf'].add_subplot(111)

        # Here you would calculate and plot the spectral crest factor
        # Instead, we'll just set up the axes
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SCF')
        ax.set_title('Spectral Crest Factor')
        ax.grid(True, linestyle='--', alpha=0.7)

        # YOU WILL IMPLEMENT THE ACTUAL PLOTTING HERE

        self.acoustic_figures['scf'].tight_layout()
        self.acoustic_canvases['scf'].draw()

    def handle_tab_changed(self, index):
        """Handle tab change events"""
        if index == 1 and self.audio_data is not None:  # Tab 2 (Acoustic Features)
            # Update acoustic features when switching to this tab
            self.update_acoustic_features()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioAnalyzerApp()
    sys.exit(app.exec_())