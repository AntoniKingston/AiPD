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

# Import functions from separate file
from functions import *


class AudioAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Signal Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.audio_data = None
        self.selected_region = None
        self.sample_rate = None
        self.file_path = None  # Store the current file path

        # Default spectrogram parameters
        self.spec_window = 'rectangular'
        self.spec_frame_dur = 0.05  # Default value in seconds
        self.spec_overlap = 0.5  # Default overlap value
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

        # Connect the selection event to the main canvas
        self.time_domain_selection_span = None
        self.time_domain_main_canvas.mpl_connect('button_press_event', self.on_time_domain_click)

    def setup_tab2(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Frequency domain plot of full signal
        freq_label = QLabel("Frequency Domain:")
        layout.addWidget(freq_label)

        freq_figure = Figure(figsize=(10, 4))
        freq_canvas = FigureCanvas(freq_figure)
        freq_toolbar = NavigationToolbar(freq_canvas, self)
        layout.addWidget(freq_toolbar)
        layout.addWidget(freq_canvas)

        # Frequency domain plot of selected subsection
        subsection_freq_label = QLabel("Selected Fragment (Frequency Domain):")
        layout.addWidget(subsection_freq_label)

        subsection_freq_figure = Figure(figsize=(10, 2))
        subsection_freq_canvas = FigureCanvas(subsection_freq_figure)
        layout.addWidget(subsection_freq_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Frequency Domain")

        # Store references to canvas and figures for later updates
        self.freq_domain_figure = freq_figure
        self.freq_domain_canvas = freq_canvas
        self.freq_domain_subsection_figure = subsection_freq_figure
        self.freq_domain_subsection_canvas = subsection_freq_canvas

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

        windowed_figure = Figure(figsize=(10, 2))
        windowed_canvas = FigureCanvas(windowed_figure)
        layout.addWidget(windowed_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Window Function")

        # Store references to canvas and figures for later updates
        self.window_function_main_figure = main_figure
        self.window_function_main_canvas = main_canvas
        self.windowed_figure = windowed_figure
        self.windowed_canvas = windowed_canvas
        self.window_combo = window_combo

        # Connect the selection event to the main canvas (similar to tab 1)
        self.window_function_selection_span = None
        self.window_function_main_canvas.mpl_connect('button_press_event', self.on_window_function_click)

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
        # This function would be triggered by the "Select Frame" button
        if self.selected_region is None:
            QMessageBox.information(self, "No Selection", "Please select a region first.")
            return

        # In a real implementation, you would store the selected region
        # and update the subsection plots
        print("Selected fragment copied")

    def update_windowed_plot(self, window_type):
        """Update the windowed signal plot based on the selected window function"""
        if self.audio_data is None or self.selected_region is None:
            return

        print(f"Window function changed to: {window_type}")
        # Update the windowed plot with the selected window function

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

    def update_time_domain_subsection_plot(self):
        """Update the subsection plot in time domain tab"""
        if self.audio_data is None or self.selected_region is None:
            return

        # In a real implementation, you would plot the selected portion of the signal
        pass

    def update_frequency_domain_plots(self):
        """Update the frequency domain plots in tab 2"""
        if self.audio_data is None or self.sample_rate is None:
            return

        # Clear the figure
        self.freq_domain_figure.clear()
        ax = self.freq_domain_figure.add_subplot(111)

        # Compute FFT of the full signal
        n = len(self.audio_data)
        yf = np.fft.rfft(self.audio_data)
        xf = np.fft.rfftfreq(n, 1 / self.sample_rate)

        # Plot magnitude spectrum in dB
        magnitude = np.abs(yf)
        magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-10)

        ax.plot(xf, magnitude_db)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title('Frequency Domain')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set axis limits
        ax.set_xlim([0, self.sample_rate / 2])
        ax.set_ylim([-80, 0])

        self.freq_domain_figure.tight_layout()
        self.freq_domain_canvas.draw()

        # Update subsection plot if we have a selection
        if self.selected_region is not None:
            self.update_frequency_domain_subsection_plot()

    def update_frequency_domain_subsection_plot(self):
        """Update the subsection plot in frequency domain tab"""
        if self.audio_data is None or self.selected_region is None:
            return

        # In a real implementation, you would compute and plot the FFT of the selected portion
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

        # Update windowed plot if we have a selection
        if self.selected_region is not None:
            self.update_windowed_subsection_plot()

    def update_windowed_subsection_plot(self):
        """Update the windowed subsection plot"""
        if self.audio_data is None or self.selected_region is None:
            return

        # In a real implementation, you would apply the selected window function to the selection
        # For now, just get the current window type from the combo box
        window_type = self.window_combo.currentText()
        # And update the plot
        pass

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioAnalyzerApp()
    sys.exit(app.exec_())