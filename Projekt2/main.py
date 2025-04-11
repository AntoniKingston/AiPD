import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, QTabWidget,
                             QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
                             QLabel, QSplitter)
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class AudioAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Signal Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.audio_data = None
        self.selected_region = None
        self.sample_rate = None

        self.init_ui()

    def init_ui(self):
        # Create main menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # Add open file action to menu
        open_action = QAction('Open Audio File (.wav)', self)
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)

        # Create tab widget for bookmarks
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs for each bookmark
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()
        self.setup_tab5()

        self.show()

    def open_audio_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "WAV Files (*.wav)")
        if filename:
            # Here we would normally load the audio file
            # Instead, we'll just print the filename for this example
            print(f"Selected file: {filename}")
            # In a real app, you would load the audio data here:
            # self.audio_data, self.sample_rate = librosa.load(filename)
            # Then update all plots

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
        window_combo.addItems(["Rectangular (None)", "Hamming", "Hanning", "Blackman", "Kaiser"])
        window_combo.currentIndexChanged.connect(self.update_windowed_plot)
        window_layout.addWidget(window_label)
        window_layout.addWidget(window_combo)
        window_layout.addStretch()

        layout.addLayout(window_layout)

        windowed_figure = Figure(figsize=(10, 2))
        windowed_canvas = FigureCanvas(windowed_figure)
        layout.addWidget(windowed_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Window Function")

    def setup_tab4(self):
        tab = QWidget()
        layout = QVBoxLayout()

        spectrogram_label = QLabel("Spectrogram:")
        layout.addWidget(spectrogram_label)

        spectrogram_figure = Figure(figsize=(10, 6))
        spectrogram_canvas = FigureCanvas(spectrogram_figure)
        spectrogram_toolbar = NavigationToolbar(spectrogram_canvas, self)
        layout.addWidget(spectrogram_toolbar)
        layout.addWidget(spectrogram_canvas)

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

    def copy_selected_fragment(self):
        # This function would be triggered by the "Select Frame" button
        # In a real implementation, it would copy the selected region from the plot
        print("Selected fragment copied")

    def update_windowed_plot(self, index):
        # This function would update the windowed signal plot based on the selected window function
        window_functions = ["Rectangular", "Hamming", "Hanning", "Blackman", "Kaiser"]
        print(f"Window function changed to: {window_functions[index]}")
        # Then update the windowed plot


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioAnalyzerApp()
    sys.exit(app.exec_())