import sys
import time
import numpy as np
import pyqtgraph as pg  
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QCheckBox
from PyQt5.QtCore import QTimer, Qt
from pynput import mouse
from core.tremor_processor import TremorProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SteadyCursor")
        self.processor = TremorProcessor()
        self.apply_to_cursor = False
        self.listener = None
        self.mouse_controller = mouse.Controller()

        self.start_btn = QPushButton("Start Filtering")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.toggle_filtering)

        self.apply_check = QCheckBox("Apply to system cursor")
        self.apply_check.toggled.connect(self.toggle_apply)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.valueChanged.connect(self.adjust_sensitivity)

        self.status_label = QLabel("Status: Stopped")

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.trajectory_plot = self.plot_widget.addPlot(title="Cursor Position (X)")
        self.raw_curve = self.trajectory_plot.plot(pen='r', name='Raw')
        self.filt_curve = self.trajectory_plot.plot(pen='g', name='Filtered')
        self.spectrum_plot = self.plot_widget.addPlot(title="Spectrum (X)")
        self.spectrum_curve = self.spectrum_plot.plot(pen='b')

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.apply_check)
        controls_layout.addWidget(QLabel("Sensitivity:"))
        controls_layout.addWidget(self.sensitivity_slider)
        controls_layout.addWidget(self.status_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.plot_widget)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.history_size = 200
        self.raw_x_history = np.zeros(self.history_size)
        self.filt_x_history = np.zeros(self.history_size)
        self.time_axis = np.arange(self.history_size)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(50)  # 20fps

        self.last_raw_x = 0
        self.last_filt_x = 0

    def toggle_filtering(self, checked):
        if checked:
            self.start_btn.setText("Stop Filtering")
            self.status_label.setText("Status: Running")
            self.start_listener()
        else:
            self.start_btn.setText("Start Filtering")
            self.status_label.setText("Status: Stopped")
            self.stop_listener()

    def toggle_apply(self, checked):
        self.apply_to_cursor = checked

    def adjust_sensitivity(self, value):
        # 调整卡尔曼测量噪声
        factor = value / 50.0
        self.processor.kf_x.R[0,0] = 0.1 * factor
        self.processor.kf_y.R[0,0] = 0.1 * factor

    def start_listener(self):
        self.listener = mouse.Listener(on_move=self.on_mouse_move, on_click=self.on_mouse_click)
        self.listener.start()

    def stop_listener(self):
        if self.listener:
            self.listener.stop()
            self.listener = None

    def on_mouse_move(self, x, y):
        filt_x, filt_y = self.processor.process(x, y, self.processor.button_pressed)

        if self.apply_to_cursor:
            self.mouse_controller.position = (int(filt_x), int(filt_y))

        self.last_raw_x = x
        self.last_filt_x = filt_x

    def on_mouse_click(self, x, y, button, pressed):
        self.processor.button_pressed = pressed

    def update_ui(self):
        self.trajectory_plot.setXRange(0, self.history_size, padding=0)
        self.trajectory_plot.setYRange(0, 1920)
        self.raw_curve.setData(self.time_axis, self.raw_x_history)
        self.filt_curve.setData(self.time_axis, self.filt_x_history)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())