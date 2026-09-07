import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QCheckBox)
from PyQt5.QtCore import QTimer, Qt
from pynput import mouse
from core.tremor_processor import TremorProcessor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aequor")  
        self.processor = TremorProcessor(fs=200) 
        self.apply_to_cursor = False
        self.listener = None
        self.mouse_controller = mouse.Controller()
        self._setting_cursor = False  # 防止反馈循环

        # UI 控件
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

        # 绘图
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.trajectory_plot = self.plot_widget.addPlot(title="Cursor Position (X)")
        self.raw_curve = self.trajectory_plot.plot(pen='r', name='Raw')
        self.filt_curve = self.trajectory_plot.plot(pen='g', name='Filtered')
        self.spectrum_plot = self.plot_widget.addPlot(title="Spectrum (X)")
        self.spectrum_curve = self.spectrum_plot.plot(pen='b')

        # 布局
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

        # 历史数据
        self.history_size = 200
        self.raw_x_history = np.zeros(self.history_size)
        self.filt_x_history = np.zeros(self.history_size)
        self.time_axis = np.arange(self.history_size)

        # 频谱数据
        self.spectrum_buffer_size = 256
        self.spectrum_buffer = np.zeros(self.spectrum_buffer_size)

        # 定时器
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(50)  # 20fps

        # 初始化滤波状态
        self.last_raw_x = 0
        self.last_filt_x = 0
        self.last_time = None

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
        if not checked:
            self._setting_cursor = False

    def adjust_sensitivity(self, value):
        """调整滤波强度：值越大，滤波越强"""
        # value: 1-100, 映射到卡尔曼 R 范围 0.5 - 20
        r_value = 0.5 + (value / 100.0) * 19.5
        self.processor.kf_x.R[0, 0] = r_value
        self.processor.kf_y.R[0, 0] = r_value

        # 同时调整小波去噪阈值
        self.processor.wavelet_x.threshold_scale = value / 50.0
        self.processor.wavelet_y.threshold_scale = value / 50.0

    def start_listener(self):
        self.listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click
        )
        self.listener.start()

    def stop_listener(self):
        if self.listener:
            self.listener.stop()
            self.listener = None

    def on_mouse_move(self, x, y):
        # 防止反馈循环：如果正在设置光标位置，忽略此事件
        if self._setting_cursor:
            return

        # 计算实际采样间隔
        current_time = time.time()
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                fs_est = 1.0 / dt
                # 动态更新处理器的采样率（限制在合理范围）
                if 50 < fs_est < 500:
                    self.processor.fs = fs_est
                    self.processor.anf_x.fs = fs_est
                    self.processor.anf_y.fs = fs_est
                    self.processor.anf_x.update_coefficients()
                    self.processor.anf_y.update_coefficients()
        self.last_time = current_time

        # 应用滤波
        filt_x, filt_y = self.processor.process(x, y, self.processor.button_pressed)

        # 更新历史数据
        self.raw_x_history = np.roll(self.raw_x_history, -1)
        self.raw_x_history[-1] = x
        self.filt_x_history = np.roll(self.filt_x_history, -1)
        self.filt_x_history[-1] = filt_x

        # 更新频谱缓冲
        self.spectrum_buffer = np.roll(self.spectrum_buffer, -1)
        self.spectrum_buffer[-1] = x - filt_x  # 误差信号（震颤成分）

        # 如果启用鼠标接管，写回滤波后的位置
        if self.apply_to_cursor:
            self._setting_cursor = True
            self.mouse_controller.position = (int(filt_x), int(filt_y))
            self._setting_cursor = False

        self.last_raw_x = x
        self.last_filt_x = filt_x

    def on_mouse_click(self, x, y, button, pressed):
        self.processor.button_pressed = pressed

    def update_ui(self):
        # 更新轨迹图
        self.trajectory_plot.setXRange(0, self.history_size, padding=0)
        self.trajectory_plot.setYRange(
            min(np.min(self.raw_x_history), np.min(self.filt_x_history)) - 10,
            max(np.max(self.raw_x_history), np.max(self.filt_x_history)) + 10
        )
        self.raw_curve.setData(self.time_axis, self.raw_x_history)
        self.filt_curve.setData(self.time_axis, self.filt_x_history)

        # 更新频谱图
        if np.any(self.spectrum_buffer != 0):
            windowed = self.spectrum_buffer * np.hanning(self.spectrum_buffer_size)
            fft = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(self.spectrum_buffer_size, d=1/100)  # 假设100Hz
            self.spectrum_curve.setData(freqs, fft)
            self.spectrum_plot.setXRange(0, 50, padding=0)
            self.spectrum_plot.setYRange(0, np.max(fft) * 1.1 if np.max(fft) > 0 else 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())