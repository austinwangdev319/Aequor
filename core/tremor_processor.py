import numpy as np
from core.filters import KalmanMouseFilter, AdaptiveNotchFilter
from core.online_wavelet import OnlineWaveletDenoiser


class TremorProcessor:
    def __init__(self, fs=200, init_freq=6.0):
        self.fs = fs
        self.init_freq = init_freq

        # 自适应陷波器
        self.anf_x = AdaptiveNotchFilter(fs, init_freq, mu=0.005, notch_width=2.0)
        self.anf_y = AdaptiveNotchFilter(fs, init_freq, mu=0.005, notch_width=2.0)

        # 卡尔曼滤波器
        self.kf_x = KalmanMouseFilter(process_noise=0.01, measurement_noise=2.0)
        self.kf_y = KalmanMouseFilter(process_noise=0.01, measurement_noise=2.0)

        # 小波去噪器
        self.wavelet_x = OnlineWaveletDenoiser(buffer_size=64, wavelet='db4', level=3, threshold_scale=1.0)
        self.wavelet_y = OnlineWaveletDenoiser(buffer_size=64, wavelet='db4', level=3, threshold_scale=1.0)

        # 点击状态
        self.button_pressed = False

        # 初始位置
        self._initialized = False
        self._last_x = 0
        self._last_y = 0

    def process(self, x, y, button_pressed=False):
        self.button_pressed = button_pressed

        # 第一步：自适应陷波
        x_notch = self.anf_x.process(x, adapt=True)
        y_notch = self.anf_y.process(y, adapt=True)

        # 第二步：小波去噪
        x_wavelet = self.wavelet_x.update(x_notch)
        y_wavelet = self.wavelet_y.update(y_notch)

        # 融合：陷波结果和小波结果平均
        x_fused = 0.5 * x_notch + 0.5 * x_wavelet
        y_fused = 0.5 * y_notch + 0.5 * y_wavelet

        # 第三步：卡尔曼滤波
        # 按钮按下时增强滤波（降低过程噪声）
        if self.button_pressed:
            self.kf_x.Q[0, 0] = 0.001
            self.kf_y.Q[0, 0] = 0.001
        else:
            self.kf_x.Q[0, 0] = 0.01
            self.kf_y.Q[0, 0] = 0.01

        x_kf = self.kf_x.update(x_fused)
        y_kf = self.kf_y.update(y_fused)

        # 初始化处理
        if not self._initialized:
            self._initialized = True
            self._last_x = x_kf
            self._last_y = y_kf

        return x_kf, y_kf

    def reset(self):
        self.anf_x.reset()
        self.anf_y.reset()
        self.wavelet_x.reset()
        self.wavelet_y.reset()
        self._initialized = False