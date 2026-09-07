import numpy as np
import pywt
from collections import deque
from .filters import AdaptiveNotchFilter, KalmanMouseFilter

class TremorProcessor:
    def __init__(self, fs=100, init_freq=6.0):
        self.fs = fs
        self.init_freq = init_freq
        self.anf_x = AdaptiveNotchFilter(fs, init_freq)
        self.anf_y = AdaptiveNotchFilter(fs, init_freq)
        self.kf_x = KalmanMouseFilter(process_noise=0.01, measurement_noise=0.5)
        self.kf_y = KalmanMouseFilter(process_noise=0.01, measurement_noise=0.5)
        self.wavelet_buffer_x = deque(maxlen=128)
        self.wavelet_buffer_y = deque(maxlen=128)
        self.use_wavelet = True

        self.button_pressed = False

        self.prev_speed = 0

    def process(self, x, y, button_pressed=False):
        # 1. 自适应陷波
        x_anf = self.anf_x.process(x, adapt=True)
        y_anf = self.anf_y.process(y, adapt=True)
 
        if self.use_wavelet and len(self.wavelet_buffer_x) == self.wavelet_buffer_x.maxlen:
            x_wav = self._wavelet_denoise(self.wavelet_buffer_x)
            y_wav = self._wavelet_denoise(self.wavelet_buffer_y)
            x_filt = 0.5 * x_anf + 0.5 * x_wav[-1]
            y_filt = 0.5 * y_anf + 0.5 * y_wav[-1]
        else:
            x_filt = x_anf
            y_filt = y_anf

        # 更新缓冲
        self.wavelet_buffer_x.append(x_filt)
        self.wavelet_buffer_y.append(y_filt)

        # 3. 卡尔曼平滑
        if self.button_pressed:
            self.kf_x.Q[0,0] = 0.001
            self.kf_y.Q[0,0] = 0.001
        else:
            self.kf_x.Q[0,0] = 0.01
            self.kf_y.Q[0,0] = 0.01

        x_kf = self.kf_x.update(x_filt)
        y_kf = self.kf_y.update(y_filt)

        return x_kf, y_kf

    def _wavelet_denoise(self, buffer):
        data = np.array(buffer)
        coeffs = pywt.wavedec(data, 'db4', level=3)
        # 估计噪声标准差（使用最细细节系数）
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        # 软阈值处理细节系数
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
        denoised = pywt.waverec(new_coeffs, 'db4')
        return denoised[:len(data)]