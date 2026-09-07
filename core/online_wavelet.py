import numpy as np
import pywt
from collections import deque


class OnlineWaveletDenoiser:
    def __init__(self, buffer_size=128, wavelet='db4', level=3, threshold_scale=1.0):
        self.buffer_size = buffer_size
        self.wavelet = wavelet
        self.level = level
        self.threshold_scale = threshold_scale
        self.buffer = deque(maxlen=buffer_size)

    def update(self, sample):
        self.buffer.append(sample)

        if len(self.buffer) < self.buffer_size:
            return sample

        denoised = self._denoise_buffer()
        return denoised[-1]

    def _denoise_buffer(self):
        data = np.array(self.buffer)

        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

        # 噪声标准差估计
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data))) * self.threshold_scale

        # 软阈值处理
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))

        denoised = pywt.waverec(new_coeffs, self.wavelet)

        # 确保长度一致
        if len(denoised) > len(data):
            denoised = denoised[:len(data)]
        elif len(denoised) < len(data):
            denoised = np.pad(denoised, (0, len(data) - len(denoised)))

        return denoised

    def reset(self):
        self.buffer.clear()