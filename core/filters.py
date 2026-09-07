import numpy as np


class KalmanMouseFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.1, initial_position=0.0):
        self.dt = 1.0
        self.A = np.array([[1, self.dt],
                           [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * process_noise
        self.R = np.eye(1) * measurement_noise
        self.P = np.eye(2) * 0.1
        self.x = np.array([[initial_position],
                           [0.0]])

    def update(self, measurement, dt=None):
        if dt is not None:
            self.dt = dt
            self.A[0, 1] = dt

        z = np.array([[measurement]])

        # 预测
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        # 更新
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        return self.x[0, 0]


class AdaptiveNotchFilter:
    def __init__(self, fs, init_freq=6.0, mu=0.005, notch_width=2.0):
        self.fs = fs
        self.freq = init_freq
        self.mu = mu
        self.notch_width = notch_width
        self.adapt = True
        self.update_coefficients()

        self._state = np.zeros(2)

    def update_coefficients(self):
        w0 = 2 * np.pi * self.freq / self.fs
        r = 1 - (self.notch_width / self.fs) * np.pi
        # 陷波器系数（零点在单位圆上，极点在零点内侧）
        self.b = np.array([1, -2*np.cos(w0), 1])
        self.a = np.array([1, -2*r*np.cos(w0), r**2])

    def process(self, x, adapt=True):

        self.adapt = adapt

        y = self.b[0] * x + self._state[0]
        self._state[0] = self.b[1] * x - self.a[1] * y + self._state[1]
        self._state[1] = self.b[2] * x - self.a[2] * y

        if adapt and self.adapt:
            # 误差信号
            error = x - y

            # 数值梯度更新频率
            delta = 0.01
            original_freq = self.freq
            original_state = self._state.copy()

            # 尝试增加频率
            self.freq = original_freq + delta
            self.update_coefficients()
            y_plus = self.b[0] * x + original_state[0]
            error_plus = x - y_plus

            # 尝试减少频率
            self.freq = original_freq - delta
            self.update_coefficients()
            y_minus = self.b[0] * x + original_state[0]
            error_minus = x - y_minus

            # 梯度估计（中心差分）
            grad = (error_plus**2 - error_minus**2) / (2 * delta)

            # 更新频率
            self.freq = original_freq - self.mu * grad

            # 限制频率范围
            self.freq = np.clip(self.freq, 3.0, 12.0)

            # 恢复系数
            self.update_coefficients()
            self._state = original_state

        return y

    def reset(self):
        self._state = np.zeros(2)