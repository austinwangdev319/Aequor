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
        self.last_time = None

    def update(self, measurement, dt=None):
        if dt is not None:
            self.dt = dt
            self.A[0, 1] = dt

        z = np.array([[measurement]])

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

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
        self.update_coefficients()

    def update_coefficients(self):
        w0 = 2 * np.pi * self.freq / self.fs
        r = 1 - (self.notch_width / self.fs) * np.pi  # 极点半径，控制带宽
        self.b = np.array([1, -2*np.cos(w0), 1])
        self.a = np.array([1, -2*r*np.cos(w0), r**2])

    def process(self, x, adapt=True):
        y = self._filter_sample(x)

        if adapt:
            error = x - y

            delta = 0.01  # Hz
            original_freq = self.freq
            self.freq += delta
            self.update_coefficients()
            y_new = self._filter_sample(x)
            error_new = x - y_new

            grad = (error_new**2 - error**2) / delta
            self.freq -= self.mu * grad

            self.freq = np.clip(self.freq, 3.0, 12.0)
            self.update_coefficients()

        return y

    def _filter_sample(self, x):
        from scipy.signal import lfilter
        y = lfilter(self.b, self.a, [x])[0]
        return y