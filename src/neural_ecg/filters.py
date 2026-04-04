import numpy as np
from scipy import signal as sps


def bandpass_sos(low_hz: float, high_hz: float, sampling_rate_hz: float, order: int) -> np.ndarray:
    nyquist = sampling_rate_hz / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError("Band edges must satisfy 0 < low < high < Nyquist")
    return sps.butter(order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate_hz, output="sos")


def zero_phase_bandpass(
    samples: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 2,
) -> np.ndarray:
    sos = bandpass_sos(low_hz, high_hz, sampling_rate_hz, order)
    return sps.sosfiltfilt(sos, np.asarray(samples, dtype=np.float64))


def five_point_derivative(samples: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 0.0, -2.0, -1.0]) * sampling_rate_hz / 8.0
    padded = np.pad(np.asarray(samples, dtype=np.float64), 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def moving_window_integration(samples: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples < 1:
        raise ValueError("Integration window must be at least one sample")
    kernel = np.full(window_samples, 1.0 / window_samples)
    return np.convolve(np.asarray(samples, dtype=np.float64), kernel, mode="same")
