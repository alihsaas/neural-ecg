import importlib.util

import numpy as np
import pytest

from neural_ecg.records import data_available

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
DATA_AVAILABLE = data_available()

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")
requires_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="MIT-BIH data is not in data/mitdb")


def synthetic_ecg(
    beat_times_s: np.ndarray,
    duration_s: float,
    sampling_rate_hz: float = 360.0,
    qrs_width_s: float = 0.012,
    qrs_amplitude_mv: float = 1.2,
    t_wave_amplitude_mv: float = 0.3,
) -> np.ndarray:
    time_s = np.arange(round(duration_s * sampling_rate_hz)) / sampling_rate_hz
    signal = np.zeros_like(time_s)
    for beat in beat_times_s:
        signal += qrs_amplitude_mv * np.exp(-0.5 * ((time_s - beat) / qrs_width_s) ** 2)
        signal -= 0.25 * qrs_amplitude_mv * np.exp(-0.5 * ((time_s - beat - 0.03) / 0.008) ** 2)
        signal += t_wave_amplitude_mv * np.exp(-0.5 * ((time_s - beat - 0.25) / 0.04) ** 2)
    return signal


@pytest.fixture
def irregular_beats() -> np.ndarray:
    rng = np.random.default_rng(7)
    intervals = rng.uniform(0.55, 1.1, size=40)
    return 1.0 + np.cumsum(intervals)
