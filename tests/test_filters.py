import numpy as np
import pytest

from neural_ecg.filters import (
    five_point_derivative,
    moving_window_integration,
    zero_phase_bandpass,
)

FS = 360.0


def tone(frequency_hz: float, seconds: float = 20.0) -> np.ndarray:
    time_s = np.arange(round(seconds * FS)) / FS
    return np.sin(2 * np.pi * frequency_hz * time_s)


def steady_gain(output: np.ndarray, reference: np.ndarray) -> float:
    middle = slice(output.size // 4, 3 * output.size // 4)
    return float(np.std(output[middle]) / np.std(reference[middle]))


def test_qrs_band_passes_10_hz_and_rejects_wander_and_mains() -> None:
    assert steady_gain(zero_phase_bandpass(tone(10.0), FS, 5.0, 15.0), tone(10.0)) > 0.9
    assert steady_gain(zero_phase_bandpass(tone(0.3), FS, 5.0, 15.0), tone(0.3)) < 0.01
    assert steady_gain(zero_phase_bandpass(tone(60.0), FS, 5.0, 15.0), tone(60.0)) < 0.05


def test_zero_phase_filter_does_not_shift_a_pulse() -> None:
    pulse = np.zeros(3600)
    pulse[1800] = 1.0
    filtered = zero_phase_bandpass(pulse, FS, 0.5, 40.0)
    assert int(np.argmax(filtered)) == 1800


def test_invalid_band_is_rejected() -> None:
    with pytest.raises(ValueError):
        zero_phase_bandpass(tone(10.0), FS, 20.0, 10.0)
    with pytest.raises(ValueError):
        zero_phase_bandpass(tone(10.0), FS, 5.0, 200.0)


def test_five_point_derivative_of_a_ramp_is_its_slope() -> None:
    ramp = 0.5 * np.arange(100) / FS
    derivative = five_point_derivative(ramp, FS)
    assert derivative.shape == ramp.shape
    np.testing.assert_allclose(derivative[2:-2], 0.5, rtol=1e-12)


def test_moving_window_integration_preserves_a_constant() -> None:
    integrated = moving_window_integration(np.ones(200), 54)
    np.testing.assert_allclose(integrated[60:-60], 1.0)
    with pytest.raises(ValueError):
        moving_window_integration(np.ones(10), 0)
