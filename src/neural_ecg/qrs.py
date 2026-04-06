from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from neural_ecg.filters import (
    five_point_derivative,
    moving_window_integration,
    zero_phase_bandpass,
)


@dataclass(frozen=True)
class FrontEndSettings:
    band_low_hz: float = 5.0
    band_high_hz: float = 15.0
    band_order: int = 2
    integration_window_s: float = 0.150
    refractory_s: float = 0.200
    refine_window_s: float = 0.100


@dataclass(frozen=True)
class PanTompkinsSettings:
    front_end: FrontEndSettings = field(default_factory=FrontEndSettings)
    learning_period_s: float = 2.0
    t_wave_window_s: float = 0.360
    t_wave_slope_ratio: float = 0.5
    searchback_factor: float = 1.66
    rr_history: int = 8


@dataclass(frozen=True)
class FixedThresholdSettings:
    front_end: FrontEndSettings = field(default_factory=FrontEndSettings)
    threshold_fraction: float = 0.3
    reference_percentile: float = 99.5


@dataclass(frozen=True)
class FrontEndStages:
    bandpassed: np.ndarray
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray


def front_end(
    samples: np.ndarray, sampling_rate_hz: float, settings: FrontEndSettings
) -> FrontEndStages:
    bandpassed = zero_phase_bandpass(
        samples, sampling_rate_hz, settings.band_low_hz, settings.band_high_hz, settings.band_order
    )
    derivative = five_point_derivative(bandpassed, sampling_rate_hz)
    squared = derivative**2
    window = max(1, round(settings.integration_window_s * sampling_rate_hz))
    integrated = moving_window_integration(squared, window)
    return FrontEndStages(bandpassed, derivative, squared, integrated)


def candidate_peaks(
    integrated: np.ndarray, sampling_rate_hz: float, refractory_s: float
) -> np.ndarray:
    distance = max(1, round(refractory_s * sampling_rate_hz))
    peaks, _ = sps.find_peaks(integrated, distance=distance)
    return peaks.astype(np.int64)


def refine_to_r_peaks(
    detections: np.ndarray,
    bandpassed: np.ndarray,
    sampling_rate_hz: float,
    settings: FrontEndSettings,
) -> np.ndarray:
    if detections.size == 0:
        return detections.astype(np.int64)
    half_window = max(1, round(settings.refine_window_s * sampling_rate_hz))
    magnitude = np.abs(bandpassed)
    refined = []
    for index in detections:
        start = max(0, int(index) - half_window)
        stop = min(magnitude.size, int(index) + half_window + 1)
        refined.append(start + int(np.argmax(magnitude[start:stop])))
    refined_array = np.asarray(refined, dtype=np.int64)
    refractory = round(settings.refractory_s * sampling_rate_hz)
    kept = [refined_array[0]]
    for index in refined_array[1:]:
        if index - kept[-1] < refractory:
            if magnitude[index] > magnitude[kept[-1]]:
                kept[-1] = index
        else:
            kept.append(index)
    return np.asarray(kept, dtype=np.int64)


def max_slope(derivative: np.ndarray, center: int, half_width: int) -> float:
    start = max(0, center - half_width)
    stop = min(derivative.size, center + half_width + 1)
    return float(np.max(np.abs(derivative[start:stop])))


def detect_qrs_fixed_threshold(
    samples: np.ndarray,
    sampling_rate_hz: float,
    settings: FixedThresholdSettings | None = None,
) -> np.ndarray:
    settings = settings or FixedThresholdSettings()
    stages = front_end(samples, sampling_rate_hz, settings.front_end)
    reference = float(np.percentile(stages.integrated, settings.reference_percentile))
    if reference <= 0.0:
        return np.zeros(0, dtype=np.int64)
    distance = max(1, round(settings.front_end.refractory_s * sampling_rate_hz))
    peaks, _ = sps.find_peaks(
        stages.integrated, height=settings.threshold_fraction * reference, distance=distance
    )
    return refine_to_r_peaks(peaks, stages.bandpassed, sampling_rate_hz, settings.front_end)


def detect_qrs_pan_tompkins(
    samples: np.ndarray,
    sampling_rate_hz: float,
    settings: PanTompkinsSettings | None = None,
) -> np.ndarray:
    settings = settings or PanTompkinsSettings()
    stages = front_end(samples, sampling_rate_hz, settings.front_end)
    integrated = stages.integrated
    peaks = candidate_peaks(integrated, sampling_rate_hz, settings.front_end.refractory_s)
    if peaks.size == 0:
        return np.zeros(0, dtype=np.int64)
    learning = max(1, round(settings.learning_period_s * sampling_rate_hz))
    signal_level = 0.25 * float(np.max(integrated[:learning]))
    noise_level = 0.5 * float(np.mean(integrated[:learning]))
    refractory = round(settings.front_end.refractory_s * sampling_rate_hz)
    t_wave_window = round(settings.t_wave_window_s * sampling_rate_hz)
    slope_half_width = max(1, round(settings.front_end.integration_window_s * sampling_rate_hz / 2))
    accepted: list[int] = []
    accepted_slopes: list[float] = []
    rr_intervals: list[int] = []
    rejected: list[int] = []

    def accept(index: int, weight: float) -> None:
        nonlocal signal_level
        signal_level = weight * float(integrated[index]) + (1.0 - weight) * signal_level
        if accepted:
            rr_intervals.append(index - accepted[-1])
        accepted.append(index)
        accepted_slopes.append(max_slope(stages.derivative, index, slope_half_width))
        rejected.clear()

    for peak in peaks:
        peak = int(peak)
        threshold_high = noise_level + 0.25 * (signal_level - noise_level)
        threshold_low = 0.5 * threshold_high
        if accepted and rr_intervals:
            rr_average = float(np.mean(rr_intervals[-settings.rr_history :]))
            if peak - accepted[-1] > settings.searchback_factor * rr_average:
                eligible = [
                    index
                    for index in rejected
                    if index - accepted[-1] >= refractory and integrated[index] > threshold_low
                ]
                if eligible:
                    best = max(eligible, key=lambda index: integrated[index])
                    accept(best, 0.25)
                    threshold_high = noise_level + 0.25 * (signal_level - noise_level)
        value = float(integrated[peak])
        if accepted and peak - accepted[-1] < refractory:
            continue
        if value > threshold_high:
            if accepted and peak - accepted[-1] < t_wave_window:
                slope = max_slope(stages.derivative, peak, slope_half_width)
                if slope < settings.t_wave_slope_ratio * accepted_slopes[-1]:
                    noise_level = 0.125 * value + 0.875 * noise_level
                    continue
            accept(peak, 0.125)
        else:
            noise_level = 0.125 * value + 0.875 * noise_level
            rejected.append(peak)
    detections = np.asarray(accepted, dtype=np.int64)
    return refine_to_r_peaks(detections, stages.bandpassed, sampling_rate_hz, settings.front_end)
