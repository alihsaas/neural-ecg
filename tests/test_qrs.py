import numpy as np
import pytest
from conftest import requires_data, synthetic_ecg

from neural_ecg.detection_metrics import DetectionCounts, match_detections
from neural_ecg.noise import add_white_noise
from neural_ecg.qrs import (
    PanTompkinsSettings,
    detect_qrs_fixed_threshold,
    detect_qrs_pan_tompkins,
)
from neural_ecg.records import load_record

FS = 360.0
TOLERANCE = round(0.15 * FS)


@pytest.mark.parametrize("detector", [detect_qrs_pan_tompkins, detect_qrs_fixed_threshold])
def test_detectors_find_every_clean_synthetic_beat(detector, irregular_beats) -> None:
    signal = synthetic_ecg(irregular_beats, irregular_beats[-1] + 1.5)
    reference = np.round(irregular_beats * FS).astype(np.int64)
    counts = match_detections(reference, detector(signal, FS), TOLERANCE)
    assert counts == DetectionCounts(reference.size, 0, 0)


def test_pan_tompkins_locates_r_peaks_within_a_few_samples(irregular_beats) -> None:
    signal = synthetic_ecg(irregular_beats, irregular_beats[-1] + 1.5)
    detections = detect_qrs_pan_tompkins(signal, FS)
    reference = np.round(irregular_beats * FS).astype(np.int64)
    assert detections.size == reference.size
    assert np.max(np.abs(detections - reference)) <= 3


def test_pan_tompkins_ignores_tall_t_waves(irregular_beats) -> None:
    signal = synthetic_ecg(irregular_beats, irregular_beats[-1] + 1.5, t_wave_amplitude_mv=0.8)
    reference = np.round(irregular_beats * FS).astype(np.int64)
    counts = match_detections(reference, detect_qrs_pan_tompkins(signal, FS), TOLERANCE)
    assert counts.false_positives == 0
    assert counts.false_negatives == 0


def test_pan_tompkins_searchback_recovers_a_small_beat() -> None:
    beats = 1.0 + 0.8 * np.arange(30)
    signal = synthetic_ecg(beats, beats[-1] + 1.5)
    small_index = 20
    small = synthetic_ecg(beats[[small_index]], beats[-1] + 1.5)
    signal = signal - 0.58 * small
    reference = np.round(beats * FS).astype(np.int64)
    counts = match_detections(reference, detect_qrs_pan_tompkins(signal, FS), TOLERANCE)
    assert counts.false_negatives == 0
    without_searchback = PanTompkinsSettings(searchback_factor=1e9)
    missed = match_detections(
        reference, detect_qrs_pan_tompkins(signal, FS, without_searchback), TOLERANCE
    )
    assert missed.false_negatives == 1


def test_pan_tompkins_on_a_flat_signal_finds_nothing() -> None:
    assert detect_qrs_pan_tompkins(np.zeros(3600), FS).size == 0
    assert detect_qrs_fixed_threshold(np.zeros(3600), FS).size == 0


def test_pan_tompkins_tolerates_moderate_noise(irregular_beats) -> None:
    clean = synthetic_ecg(irregular_beats, irregular_beats[-1] + 1.5)
    noisy = add_white_noise(clean, 6.0, np.random.default_rng(3))
    reference = np.round(irregular_beats * FS).astype(np.int64)
    counts = match_detections(reference, detect_qrs_pan_tompkins(noisy, FS), TOLERANCE)
    assert counts.sensitivity >= 0.97
    assert counts.positive_predictivity >= 0.97


@requires_data
def test_pan_tompkins_on_record_100_is_essentially_perfect() -> None:
    record = load_record("100")
    reference, _ = record.beat_annotations()
    counts = match_detections(
        reference, detect_qrs_pan_tompkins(record.signal_mv, record.sampling_rate_hz), TOLERANCE
    )
    assert reference.size == 2273
    assert counts.false_negatives <= 2
    assert counts.false_positives <= 2
