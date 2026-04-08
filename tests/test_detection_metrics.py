import math

import numpy as np

from neural_ecg.detection_metrics import DetectionCounts, match_detections


def test_exact_match_counts_every_beat_once() -> None:
    reference = np.array([100, 400, 700])
    assert match_detections(reference, reference, 54) == DetectionCounts(3, 0, 0)


def test_detection_outside_tolerance_is_a_miss_and_a_false_alarm() -> None:
    counts = match_detections(np.array([100]), np.array([160]), 54)
    assert counts == DetectionCounts(0, 1, 1)


def test_edge_of_tolerance_window_counts_as_a_match() -> None:
    assert match_detections(np.array([100]), np.array([154]), 54) == DetectionCounts(1, 0, 0)


def test_double_detection_counts_one_false_positive() -> None:
    counts = match_detections(np.array([500]), np.array([495, 520]), 54)
    assert counts == DetectionCounts(1, 1, 0)


def test_unsorted_inputs_and_empty_lists() -> None:
    counts = match_detections(np.array([700, 100, 400]), np.array([402, 98]), 54)
    assert counts == DetectionCounts(2, 0, 1)
    assert match_detections(np.array([1, 2]), np.array([], dtype=int), 54) == DetectionCounts(
        0, 0, 2
    )
    assert match_detections(np.array([], dtype=int), np.array([5]), 54) == DetectionCounts(0, 1, 0)


def test_rates_and_gross_aggregation() -> None:
    total = DetectionCounts(90, 5, 10) + DetectionCounts(10, 5, 0)
    assert total == DetectionCounts(100, 10, 10)
    assert math.isclose(total.sensitivity, 100 / 110)
    assert math.isclose(total.positive_predictivity, 100 / 110)
    assert math.isclose(total.detection_error_rate, 20 / 110)
    assert math.isnan(DetectionCounts(0, 0, 0).sensitivity)
