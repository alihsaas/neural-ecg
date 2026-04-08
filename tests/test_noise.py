import numpy as np
import pytest

from neural_ecg.conditions import SignalCondition
from neural_ecg.noise import add_baseline_wander, add_white_noise, condition_rng, signal_power
from neural_ecg.records import EcgRecord


@pytest.mark.parametrize("snr_db", [0.0, 6.0, 18.0])
def test_white_noise_reaches_the_requested_snr(snr_db: float) -> None:
    clean = np.sin(np.linspace(0, 400 * np.pi, 200_000)) + 0.5
    noisy = add_white_noise(clean, snr_db, np.random.default_rng(1))
    measured = 10 * np.log10(signal_power(clean) / signal_power(noisy - clean))
    assert abs(measured - snr_db) < 0.05


def test_noise_is_reproducible_per_record_and_condition() -> None:
    first = condition_rng(2026, "100", 3).normal(size=5)
    again = condition_rng(2026, "100", 3).normal(size=5)
    other_record = condition_rng(2026, "101", 3).normal(size=5)
    other_condition = condition_rng(2026, "100", 4).normal(size=5)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other_record)
    assert not np.allclose(first, other_condition)


def test_baseline_wander_has_the_requested_amplitude() -> None:
    wander = add_baseline_wander(np.zeros(36_000), 360.0, 1.0, 0.33, np.random.default_rng(0))
    assert abs(np.max(np.abs(wander)) - 1.0) < 1e-3


def test_condition_keeps_annotations_and_length() -> None:
    record = EcgRecord(
        "100", 360.0, "MLII", np.ones(1000), np.array([10, 500]), np.array(["N", "V"])
    )
    noisy = SignalCondition("white", 1, "white", snr_db=10.0).apply(
        record.with_signal(np.sin(np.arange(1000.0))), 5
    )
    assert noisy.signal_mv.shape == record.signal_mv.shape
    np.testing.assert_array_equal(noisy.annotation_samples, record.annotation_samples)
    assert SignalCondition("clean", 0).apply(record, 5) is record
    with pytest.raises(ValueError):
        SignalCondition("bad", 9, "pink").apply(record, 5)
