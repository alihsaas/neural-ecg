import math

import numpy as np

from neural_ecg.beats import RR_FEATURE_NAMES, BeatSet
from neural_ecg.classical import CLASSICAL_MODELS, classical_model, fit_predict
from neural_ecg.classification_metrics import confusion_matrix, record_bootstrap, score


def test_confusion_matrix_rows_are_true_classes() -> None:
    matrix = confusion_matrix(np.array([0, 0, 1, 2, 3]), np.array([0, 1, 1, 2, 0]))
    expected = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
    np.testing.assert_array_equal(matrix, expected)


def test_scores_on_a_known_matrix() -> None:
    scores = score(np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 0, 2]))
    np.testing.assert_allclose(scores.sensitivity[:3], [2 / 3, 1 / 2, 1.0])
    np.testing.assert_allclose(scores.positive_predictivity[:3], [2 / 3, 1 / 2, 1.0])
    assert math.isnan(scores.f1[3])
    assert math.isclose(scores.macro_f1, np.mean([2 / 3, 1 / 2, 1.0]))
    assert math.isclose(scores.accuracy, 4 / 6)


def test_record_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=2000)
    predicted = np.where(rng.random(2000) < 0.8, labels, rng.integers(0, 4, size=2000))
    records = np.repeat([f"r{i}" for i in range(20)], 100)
    first = record_bootstrap(labels, predicted, records, 200, np.random.default_rng(1))
    second = record_bootstrap(labels, predicted, records, 200, np.random.default_rng(1))
    assert first == second
    estimate = score(labels, predicted).macro_f1
    assert first["macro_f1"][0] <= estimate <= first["macro_f1"][1]


def toy_beats(count: int, seed: int) -> BeatSet:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=count)
    waveforms = rng.normal(0, 0.05, size=(count, 234)).astype(np.float32)
    waveforms[:, 90] += 1.0 + labels
    rr = rng.normal(1.0, 0.05, size=(count, len(RR_FEATURE_NAMES))).astype(np.float32)
    rr[:, 4] -= 0.3 * (labels == 1)
    return BeatSet(waveforms, rr, labels, np.full(count, "900"), np.arange(count))


def test_every_classical_model_learns_a_separable_toy_problem() -> None:
    train = toy_beats(800, 1)
    test = toy_beats(400, 2)
    for model in CLASSICAL_MODELS:
        predicted = fit_predict(model, train, {"test": test})["test"]
        threshold = 0.45 if model.name.startswith("rr_") else 0.9
        assert score(test.labels, predicted).accuracy > threshold, model.name


def test_classical_fit_is_deterministic() -> None:
    train = toy_beats(500, 3)
    test = toy_beats(200, 4)
    model = classical_model("logistic_norm")
    first = fit_predict(model, train, {"test": test})["test"]
    second = fit_predict(model, train, {"test": test})["test"]
    np.testing.assert_array_equal(first, second)
