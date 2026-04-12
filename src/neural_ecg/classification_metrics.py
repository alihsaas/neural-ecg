from dataclasses import dataclass

import numpy as np

from neural_ecg.config import AAMI_CLASSES


def confusion_matrix(
    true_labels: np.ndarray, predicted_labels: np.ndarray, class_count: int = len(AAMI_CLASSES)
) -> np.ndarray:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("Label arrays must have the same shape")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (true_labels, predicted_labels), 1)
    return matrix


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


@dataclass(frozen=True)
class ClassificationScores:
    matrix: np.ndarray

    @property
    def sensitivity(self) -> np.ndarray:
        return np.array([safe_ratio(self.matrix[k, k], self.matrix[k].sum()) for k in self.classes])

    @property
    def positive_predictivity(self) -> np.ndarray:
        return np.array(
            [safe_ratio(self.matrix[k, k], self.matrix[:, k].sum()) for k in self.classes]
        )

    @property
    def f1(self) -> np.ndarray:
        values = []
        for k in self.classes:
            true_positive = self.matrix[k, k]
            false_positive = self.matrix[:, k].sum() - true_positive
            false_negative = self.matrix[k].sum() - true_positive
            values.append(
                safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
            )
        return np.array(values)

    @property
    def classes(self) -> range:
        return range(self.matrix.shape[0])

    @property
    def macro_f1(self) -> float:
        return float(np.nanmean(self.f1))

    @property
    def accuracy(self) -> float:
        return safe_ratio(float(np.trace(self.matrix)), float(self.matrix.sum()))

    def as_row(self) -> dict[str, float]:
        row: dict[str, float] = {"accuracy": self.accuracy, "macro_f1": self.macro_f1}
        for index, name in enumerate(AAMI_CLASSES[: self.matrix.shape[0]]):
            row[f"se_{name}"] = float(self.sensitivity[index])
            row[f"ppv_{name}"] = float(self.positive_predictivity[index])
            row[f"f1_{name}"] = float(self.f1[index])
        return row


def score(true_labels: np.ndarray, predicted_labels: np.ndarray) -> ClassificationScores:
    return ClassificationScores(confusion_matrix(true_labels, predicted_labels))


def per_record_matrices(
    true_labels: np.ndarray, predicted_labels: np.ndarray, records: np.ndarray
) -> tuple[list[str], np.ndarray]:
    names = sorted(set(np.asarray(records).tolist()))
    matrices = np.stack(
        [
            confusion_matrix(true_labels[records == name], predicted_labels[records == name])
            for name in names
        ]
    )
    return names, matrices


def record_bootstrap(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    records: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    _, matrices = per_record_matrices(true_labels, predicted_labels, records)
    draws: dict[str, list[float]] = {}
    for _ in range(resamples):
        chosen = rng.integers(0, matrices.shape[0], size=matrices.shape[0])
        row = ClassificationScores(matrices[chosen].sum(axis=0)).as_row()
        for key, value in row.items():
            draws.setdefault(key, []).append(value)
    return {
        key: (float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5)))
        for key, values in draws.items()
    }
