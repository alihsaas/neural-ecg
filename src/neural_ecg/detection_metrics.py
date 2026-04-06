from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionCounts:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def reference_beats(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def sensitivity(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else float("nan")

    @property
    def positive_predictivity(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else float("nan")

    @property
    def f1(self) -> float:
        total = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / total if total else float("nan")

    @property
    def detection_error_rate(self) -> float:
        reference = self.reference_beats
        return (
            (self.false_positives + self.false_negatives) / reference if reference else float("nan")
        )

    def __add__(self, other: "DetectionCounts") -> "DetectionCounts":
        return DetectionCounts(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )


def match_detections(
    reference: np.ndarray, detected: np.ndarray, tolerance_samples: int
) -> DetectionCounts:
    reference = np.sort(np.asarray(reference, dtype=np.int64))
    detected = np.sort(np.asarray(detected, dtype=np.int64))
    reference_index = 0
    detected_index = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    while reference_index < reference.size and detected_index < detected.size:
        difference = int(detected[detected_index]) - int(reference[reference_index])
        if abs(difference) <= tolerance_samples:
            true_positives += 1
            reference_index += 1
            detected_index += 1
        elif difference < 0:
            false_positives += 1
            detected_index += 1
        else:
            false_negatives += 1
            reference_index += 1
    false_negatives += reference.size - reference_index
    false_positives += detected.size - detected_index
    return DetectionCounts(true_positives, false_positives, false_negatives)
