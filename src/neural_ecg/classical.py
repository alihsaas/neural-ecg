from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_ecg.beats import NORMALIZED_RR_COLUMNS, BeatSet
from neural_ecg.config import AAMI_CLASSES

MORPHOLOGY_STEP = 6


def morphology_features(beats: BeatSet, step: int = MORPHOLOGY_STEP) -> np.ndarray:
    return beats.waveforms[:, ::step].astype(np.float64)


def all_rr_features(beats: BeatSet) -> np.ndarray:
    return beats.rr_features.astype(np.float64)


def normalized_rr_features(beats: BeatSet) -> np.ndarray:
    return beats.rr_features[:, list(NORMALIZED_RR_COLUMNS)].astype(np.float64)


def morphology_and_all_rr(beats: BeatSet) -> np.ndarray:
    return np.hstack([morphology_features(beats), all_rr_features(beats)])


def morphology_and_normalized_rr(beats: BeatSet) -> np.ndarray:
    return np.hstack([morphology_features(beats), normalized_rr_features(beats)])


def logistic_regression() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0),
    )


def weighted_lda() -> Pipeline:
    uniform = np.full(len(AAMI_CLASSES), 1.0 / len(AAMI_CLASSES))
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=uniform),
    )


@dataclass(frozen=True)
class ClassicalModel:
    name: str
    features: Callable[[BeatSet], np.ndarray]
    build: Callable[[], Pipeline]
    description: str


CLASSICAL_MODELS = (
    ClassicalModel(
        "rr_logistic",
        all_rr_features,
        logistic_regression,
        "Balanced logistic regression, 7 RR features (absolute seconds and ratios)",
    ),
    ClassicalModel(
        "rr_norm_logistic",
        normalized_rr_features,
        logistic_regression,
        "Balanced logistic regression, 3 patient-normalized RR ratios only",
    ),
    ClassicalModel(
        "lda",
        morphology_and_all_rr,
        weighted_lda,
        "Shrinkage LDA, equal priors, 39 morphology samples + 7 RR features",
    ),
    ClassicalModel(
        "lda_norm",
        morphology_and_normalized_rr,
        weighted_lda,
        "Shrinkage LDA, equal priors, 39 morphology samples + 3 normalized RR ratios",
    ),
    ClassicalModel(
        "logistic",
        morphology_and_all_rr,
        logistic_regression,
        "Balanced logistic regression, 39 morphology samples + 7 RR features",
    ),
    ClassicalModel(
        "logistic_norm",
        morphology_and_normalized_rr,
        logistic_regression,
        "Balanced logistic regression, 39 morphology samples + 3 normalized RR ratios",
    ),
)


def classical_model(name: str) -> ClassicalModel:
    for model in CLASSICAL_MODELS:
        if model.name == name:
            return model
    raise KeyError(name)


def fit_predict(
    model: ClassicalModel, train: BeatSet, evaluation_sets: dict[str, BeatSet]
) -> dict[str, np.ndarray]:
    pipeline = model.build()
    pipeline.fit(model.features(train), train.labels)
    return {
        name: pipeline.predict(model.features(beats)).astype(np.int64)
        for name, beats in evaluation_sets.items()
    }
