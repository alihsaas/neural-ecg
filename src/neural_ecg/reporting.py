import numpy as np

from neural_ecg.classification_metrics import record_bootstrap, score

INTERVAL_KEYS = ("macro_f1", "se_S", "ppv_S", "se_V", "ppv_V", "se_F")


def metrics_row(
    identity: dict,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    records: np.ndarray | None = None,
    resamples: int = 0,
    seed: int = 0,
) -> dict:
    row = dict(identity)
    row["beats"] = int(true_labels.size)
    row.update(score(true_labels, predicted_labels).as_row())
    if records is not None and resamples > 0:
        intervals = record_bootstrap(
            true_labels, predicted_labels, records, resamples, np.random.default_rng(seed)
        )
        for key in INTERVAL_KEYS:
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = intervals[key]
    return row


def percent(value: float | str) -> str:
    number = float(value)
    return "n/a" if np.isnan(number) else f"{100 * number:.1f}"
