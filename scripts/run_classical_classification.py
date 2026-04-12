import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from neural_ecg.artifacts import package_versions, write_csv, write_json
from neural_ecg.beats import RR_FEATURE_NAMES, concatenate_beats
from neural_ecg.classical import CLASSICAL_MODELS, MORPHOLOGY_STEP, classical_model, fit_predict
from neural_ecg.classification_metrics import confusion_matrix
from neural_ecg.conditions import cached_beats
from neural_ecg.config import AAMI_CLASSES
from neural_ecg.plotting import plot_confusion, plot_grouped_bars
from neural_ecg.protocol import DEFAULT_PROTOCOL, ClassificationProtocol
from neural_ecg.reporting import metrics_row, percent


def matrix_rows(model: str, split: str, matrix: np.ndarray) -> list[dict]:
    rows = []
    for true_index, true_name in enumerate(AAMI_CLASSES):
        row = {"model": model, "split": split, "true_class": true_name}
        for predicted_index, predicted_name in enumerate(AAMI_CLASSES):
            row[f"predicted_{predicted_name}"] = int(matrix[true_index, predicted_index])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    arguments = parser.parse_args()
    protocol = ClassificationProtocol.load(arguments.protocol)
    output = protocol.path("classical_results")
    window = protocol.beat_window
    bootstrap = protocol.values["bootstrap"]
    train = cached_beats(protocol.train_records, window=window)
    validation = cached_beats(protocol.validation_records, window=window)
    test = cached_beats(protocol.test_records, window=window)
    rows = []
    matrices = []
    for model in CLASSICAL_MODELS:
        predictions = fit_predict(model, train, {"validation": validation, "test": test})
        for split, beats in (
            ("inter_patient_validation", validation),
            ("inter_patient_test", test),
        ):
            predicted = predictions[split.removeprefix("inter_patient_")]
            rows.append(
                metrics_row(
                    {"model": model.name, "split": split, "train_beats": len(train)},
                    beats.labels,
                    predicted,
                    beats.records,
                    bootstrap["resamples"] if split == "inter_patient_test" else 0,
                    bootstrap["seed"],
                )
            )
        matrix = confusion_matrix(test.labels, predictions["test"])
        matrices += matrix_rows(model.name, "inter_patient_test", matrix)
        plot_confusion(
            matrix, f"{model.name}, inter-patient DS2", output / f"confusion_{model.name}_inter.png"
        )
    pooled = concatenate_beats([train, validation, test])
    intra = protocol.values["intra_patient"]
    indices = np.arange(len(pooled))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=intra["test_fraction"],
        random_state=intra["seed"],
        stratify=pooled.labels,
    )
    intra_train = pooled.subset(np.isin(indices, train_indices))
    intra_test = pooled.subset(np.isin(indices, test_indices))
    logistic = classical_model("logistic_norm")
    intra_predicted = fit_predict(logistic, intra_train, {"test": intra_test})["test"]
    rows.append(
        metrics_row(
            {
                "model": "logistic_norm",
                "split": "intra_patient_test",
                "train_beats": len(intra_train),
            },
            intra_test.labels,
            intra_predicted,
        )
    )
    intra_matrix = confusion_matrix(intra_test.labels, intra_predicted)
    matrices += matrix_rows("logistic_norm", "intra_patient_test", intra_matrix)
    plot_confusion(
        intra_matrix,
        "logistic_norm, intra-patient random split",
        output / "confusion_logistic_norm_intra.png",
    )
    write_csv(rows, output / "metrics.csv")
    write_csv(matrices, output / "confusion_matrices.csv")
    test_rows = [row for row in rows if row["split"] == "inter_patient_test"]
    groups = ["macro F1", "Se S", "+P S", "Se V", "+P V", "Se F"]
    keys = ["macro_f1", "se_S", "ppv_S", "se_V", "ppv_V", "se_F"]
    plot_grouped_bars(
        groups,
        {row["model"]: [100 * row[key] for key in keys] for row in test_rows},
        "Percent",
        "Classical models, inter-patient test (DS2), 95% record-bootstrap intervals",
        output / "classical_inter_patient.png",
        {
            row["model"]: [
                (100 * row[f"{key}_ci_low"], 100 * row[f"{key}_ci_high"]) for key in keys
            ]
            for row in test_rows
        },
    )
    write_json(
        {
            "protocol": protocol.values,
            "class_counts": {
                "train": train.class_counts(),
                "validation": validation.class_counts(),
                "test": test.class_counts(),
                "intra_patient_train": intra_train.class_counts(),
                "intra_patient_test": intra_test.class_counts(),
            },
            "models": {model.name: model.description for model in CLASSICAL_MODELS},
            "rr_features": list(RR_FEATURE_NAMES),
            "morphology": f"every {MORPHOLOGY_STEP}th sample of the 0.5-40 Hz filtered beat window",
            "hyperparameters": "fixed in code; nothing tuned on validation or test beats",
            "versions": package_versions(),
        },
        output / "metadata.json",
    )
    for row in rows:
        print(
            f"{row['model']:>12} {row['split']:>26} acc={percent(row['accuracy'])} "
            f"macroF1={percent(row['macro_f1'])} SeS={percent(row['se_S'])} +PS={percent(row['ppv_S'])} "
            f"SeV={percent(row['se_V'])} +PV={percent(row['ppv_V'])} SeF={percent(row['se_F'])}"
        )


if __name__ == "__main__":
    main()
