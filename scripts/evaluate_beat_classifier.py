import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_ecg.artifacts import package_versions, read_csv, write_csv, write_json
from neural_ecg.classical import classical_model, fit_predict
from neural_ecg.classification_metrics import (
    ClassificationScores,
    confusion_matrix,
    per_record_matrices,
)
from neural_ecg.conditions import CLEAN, cached_beats
from neural_ecg.neural import load_checkpoint, predict
from neural_ecg.plotting import plot_confusion, plot_grouped_bars
from neural_ecg.protocol import DEFAULT_PROTOCOL, ClassificationProtocol
from neural_ecg.reporting import metrics_row, percent

CLASSICAL_REFERENCES = ("rr_norm_logistic", "logistic", "logistic_norm", "lda")
SUMMARY_KEYS = (
    "accuracy",
    "macro_f1",
    "se_S",
    "ppv_S",
    "f1_S",
    "se_V",
    "ppv_V",
    "f1_V",
    "se_F",
    "ppv_F",
)


def paired_difference(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    records: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    _, first_matrices = per_record_matrices(labels, first, records)
    _, second_matrices = per_record_matrices(labels, second, records)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        chosen = rng.integers(0, first_matrices.shape[0], size=first_matrices.shape[0])
        differences.append(
            ClassificationScores(first_matrices[chosen].sum(axis=0)).macro_f1
            - ClassificationScores(second_matrices[chosen].sum(axis=0)).macro_f1
        )
    observed = (
        ClassificationScores(first_matrices.sum(axis=0)).macro_f1
        - ClassificationScores(second_matrices.sum(axis=0)).macro_f1
    )
    return observed, float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))


def cpu_latency_ms(
    checkpoint: Path, beat_waveform: np.ndarray, rr_width: int, repeats: int = 500
) -> float:
    model, _, _, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    torch.set_num_threads(1)
    waveform = torch.from_numpy(beat_waveform[None, :].copy())
    rr = torch.zeros((1, rr_width))
    with torch.no_grad():
        for _ in range(50):
            model(waveform, rr)
        start = time.perf_counter()
        for _ in range(repeats):
            model(waveform, rr)
    return 1000.0 * (time.perf_counter() - start) / repeats


def seed_summary(rows: list[dict]) -> list[dict]:
    summary = []
    groups = sorted({(row["model"], row["condition"]) for row in rows if row["family"] == "neural"})
    for model, condition in groups:
        chosen = [row for row in rows if row["model"] == model and row["condition"] == condition]
        entry = {"model": model, "condition": condition, "seeds": len(chosen)}
        for key in SUMMARY_KEYS:
            values = np.array([float(row[key]) for row in chosen])
            entry[f"{key}_mean"] = float(np.nanmean(values))
            entry[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
            entry[f"{key}_min"] = float(np.nanmin(values))
            entry[f"{key}_max"] = float(np.nanmax(values))
        summary.append(entry)
    return summary


def plot_overview(rows: list[dict], summary: list[dict], output: Path) -> None:
    keys = ["macro_f1", "se_S", "ppv_S", "se_V", "ppv_V", "se_F"]
    groups = ["macro F1", "Se S", "+P S", "Se V", "+P V", "Se F"]
    series = {}
    errors = {}
    for name in CLASSICAL_REFERENCES:
        row = next(r for r in rows if r["model"] == name and r["condition"] == "clean")
        series[name] = [100 * float(row[key]) for key in keys]
        errors[name] = [
            (100 * float(row[f"{key}_ci_low"]), 100 * float(row[f"{key}_ci_high"])) for key in keys
        ]
    for entry in summary:
        if entry["condition"] != "clean":
            continue
        label = f"{entry['model']} (3 seeds)"
        series[label] = [100 * float(entry[f"{key}_mean"]) for key in keys]
        errors[label] = [
            (100 * float(entry[f"{key}_min"]), 100 * float(entry[f"{key}_max"])) for key in keys
        ]
    plot_grouped_bars(
        groups,
        series,
        "Percent",
        "Inter-patient test (DS2). Classical: 95% record bootstrap. CNN: seed mean, min-max",
        output / "test_overview.png",
        errors,
    )


def plot_robustness(
    rows: list[dict], summary: list[dict], conditions: list[str], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, key, label in zip(
        axes, ("macro_f1", "se_S"), ("Macro F1 (%)", "S sensitivity (%)"), strict=True
    ):
        for name in CLASSICAL_REFERENCES:
            values = [
                100
                * float(next(r for r in rows if r["model"] == name and r["condition"] == c)[key])
                for c in conditions
            ]
            axis.plot(conditions, values, "o--", label=name)
        for model in sorted({entry["model"] for entry in summary}):
            values = [
                100
                * float(
                    next(e for e in summary if e["model"] == model and e["condition"] == c)[
                        f"{key}_mean"
                    ]
                )
                for c in conditions
            ]
            axis.plot(conditions, values, "s-", label=f"{model} (mean)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.tick_params(axis="x", labelrotation=20)
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figure.suptitle("Noise robustness on DS2 (models trained on clean DS1 only)")
    figure.tight_layout()
    figure.savefig(output / "robustness.png", dpi=150)
    plt.close(figure)


def plot_training(checkpoints: Path, runs: list[str], seeds: list[int], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        for seed in seeds:
            history = read_csv(checkpoints / f"{run}_seed{seed}_history.csv")
            epochs = [int(row["epoch"]) for row in history]
            axes[0].plot(
                epochs,
                [float(row["validation_loss"]) for row in history],
                label=f"{run} s{seed}",
                linewidth=0.9,
            )
            axes[1].plot(
                epochs, [100 * float(row["validation_macro_f1"]) for row in history], linewidth=0.9
            )
    axes[0].set_ylabel("Validation loss (weighted CE)")
    axes[1].set_ylabel("Validation macro F1 (%)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    figure.suptitle("Training curves on held-out DS1 validation records")
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    arguments = parser.parse_args()
    protocol = ClassificationProtocol.load(arguments.protocol)
    output = protocol.path("neural_results")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = protocol.path("checkpoints")
    window = protocol.beat_window
    bootstrap = protocol.values["bootstrap"]
    root_seed = protocol.values["noise_root_seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = cached_beats(protocol.train_records, window=window)
    conditions = [CLEAN] + protocol.robustness_conditions
    test_sets = {
        c.name: cached_beats(protocol.test_records, c, root_seed, window) for c in conditions
    }
    reference = test_sets["clean"]
    for name, beats in test_sets.items():
        if not (
            np.array_equal(beats.labels, reference.labels)
            and np.array_equal(beats.samples, reference.samples)
        ):
            raise RuntimeError(
                f"Condition {name} does not contain the same beats as the clean test set"
            )
    predictions: dict[tuple[str, int | None], dict[str, np.ndarray]] = {}
    for name in CLASSICAL_REFERENCES:
        predictions[(name, None)] = fit_predict(classical_model(name), train, test_sets)
    runs = [run["name"] for run in protocol.values["neural"]["runs"]]
    seeds = list(protocol.values["neural"]["seeds"])
    parameters = {}
    best_epochs = {}
    for run in runs:
        for seed in seeds:
            path = checkpoints / f"{run}_seed{seed}.pt"
            model, scaler, _, payload = load_checkpoint(path, device)
            parameters[run] = payload["parameters"]
            best_epochs[f"{run}_seed{seed}"] = payload["best_epoch"]
            predictions[(run, seed)] = {
                name: predict(model, beats, scaler, device) for name, beats in test_sets.items()
            }
    rows = []
    for (model_name, seed), by_condition in predictions.items():
        for condition in conditions:
            clean = condition.name == "clean"
            rows.append(
                metrics_row(
                    {
                        "model": model_name,
                        "family": "classical" if seed is None else "neural",
                        "seed": "" if seed is None else seed,
                        "condition": condition.name,
                    },
                    reference.labels,
                    by_condition[condition.name],
                    reference.records if clean else None,
                    bootstrap["resamples"] if clean else 0,
                    bootstrap["seed"],
                )
            )
    summary = seed_summary(rows)
    comparisons = []
    for run in runs:
        for seed in seeds:
            for name in CLASSICAL_REFERENCES:
                observed, low, high = paired_difference(
                    reference.labels,
                    predictions[(run, seed)]["clean"],
                    predictions[(name, None)]["clean"],
                    reference.records,
                    bootstrap["resamples"],
                    bootstrap["seed"],
                )
                comparisons.append(
                    {
                        "neural": run,
                        "seed": seed,
                        "classical": name,
                        "macro_f1_difference": observed,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    latency = {
        run: cpu_latency_ms(
            checkpoints / f"{run}_seed{seeds[0]}.pt",
            reference.waveforms[0],
            len(
                load_checkpoint(checkpoints / f"{run}_seed{seeds[0]}.pt", torch.device("cpu"))[
                    1
                ].columns
            ),
        )
        for run in runs
    }
    write_csv(rows, output / "metrics.csv")
    write_csv(summary, output / "seed_summary.csv")
    write_csv(comparisons, output / "paired_macro_f1_differences.csv")
    for run in runs:
        matrix = confusion_matrix(reference.labels, predictions[(run, seeds[0])]["clean"])
        plot_confusion(
            matrix,
            f"{run} seed {seeds[0]}, inter-patient DS2",
            output / f"confusion_{run}_seed{seeds[0]}.png",
        )
    plot_overview(rows, summary, output)
    plot_robustness(rows, summary, [c.name for c in conditions], output)
    plot_training(checkpoints, runs, seeds, output)
    write_json(
        {
            "protocol": protocol.values,
            "test_beats": reference.class_counts(),
            "classical_references": list(CLASSICAL_REFERENCES),
            "parameters": parameters,
            "best_epochs": best_epochs,
            "cpu_latency_ms_per_beat_batch1_one_thread": latency,
            "device": str(device),
            "versions": package_versions(),
        },
        output / "metadata.json",
    )
    for row in rows:
        if row["condition"] == "clean":
            print(
                f"{row['model']:>16} {str(row['seed']):>3} macroF1={percent(row['macro_f1'])} "
                f"SeS={percent(row['se_S'])} +PS={percent(row['ppv_S'])} SeV={percent(row['se_V'])} "
                f"+PV={percent(row['ppv_V'])} SeF={percent(row['se_F'])} acc={percent(row['accuracy'])}"
            )
    for entry in summary:
        print(
            f"{entry['model']:>14} {entry['condition']:>10} macroF1={100 * entry['macro_f1_mean']:.1f}"
            f"+-{100 * entry['macro_f1_std']:.1f} SeS={100 * entry['se_S_mean']:.1f} SeV={100 * entry['se_V_mean']:.1f}"
        )
    print("latency ms", latency, "params", parameters, "best epochs", best_epochs)


if __name__ == "__main__":
    main()
