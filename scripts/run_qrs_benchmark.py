import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_ecg.artifacts import load_protocol, package_versions, read_csv, write_csv, write_json
from neural_ecg.conditions import SignalCondition
from neural_ecg.config import ALL_RECORDS, PROJECT_ROOT
from neural_ecg.detection_metrics import DetectionCounts, match_detections
from neural_ecg.qrs import (
    FixedThresholdSettings,
    PanTompkinsSettings,
    detect_qrs_fixed_threshold,
    detect_qrs_pan_tompkins,
    front_end,
)
from neural_ecg.records import load_record

DETECTORS = {
    "fixed_threshold": detect_qrs_fixed_threshold,
    "pan_tompkins": detect_qrs_pan_tompkins,
}


def evaluate_record(arguments: tuple[str, list[dict], int, float]) -> list[dict]:
    name, condition_dicts, root_seed, tolerance_s = arguments
    record = load_record(name)
    reference, _ = record.beat_annotations()
    tolerance = round(tolerance_s * record.sampling_rate_hz)
    rows = []
    for values in condition_dicts:
        condition = SignalCondition.from_dict(values)
        noisy = condition.apply(record, root_seed)
        for detector_name, detector in DETECTORS.items():
            detections = detector(noisy.signal_mv, noisy.sampling_rate_hz)
            counts = match_detections(reference, detections, tolerance)
            rows.append(
                {
                    "record": name,
                    "lead": record.lead,
                    "condition": condition.name,
                    "detector": detector_name,
                    "reference_beats": counts.reference_beats,
                    "true_positives": counts.true_positives,
                    "false_positives": counts.false_positives,
                    "false_negatives": counts.false_negatives,
                    "sensitivity": counts.sensitivity,
                    "positive_predictivity": counts.positive_predictivity,
                }
            )
    return rows


def summarize(rows: list[dict], conditions: list[str]) -> list[dict]:
    summary = []
    for condition in conditions:
        for detector in DETECTORS:
            total = DetectionCounts(0, 0, 0)
            for row in rows:
                if row["condition"] == condition and row["detector"] == detector:
                    total = total + DetectionCounts(
                        int(row["true_positives"]),
                        int(row["false_positives"]),
                        int(row["false_negatives"]),
                    )
            summary.append(
                {
                    "condition": condition,
                    "detector": detector,
                    "reference_beats": total.reference_beats,
                    "true_positives": total.true_positives,
                    "false_positives": total.false_positives,
                    "false_negatives": total.false_negatives,
                    "sensitivity": total.sensitivity,
                    "positive_predictivity": total.positive_predictivity,
                    "f1": total.f1,
                    "detection_error_rate": total.detection_error_rate,
                }
            )
    return summary


def plot_stages(output: Path) -> None:
    record = load_record("100")
    fs = record.sampling_rate_hz
    stop = round(3.0 * fs)
    stages = front_end(record.signal_mv, fs, PanTompkinsSettings().front_end)
    detections = detect_qrs_pan_tompkins(record.signal_mv, fs)
    reference, _ = record.beat_annotations()
    time_s = np.arange(stop) / fs
    panels = [
        ("ECG (mV)", record.signal_mv),
        ("Band-pass 5-15 Hz", stages.bandpassed),
        ("Squared derivative", stages.squared),
        ("Moving-window integral", stages.integrated),
    ]
    figure, axes = plt.subplots(len(panels), 1, figsize=(9, 7), sharex=True)
    for axis, (label, values) in zip(axes, panels, strict=True):
        axis.plot(time_s, values[:stop], linewidth=0.9)
        axis.set_ylabel(label, fontsize=8)
        axis.grid(alpha=0.3)
    shown_detections = detections[detections < stop]
    shown_reference = reference[reference < stop]
    axes[0].plot(
        shown_reference / fs,
        record.signal_mv[shown_reference],
        "o",
        fillstyle="none",
        label="annotation",
    )
    axes[0].plot(
        shown_detections / fs, record.signal_mv[shown_detections], "x", label="Pan-Tompkins"
    )
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("Pan-Tompkins stages on record 100 (first 3 s)")
    figure.tight_layout()
    figure.savefig(output / "pan_tompkins_stages.png", dpi=150)
    plt.close(figure)


def plot_summary(summary: list[dict], conditions: list[str], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    positions = np.arange(len(conditions))
    width = 0.38
    for offset, detector in zip((-width / 2, width / 2), DETECTORS, strict=True):
        rows = [
            next(r for r in summary if r["condition"] == c and r["detector"] == detector)
            for c in conditions
        ]
        for axis, key, label in zip(
            axes,
            ("sensitivity", "positive_predictivity"),
            ("Sensitivity (%)", "Positive predictivity (%)"),
            strict=True,
        ):
            axis.bar(positions + offset, [100 * float(r[key]) for r in rows], width, label=detector)
            axis.set_ylabel(label)
    for axis in axes:
        axis.set_xticks(positions, conditions, rotation=30, ha="right", fontsize=8)
        axis.set_ylim(80, 100.5)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8, loc="lower left")
    figure.suptitle("QRS detection on all 48 MIT-BIH records (gross counts)")
    figure.tight_layout()
    figure.savefig(output / "detection_by_condition.png", dpi=150)
    plt.close(figure)


def plot_per_record(rows: list[dict], output: Path) -> None:
    clean = [r for r in rows if r["condition"] == "clean" and r["detector"] == "pan_tompkins"]
    names = [r["record"] for r in clean]
    errors = [int(r["false_positives"]) + int(r["false_negatives"]) for r in clean]
    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.bar(np.arange(len(names)), errors)
    axis.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=7)
    axis.set_ylabel("FP + FN beats")
    axis.set_title("Pan-Tompkins errors per record (clean signals)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "pan_tompkins_errors_per_record.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs/qrs_detection/protocol.json"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--plots-only", action="store_true")
    arguments = parser.parse_args()
    protocol = load_protocol(arguments.protocol)
    output = PROJECT_ROOT / protocol["paths"]["results"]
    output.mkdir(parents=True, exist_ok=True)
    conditions = [values["name"] for values in protocol["conditions"]]
    if arguments.plots_only:
        rows = read_csv(output / "per_record.csv")
        summary = read_csv(output / "summary.csv")
    else:
        jobs = [
            (name, protocol["conditions"], protocol["noise_root_seed"], protocol["tolerance_s"])
            for name in ALL_RECORDS
        ]
        if arguments.workers > 1:
            with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
                results = list(pool.map(evaluate_record, jobs))
        else:
            results = [evaluate_record(job) for job in jobs]
        rows = [row for record_rows in results for row in record_rows]
        summary = summarize(rows, conditions)
        write_csv(rows, output / "per_record.csv")
        write_csv(summary, output / "summary.csv")
        write_json(
            {
                "protocol": protocol,
                "records": list(ALL_RECORDS),
                "lead_rule": "MLII when present, otherwise the first channel",
                "reference_beats": "MIT-BIH beat annotations (wfdb beat symbols), whole record",
                "snr_definition": "10*log10(var(clean record) / var(added white noise)), per record",
                "pan_tompkins_settings": repr(PanTompkinsSettings()),
                "fixed_threshold_settings": repr(FixedThresholdSettings()),
                "versions": package_versions(),
            },
            output / "metadata.json",
        )
    plot_stages(output)
    plot_summary(summary, conditions, output)
    plot_per_record(rows, output)
    for row in summary:
        print(
            f"{row['condition']:>11} {row['detector']:>15} "
            f"Se={100 * float(row['sensitivity']):.3f}% +P={100 * float(row['positive_predictivity']):.3f}% "
            f"FP={row['false_positives']} FN={row['false_negatives']}"
        )


if __name__ == "__main__":
    main()
