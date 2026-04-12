from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_ecg.config import AAMI_CLASSES


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros(matrix.shape), where=row_totals > 0)
    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if normalized[row, column] > 0.6 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{100 * normalized[row, column]:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    labels = list(AAMI_CLASSES[: matrix.shape[0]])
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title, fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_grouped_bars(
    groups: list[str],
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    path: Path,
    errors: dict[str, list[tuple[float, float]]] | None = None,
) -> None:
    positions = np.arange(len(groups))
    width = 0.8 / max(1, len(series))
    figure, axis = plt.subplots(figsize=(max(6.0, 1.4 * len(groups) + 2), 4.0))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        yerr = None
        if errors and name in errors:
            low = [value - bounds[0] for value, bounds in zip(values, errors[name], strict=True)]
            high = [bounds[1] - value for value, bounds in zip(values, errors[name], strict=True)]
            yerr = np.array([low, high])
        axis.bar(positions + offset, values, width, label=name, yerr=yerr, capsize=2)
    axis.set_xticks(positions, groups, fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
