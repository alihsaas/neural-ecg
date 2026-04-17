# NeuralECG

I use this project to study ECG signal processing and heartbeat classification.

## Setup

I use Python 3.12 and uv. Install the locked dependencies with `uv sync --locked`.

## ECG records

I use the MIT-BIH Arrhythmia Database. I put the `.dat`, `.hea`, and `.atr` files in `data/mitdb/` and use the record annotations to evaluate the detectors.

## QRS detection

I compare a Pan-Tompkins detector with a fixed-threshold detector. Run `uv run python scripts/run_qrs_benchmark.py`.

## Tests

Run `uv run pytest -q --basetemp=.pytest-tmp`.

## QRS results

The detection metrics and figures are in [results/qrs_detection](results/qrs_detection/).

## Classical beat classification

I compare logistic regression and LDA on RR intervals and beat shape. Run `uv run python scripts/run_classical_classification.py`.

The classical results are in [results/classical_classification](results/classical_classification/). I use separate patients for the training and test sets.

## Learned beat classification

I use a small 1-D CNN with optional RR features. Install the extra with `uv sync --locked --extra neural`. Train with `uv run --extra neural python scripts/train_beat_classifier.py` and score the checkpoints with `uv run --extra neural python scripts/evaluate_beat_classifier.py`.
