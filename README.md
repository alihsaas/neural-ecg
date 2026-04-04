# NeuralECG

I use this project to study ECG signal processing and heartbeat classification.

## Setup

I use Python 3.12 and uv. Install the locked dependencies with `uv sync --locked`.

## ECG records

I use the MIT-BIH Arrhythmia Database. I put the `.dat`, `.hea`, and `.atr` files in `data/mitdb/` and use the record annotations to evaluate the detectors.
