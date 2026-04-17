import argparse
import time
from pathlib import Path

import torch

from neural_ecg.artifacts import package_versions, write_csv, write_json
from neural_ecg.conditions import cached_beats
from neural_ecg.neural import RunSettings, checkpoint_payload, parameter_count, train_model
from neural_ecg.protocol import DEFAULT_PROTOCOL, ClassificationProtocol


def run_settings(
    protocol: ClassificationProtocol, run: dict, seed: int, epochs: int | None
) -> RunSettings:
    neural = protocol.values["neural"]
    augmentation = neural["augmentation"]
    return RunSettings(
        name=run["name"],
        rr_input=run["rr_input"],
        seed=seed,
        epochs=epochs or neural["epochs"],
        batch_size=neural["batch_size"],
        learning_rate=neural["learning_rate"],
        weight_decay=neural["weight_decay"],
        dropout=neural["dropout"],
        amplitude_low=augmentation["amplitude_low"],
        amplitude_high=augmentation["amplitude_high"],
        noise_std_mv=augmentation["noise_std_mv"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run", nargs="*", default=None)
    parser.add_argument("--seed", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    arguments = parser.parse_args()
    protocol = ClassificationProtocol.load(arguments.protocol)
    output = arguments.output or protocol.path("checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = protocol.beat_window
    train = cached_beats(protocol.train_records, window=window)
    validation = cached_beats(protocol.validation_records, window=window)
    runs = [
        run
        for run in protocol.values["neural"]["runs"]
        if arguments.run is None or run["name"] in arguments.run
    ]
    seeds = arguments.seed or protocol.values["neural"]["seeds"]
    print(f"device={device} train={train.class_counts()} validation={validation.class_counts()}")
    for run in runs:
        for seed in seeds:
            path = output / f"{run['name']}_seed{seed}.pt"
            if arguments.skip_existing and path.exists():
                print(f"skip {path.name}")
                continue
            settings = run_settings(protocol, run, seed, arguments.epochs)
            start = time.perf_counter()
            model, scaler, history, best_epoch = train_model(settings, train, validation, device)
            seconds = time.perf_counter() - start
            torch.save(checkpoint_payload(model, scaler, settings, best_epoch, history), path)
            write_csv(history, output / f"{run['name']}_seed{seed}_history.csv")
            best = history[best_epoch - 1]
            print(
                f"{run['name']} seed={seed} params={parameter_count(model)} best_epoch={best_epoch} "
                f"val_macro_f1={best['validation_macro_f1']:.4f} seconds={seconds:.1f}"
            )
    write_json(
        {
            "protocol": protocol.values,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "versions": package_versions(),
        },
        output / "training_metadata.json",
    )


if __name__ == "__main__":
    main()
