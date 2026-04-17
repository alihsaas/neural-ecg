import numpy as np
import pytest
from conftest import TORCH_AVAILABLE, requires_torch

if TORCH_AVAILABLE:
    import torch

    from neural_ecg.neural import (
        BeatCnn,
        RrScaler,
        RunSettings,
        checkpoint_payload,
        class_weights,
        load_checkpoint,
        parameter_count,
        predict,
        train_model,
    )

from test_classification import toy_beats


def settings(rr_input: str = "normalized", epochs: int = 3, seed: int = 5) -> "RunSettings":
    return RunSettings("toy", rr_input, seed, epochs, 64, 0.003, 0.0, 0.0, 0.9, 1.1, 0.01)


@requires_torch
@pytest.mark.parametrize("rr_input, width", [("none", 0), ("normalized", 3), ("all", 7)])
def test_model_shapes_and_size(rr_input: str, width: int) -> None:
    columns = settings(rr_input).rr_columns()
    assert len(columns) == width
    model = BeatCnn(width, 0.3)
    logits = model(torch.zeros(5, 234), torch.zeros(5, width))
    assert logits.shape == (5, 4)
    assert parameter_count(model) < 20_000


@requires_torch
def test_class_weights_favor_rare_classes() -> None:
    weights = class_weights(np.array([0] * 100 + [1] * 4 + [2] * 25 + [3] * 1))
    assert weights[3] > weights[1] > weights[2] > weights[0]
    assert np.isclose(weights.mean(), 1.0)


@requires_torch
def test_scaler_uses_training_statistics_only() -> None:
    train = toy_beats(300, 1)
    test = toy_beats(300, 2)
    scaler = RrScaler.fit(train, [4, 5, 6])
    np.testing.assert_allclose(scaler.transform(train).mean(axis=0), 0.0, atol=1e-5)
    assert scaler.transform(test).shape == (300, 3)
    assert RrScaler.fit(train, []).transform(test).shape == (300, 0)


@requires_torch
def test_training_is_reproducible_and_learns(tmp_path) -> None:
    train = toy_beats(600, 1)
    validation = toy_beats(200, 2)
    device = torch.device("cpu")
    model, scaler, history, best_epoch = train_model(settings(), train, validation, device)
    _, _, history_again, _ = train_model(settings(), train, validation, device)
    assert history == history_again
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert 1 <= best_epoch <= 3
    path = tmp_path / "toy.pt"
    torch.save(checkpoint_payload(model, scaler, settings(), best_epoch, history), path)
    restored, restored_scaler, restored_settings, _ = load_checkpoint(path, device)
    np.testing.assert_array_equal(
        predict(model, validation, scaler, device),
        predict(restored, validation, restored_scaler, device),
    )
    assert restored_settings == settings()
