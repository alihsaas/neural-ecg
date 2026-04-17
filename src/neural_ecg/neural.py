from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from neural_ecg.beats import NORMALIZED_RR_COLUMNS, RR_FEATURE_NAMES, BeatSet
from neural_ecg.classification_metrics import score
from neural_ecg.config import AAMI_CLASSES


@dataclass(frozen=True)
class RunSettings:
    name: str
    rr_input: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    amplitude_low: float
    amplitude_high: float
    noise_std_mv: float

    def rr_columns(self) -> list[int]:
        if self.rr_input == "none":
            return []
        if self.rr_input == "normalized":
            return list(NORMALIZED_RR_COLUMNS)
        if self.rr_input == "all":
            return list(range(len(RR_FEATURE_NAMES)))
        raise ValueError(f"Unknown rr_input: {self.rr_input}")


class BeatCnn(nn.Module):
    def __init__(self, rr_features: int, dropout: float, class_count: int = len(AAMI_CLASSES)):
        super().__init__()
        self.rr_features = rr_features
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + rr_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, class_count),
        )

    def forward(self, waveforms: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(waveforms.unsqueeze(1)).squeeze(-1)
        if self.rr_features:
            encoded = torch.cat([encoded, rr], dim=1)
        return self.head(encoded)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class RrScaler:
    columns: list[int]
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, beats: BeatSet, columns: list[int]) -> "RrScaler":
        values = beats.rr_features[:, columns].astype(np.float64)
        mean = values.mean(axis=0) if columns else np.zeros(0)
        std = values.std(axis=0) if columns else np.ones(0)
        return cls(columns, mean.tolist(), np.where(std > 0, std, 1.0).tolist())

    def transform(self, beats: BeatSet) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(beats), 0), dtype=np.float32)
        values = beats.rr_features[:, self.columns].astype(np.float64)
        return ((values - np.array(self.mean)) / np.array(self.std)).astype(np.float32)


def class_weights(labels: np.ndarray, class_count: int = len(AAMI_CLASSES)) -> np.ndarray:
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return weights / weights.mean()


def seed_everything(seed: int, device: torch.device) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def to_tensors(
    beats: BeatSet, scaler: RrScaler, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(beats.waveforms)).to(device),
        torch.from_numpy(scaler.transform(beats)).to(device),
        torch.from_numpy(beats.labels).to(device),
    )


@torch.no_grad()
def predict_logits(
    model: BeatCnn,
    waveforms: torch.Tensor,
    rr: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, waveforms.shape[0], batch_size):
        outputs.append(model(waveforms[start : start + batch_size], rr[start : start + batch_size]))
    return torch.cat(outputs)


def predict(model: BeatCnn, beats: BeatSet, scaler: RrScaler, device: torch.device) -> np.ndarray:
    waveforms, rr, _ = to_tensors(beats, scaler, device)
    return predict_logits(model, waveforms, rr).argmax(dim=1).cpu().numpy().astype(np.int64)


def augment(
    waveforms: torch.Tensor, settings: RunSettings, generator: torch.Generator
) -> torch.Tensor:
    device = waveforms.device
    scale = torch.empty(waveforms.shape[0], 1, device=device).uniform_(
        settings.amplitude_low, settings.amplitude_high, generator=generator
    )
    noise = torch.randn(waveforms.shape, generator=generator, device=device) * settings.noise_std_mv
    return waveforms * scale + noise


def train_model(
    settings: RunSettings,
    train: BeatSet,
    validation: BeatSet,
    device: torch.device,
) -> tuple[BeatCnn, RrScaler, list[dict], int]:
    generator = seed_everything(settings.seed, device)
    scaler = RrScaler.fit(train, settings.rr_columns())
    model = BeatCnn(len(scaler.columns), settings.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    weights = torch.tensor(class_weights(train.labels), dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    train_waveforms, train_rr, train_labels = to_tensors(train, scaler, device)
    validation_waveforms, validation_rr, validation_labels = to_tensors(validation, scaler, device)
    history = []
    best_state = None
    best_epoch = -1
    best_macro_f1 = -1.0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(train_labels.shape[0], generator=generator, device=device)
        total_loss = 0.0
        for start in range(0, order.shape[0], settings.batch_size):
            batch = order[start : start + settings.batch_size]
            logits = model(augment(train_waveforms[batch], settings, generator), train_rr[batch])
            loss = loss_function(logits, train_labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.shape[0]
        validation_logits = predict_logits(model, validation_waveforms, validation_rr)
        validation_loss = float(loss_function(validation_logits, validation_labels).item())
        validation_predicted = validation_logits.argmax(dim=1).cpu().numpy()
        validation_scores = score(validation.labels, validation_predicted)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / order.shape[0],
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_scores.macro_f1,
                "validation_accuracy": validation_scores.accuracy,
            }
        )
        if validation_scores.macro_f1 > best_macro_f1:
            best_macro_f1 = validation_scores.macro_f1
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history, best_epoch


def checkpoint_payload(
    model: BeatCnn, scaler: RrScaler, settings: RunSettings, best_epoch: int, history: list[dict]
) -> dict:
    return {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "scaler": asdict(scaler),
        "settings": asdict(settings),
        "best_epoch": best_epoch,
        "history": history,
        "parameters": parameter_count(model),
    }


def load_checkpoint(path, device: torch.device) -> tuple[BeatCnn, RrScaler, RunSettings, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    settings = RunSettings(**payload["settings"])
    scaler = RrScaler(**payload["scaler"])
    model = BeatCnn(len(scaler.columns), settings.dropout).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, scaler, settings, payload
