from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from neural_ecg.config import BEAT_SYMBOLS, DEFAULT_DATA_DIR


@dataclass(frozen=True)
class EcgRecord:
    name: str
    sampling_rate_hz: float
    lead: str
    signal_mv: np.ndarray
    annotation_samples: np.ndarray
    annotation_symbols: np.ndarray

    def beat_annotations(self) -> tuple[np.ndarray, np.ndarray]:
        is_beat = np.isin(self.annotation_symbols, list(BEAT_SYMBOLS))
        return self.annotation_samples[is_beat], self.annotation_symbols[is_beat]

    def with_signal(self, signal_mv: np.ndarray) -> "EcgRecord":
        if signal_mv.shape != self.signal_mv.shape:
            raise ValueError("Replacement signal must keep the record length")
        return EcgRecord(
            self.name,
            self.sampling_rate_hz,
            self.lead,
            np.asarray(signal_mv, dtype=np.float64),
            self.annotation_samples,
            self.annotation_symbols,
        )


def choose_lead(lead_names: list[str], preferred_lead: str) -> int:
    if preferred_lead in lead_names:
        return lead_names.index(preferred_lead)
    return 0


def load_record(
    name: str, data_dir: Path = DEFAULT_DATA_DIR, preferred_lead: str = "MLII"
) -> EcgRecord:
    path = str(Path(data_dir) / name)
    raw = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    lead_index = choose_lead(list(raw.sig_name), preferred_lead)
    signal = np.asarray(raw.p_signal[:, lead_index], dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Record {name} has non-finite samples")
    return EcgRecord(
        name=name,
        sampling_rate_hz=float(raw.fs),
        lead=raw.sig_name[lead_index],
        signal_mv=signal,
        annotation_samples=np.asarray(annotation.sample, dtype=np.int64),
        annotation_symbols=np.asarray(annotation.symbol, dtype=str),
    )


def data_available(data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    return (Path(data_dir) / "100.hea").exists()
