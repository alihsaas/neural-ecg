import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_ecg.beats import BeatSet, BeatWindow, concatenate_beats, extract_beats
from neural_ecg.config import DEFAULT_DATA_DIR, PROJECT_ROOT
from neural_ecg.noise import add_baseline_wander, add_white_noise, condition_rng
from neural_ecg.records import EcgRecord, load_record


@dataclass(frozen=True)
class SignalCondition:
    name: str
    index: int
    kind: str = "clean"
    snr_db: float | None = None
    amplitude_mv: float | None = None
    frequency_hz: float | None = None

    @classmethod
    def from_dict(cls, values: dict) -> "SignalCondition":
        return cls(**values)

    def apply(self, record: EcgRecord, root_seed: int) -> EcgRecord:
        if self.kind == "clean":
            return record
        rng = condition_rng(root_seed, record.name, self.index)
        if self.kind == "white":
            if self.snr_db is None:
                raise ValueError(f"Condition {self.name} needs snr_db")
            return record.with_signal(add_white_noise(record.signal_mv, self.snr_db, rng))
        if self.kind == "wander":
            if self.amplitude_mv is None or self.frequency_hz is None:
                raise ValueError(f"Condition {self.name} needs amplitude_mv and frequency_hz")
            noisy = add_baseline_wander(
                record.signal_mv, record.sampling_rate_hz, self.amplitude_mv, self.frequency_hz, rng
            )
            return record.with_signal(noisy)
        raise ValueError(f"Unknown condition kind: {self.kind}")


CLEAN = SignalCondition("clean", 0)
BEAT_CACHE_DIR = PROJECT_ROOT / ".runtime" / "beat_cache"
BEAT_FIELDS = ("waveforms", "rr_features", "labels", "records", "samples")


def load_beats(
    records: Iterable[str],
    condition: SignalCondition = CLEAN,
    root_seed: int = 0,
    window: BeatWindow | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> BeatSet:
    return concatenate_beats(
        extract_beats(condition.apply(load_record(name, data_dir), root_seed), window)
        for name in records
    )


def cache_key(
    records: Iterable[str], condition: SignalCondition, root_seed: int, window: BeatWindow
) -> str:
    text = repr((tuple(records), condition, root_seed, window))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:20]


def cached_beats(
    records: Iterable[str],
    condition: SignalCondition = CLEAN,
    root_seed: int = 0,
    window: BeatWindow | None = None,
    cache_dir: Path = BEAT_CACHE_DIR,
) -> BeatSet:
    records = tuple(records)
    window = window or BeatWindow()
    path = Path(cache_dir) / f"{cache_key(records, condition, root_seed, window)}.npz"
    if path.exists():
        with np.load(path) as stored:
            return BeatSet(**{name: stored[name] for name in BEAT_FIELDS})
    beats = load_beats(records, condition, root_seed, window)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: getattr(beats, name) for name in BEAT_FIELDS})
    return beats
