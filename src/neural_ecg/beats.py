from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from neural_ecg.config import AAMI_CLASSES, AAMI_SYMBOL_MAP
from neural_ecg.filters import zero_phase_bandpass
from neural_ecg.records import EcgRecord

RR_FEATURE_NAMES = (
    "pre_rr_s",
    "post_rr_s",
    "local_rr_s",
    "long_rr_s",
    "pre_over_long",
    "post_over_long",
    "local_over_long",
)

NORMALIZED_RR_COLUMNS = (4, 5, 6)


@dataclass(frozen=True)
class BeatWindow:
    before_s: float = 0.25
    after_s: float = 0.40
    local_beats: int = 10
    long_window_s: float = 300.0
    band_low_hz: float = 0.5
    band_high_hz: float = 40.0

    def samples(self, sampling_rate_hz: float) -> tuple[int, int]:
        return round(self.before_s * sampling_rate_hz), round(self.after_s * sampling_rate_hz)


@dataclass(frozen=True)
class BeatSet:
    waveforms: np.ndarray
    rr_features: np.ndarray
    labels: np.ndarray
    records: np.ndarray
    samples: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.size)

    def subset(self, mask: np.ndarray) -> "BeatSet":
        return BeatSet(
            self.waveforms[mask],
            self.rr_features[mask],
            self.labels[mask],
            self.records[mask],
            self.samples[mask],
        )

    def class_counts(self) -> dict[str, int]:
        return {name: int(np.sum(self.labels == index)) for index, name in enumerate(AAMI_CLASSES)}


def aami_label(symbol: str) -> int:
    name = AAMI_SYMBOL_MAP.get(symbol)
    return -1 if name is None else AAMI_CLASSES.index(name)


def rr_features(
    beat_samples: np.ndarray, sampling_rate_hz: float, window: BeatWindow
) -> np.ndarray:
    times = np.asarray(beat_samples, dtype=np.float64) / sampling_rate_hz
    count = times.size
    features = np.full((count, len(RR_FEATURE_NAMES)), np.nan)
    if count < 3:
        return features
    pre = np.full(count, np.nan)
    post = np.full(count, np.nan)
    pre[1:] = np.diff(times)
    post[:-1] = np.diff(times)
    cumulative = np.concatenate(([0.0], np.cumsum(np.nan_to_num(pre[1:]))))
    local = np.full(count, np.nan)
    long = np.full(count, np.nan)
    for index in range(1, count):
        first_local = max(1, index - window.local_beats + 1)
        local[index] = (cumulative[index] - cumulative[first_local - 1]) / (index - first_local + 1)
        first_long = int(np.searchsorted(times, times[index] - window.long_window_s, side="left"))
        first_long = max(1, first_long)
        if first_long > index:
            first_long = index
        long[index] = (cumulative[index] - cumulative[first_long - 1]) / (index - first_long + 1)
    features[:, 0] = pre
    features[:, 1] = post
    features[:, 2] = local
    features[:, 3] = long
    features[:, 4] = pre / long
    features[:, 5] = post / long
    features[:, 6] = local / long
    return features


def extract_beats(record: EcgRecord, window: BeatWindow | None = None) -> BeatSet:
    window = window or BeatWindow()
    fs = record.sampling_rate_hz
    filtered = zero_phase_bandpass(record.signal_mv, fs, window.band_low_hz, window.band_high_hz)
    beat_samples, beat_symbols = record.beat_annotations()
    features = rr_features(beat_samples, fs, window)
    before, after = window.samples(fs)
    labels = np.array([aami_label(symbol) for symbol in beat_symbols], dtype=np.int64)
    inside = (beat_samples - before >= 0) & (beat_samples + after <= filtered.size)
    keep = (labels >= 0) & inside & np.all(np.isfinite(features), axis=1)
    kept_samples = beat_samples[keep]
    offsets = np.arange(-before, after)
    waveforms = filtered[kept_samples[:, None] + offsets[None, :]]
    waveforms = waveforms - np.median(waveforms, axis=1, keepdims=True)
    return BeatSet(
        waveforms=waveforms.astype(np.float32),
        rr_features=features[keep].astype(np.float32),
        labels=labels[keep],
        records=np.full(kept_samples.size, record.name),
        samples=kept_samples.astype(np.int64),
    )


def concatenate_beats(beat_sets: Iterable[BeatSet]) -> BeatSet:
    sets = list(beat_sets)
    if not sets:
        raise ValueError("At least one beat set is required")
    return BeatSet(
        np.concatenate([item.waveforms for item in sets]),
        np.concatenate([item.rr_features for item in sets]),
        np.concatenate([item.labels for item in sets]),
        np.concatenate([item.records for item in sets]),
        np.concatenate([item.samples for item in sets]),
    )
