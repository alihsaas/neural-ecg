import numpy as np


def signal_power(samples: np.ndarray) -> float:
    centered = np.asarray(samples, dtype=np.float64) - float(np.mean(samples))
    return float(np.mean(centered**2))


def add_white_noise(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = signal_power(samples)
    noise_variance = power / 10.0 ** (snr_db / 10.0)
    noise = rng.normal(0.0, np.sqrt(noise_variance), size=np.shape(samples))
    return np.asarray(samples, dtype=np.float64) + noise


def add_baseline_wander(
    samples: np.ndarray,
    sampling_rate_hz: float,
    amplitude_mv: float,
    frequency_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    time_s = np.arange(np.size(samples)) / sampling_rate_hz
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wander = amplitude_mv * np.sin(2.0 * np.pi * frequency_hz * time_s + phase)
    return np.asarray(samples, dtype=np.float64) + wander


def condition_rng(root_seed: int, record_name: str, condition_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([root_seed, int(record_name), condition_index])
    return np.random.default_rng(sequence)
