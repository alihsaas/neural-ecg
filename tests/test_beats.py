import numpy as np
import pytest
from conftest import requires_data, synthetic_ecg

from neural_ecg.beats import RR_FEATURE_NAMES, BeatWindow, aami_label, extract_beats, rr_features
from neural_ecg.config import (
    AAMI_CLASSES,
    ALL_RECORDS,
    DS1_RECORDS,
    DS2_RECORDS,
    PACED_RECORDS,
)
from neural_ecg.protocol import ClassificationProtocol
from neural_ecg.records import EcgRecord, load_record

FS = 360.0


def test_aami_mapping_follows_the_standard_groups() -> None:
    expected = {
        "N": "N",
        "L": "N",
        "R": "N",
        "e": "N",
        "j": "N",
        "A": "S",
        "a": "S",
        "J": "S",
        "S": "S",
        "V": "V",
        "E": "V",
        "F": "F",
    }
    for symbol, group in expected.items():
        assert AAMI_CLASSES[aami_label(symbol)] == group
    for excluded in ("/", "f", "Q", "+", "~", "|"):
        assert aami_label(excluded) == -1


def test_inter_patient_split_is_disjoint_and_excludes_paced_records() -> None:
    assert len(DS1_RECORDS) == 22 and len(DS2_RECORDS) == 22
    assert not set(DS1_RECORDS) & set(DS2_RECORDS)
    assert not (set(DS1_RECORDS) | set(DS2_RECORDS)) & set(PACED_RECORDS)
    assert set(DS1_RECORDS) | set(DS2_RECORDS) | set(PACED_RECORDS) == set(ALL_RECORDS)
    protocol = ClassificationProtocol.load()
    assert not set(protocol.train_records) & set(protocol.validation_records)


def test_rr_features_for_a_premature_beat() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 3.6, 5.0, 6.0])
    features = rr_features(np.round(times * FS).astype(int), FS, BeatWindow(local_beats=3))
    column = {name: index for index, name in enumerate(RR_FEATURE_NAMES)}
    assert np.isnan(features[0, column["pre_rr_s"]])
    assert np.isnan(features[-1, column["post_rr_s"]])
    np.testing.assert_allclose(features[4, column["pre_rr_s"]], 0.6, atol=1 / FS)
    np.testing.assert_allclose(features[4, column["post_rr_s"]], 1.4, atol=1 / FS)
    np.testing.assert_allclose(
        features[4, column["local_rr_s"]], (1.0 + 1.0 + 0.6) / 3, atol=1 / FS
    )
    np.testing.assert_allclose(features[4, column["long_rr_s"]], 3.6 / 4, atol=1 / FS)
    assert features[4, column["pre_over_long"]] < 0.7
    assert features[4, column["post_over_long"]] > 1.5


def test_extract_beats_skips_edges_and_unclassified_symbols() -> None:
    times = 0.1 + 0.8 * np.arange(12)
    samples = np.round(times * FS).astype(np.int64)
    symbols = np.array(["N"] * 12)
    symbols[5] = "V"
    symbols[6] = "Q"
    signal = synthetic_ecg(times, times[-1] + 0.2)
    record = EcgRecord("999", FS, "MLII", signal, samples, symbols)
    window = BeatWindow()
    beats = extract_beats(record, window)
    before, after = window.samples(FS)
    assert beats.waveforms.shape[1] == before + after
    assert 0 not in beats.samples.tolist()
    assert samples[6] not in beats.samples.tolist()
    assert samples[5] in beats.samples.tolist()
    assert beats.labels[beats.samples.tolist().index(samples[5])] == AAMI_CLASSES.index("V")
    assert np.all(np.isfinite(beats.rr_features))
    assert beats.waveforms.dtype == np.float32
    peak = np.argmax(beats.waveforms, axis=1)
    assert np.all(np.abs(peak - before) <= 2)


def test_protocol_rejects_leaky_splits(tmp_path) -> None:
    protocol = ClassificationProtocol.load()
    values = dict(protocol.values)
    values["validation_records"] = list(protocol.validation_records) + ["100"]
    with pytest.raises(ValueError):
        ClassificationProtocol(values).validate()


@requires_data
def test_record_114_uses_mlii_even_though_it_is_the_second_channel() -> None:
    assert load_record("114").lead == "MLII"
    assert load_record("102").lead == "V5"


@requires_data
def test_record_100_beat_counts_match_the_annotations() -> None:
    beats = extract_beats(load_record("100"))
    assert beats.class_counts() == {"N": 2237, "S": 33, "V": 1, "F": 0}
