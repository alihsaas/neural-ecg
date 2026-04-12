from dataclasses import dataclass
from pathlib import Path

from neural_ecg.artifacts import load_protocol
from neural_ecg.beats import BeatWindow
from neural_ecg.conditions import SignalCondition
from neural_ecg.config import DS1_RECORDS, DS2_RECORDS, PACED_RECORDS, PROJECT_ROOT

DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "beat_classification" / "protocol.json"


@dataclass(frozen=True)
class ClassificationProtocol:
    values: dict

    @classmethod
    def load(cls, path: Path = DEFAULT_PROTOCOL) -> "ClassificationProtocol":
        protocol = cls(load_protocol(path))
        protocol.validate()
        return protocol

    @property
    def train_records(self) -> tuple[str, ...]:
        return tuple(self.values["train_records"])

    @property
    def validation_records(self) -> tuple[str, ...]:
        return tuple(self.values["validation_records"])

    @property
    def test_records(self) -> tuple[str, ...]:
        return tuple(self.values["test_records"])

    @property
    def beat_window(self) -> BeatWindow:
        return BeatWindow(**self.values["beat_window"])

    @property
    def robustness_conditions(self) -> list[SignalCondition]:
        return [SignalCondition.from_dict(item) for item in self.values["robustness_conditions"]]

    def path(self, key: str) -> Path:
        return PROJECT_ROOT / self.values["paths"][key]

    def validate(self) -> None:
        train = set(self.train_records)
        validation = set(self.validation_records)
        test = set(self.test_records)
        if train & validation or train & test or validation & test:
            raise ValueError("Train, validation and test records must not overlap")
        if (train | validation) != set(DS1_RECORDS):
            raise ValueError("Train and validation records must together be exactly DS1")
        if test != set(DS2_RECORDS):
            raise ValueError("Test records must be exactly DS2")
        if (train | validation | test) & set(PACED_RECORDS):
            raise ValueError("Paced records are excluded from classification")
