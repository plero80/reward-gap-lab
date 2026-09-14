"""Fit score scales on a dedicated cohort and reuse frozen constants."""

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from reward_gap.artifacts import atomic_write_json
from reward_gap.scorers import ScoreBatch


class CalibrationError(ValueError):
    """Unusable scores or incompatible frozen calibration."""


@dataclass(frozen=True)
class ScoreScale:
    source: str
    revision: str | None
    mean: float
    std: float

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise CalibrationError("Score scale requires a model source")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise CalibrationError("Revision must be a nonempty string or None for a local model")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.mean, self.std)) or self.std <= 0:
            raise CalibrationError("Calibration mean must be finite and std must be finite and positive")

    def normalize(self, batch: ScoreBatch) -> tuple[float, ...]:
        if (batch.source, batch.revision) != (self.source, self.revision):
            raise CalibrationError("Scorer source/revision differs from frozen calibration")
        _validate_scores(batch)
        values = tuple((float(score) - self.mean) / self.std for score in batch.scores)
        if not all(math.isfinite(value) for value in values):
            raise CalibrationError("Normalization produced nonfinite scores")
        return values


def _validate_scores(batch: ScoreBatch) -> None:
    if (not batch.prompt_ids or len(batch.scores) != len(batch.prompt_ids)
            or len(batch.token_counts) != len(batch.prompt_ids)):
        raise CalibrationError("Provide one score and token count per prompt")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in batch.scores):
        raise CalibrationError("Scores must be finite numbers")


@dataclass(frozen=True)
class FrozenCalibration:
    """Shared identity for a proxy/judge pair of population z-score scales.

    Fit only on the dedicated calibration cohort; never refit on PPO batches.
    Give each newly fitted artifact a distinct calibration_id. This is an
    explicit label, not a content hash or proof of cohort membership.
    """

    calibration_id: str
    proxy: ScoreScale
    judge: ScoreScale

    def __post_init__(self):
        if (not isinstance(self.calibration_id, str) or not self.calibration_id.strip()
                or self.calibration_id != self.calibration_id.strip()):
            raise CalibrationError("Provide a nonempty calibration_id without surrounding whitespace")
        if not isinstance(self.proxy, ScoreScale) or not isinstance(self.judge, ScoreScale):
            raise CalibrationError("Provide proxy and judge ScoreScale objects")

    @classmethod
    def fit(cls, proxy: ScoreBatch, judge: ScoreBatch, *, calibration_id: str) -> "FrozenCalibration":
        if proxy.role != "proxy" or judge.role != "judge" or proxy.prompt_ids != judge.prompt_ids:
            raise CalibrationError("Fit requires aligned proxy and judge batches")
        scales = []
        for batch in (proxy, judge):
            _validate_scores(batch)
            if len(batch.scores) < 2:
                raise CalibrationError("Fit requires at least two calibration examples")
            scales.append(ScoreScale(batch.source, batch.revision,
                                     statistics.mean(batch.scores), statistics.pstdev(batch.scores)))
        return cls(calibration_id, scales[0], scales[1])

    def normalize_proxy(self, batch: ScoreBatch) -> tuple[float, ...]:
        if batch.role != "proxy":
            raise CalibrationError("Expected proxy scores")
        return self.proxy.normalize(batch)

    def normalize_judge(self, batch: ScoreBatch) -> tuple[float, ...]:
        if batch.role != "judge":
            raise CalibrationError("Expected judge scores")
        return self.judge.normalize(batch)

    def save(self, path: str | Path) -> Path:
        """Atomically write constants; callers choose a separate path for each run."""
        return atomic_write_json(path, {"schema_version": 1, **asdict(self)})

    @classmethod
    def load(cls, path: str | Path) -> "FrozenCalibration":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if (not isinstance(raw, dict) or set(raw) != {"schema_version", "calibration_id", "proxy", "judge"}
                    or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
                raise CalibrationError("Invalid calibration schema")
            return cls(raw["calibration_id"], ScoreScale(**raw["proxy"]), ScoreScale(**raw["judge"]))
        except (TypeError, ValueError, KeyError) as exc:
            raise CalibrationError(f"Invalid calibration file: {exc}") from exc
