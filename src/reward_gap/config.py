"""Validated experiment and HH-RLHF preparation settings; models come later."""

import json
import math
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path


class ConfigError(ValueError):
    """An experiment configuration is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    round1_updates: int = 2
    total_updates: int = 4
    rollout_batch_size: int = 2
    learning_rate: float = 3e-6
    kl_coefficient: float = 0.05
    checkpoint_every: int = 2


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    output_root: Path = Path("outputs")
    allow_downloads: bool = False


HH_SUBSETS = ("helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base")
COHORTS = ("calibration", "initial_memory", "training", "refresh", "validation", "final_evaluation")


@dataclass(frozen=True)
class DataConfig:
    revision: str = "main"
    subsets: tuple[str, ...] = HH_SUBSETS
    split_seed: int = 42
    cache_dir: Path = Path("data/raw/hh-rlhf")
    prepared_dir: Path = Path("data/prepared/smoke")
    minimum_prompts: dict[str, int] = field(default_factory=lambda: {
        "calibration": 16, "initial_memory": 32, "training": 32,
        "refresh": 16, "validation": 8, "final_evaluation": 16,
    })


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: str
    seeds: tuple[int, ...]
    training: TrainingConfig
    runtime: RuntimeConfig
    data: DataConfig | None = None

    def to_dict(self) -> dict:
        """Return JSON-compatible settings including all resolved defaults."""
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["runtime"]["output_root"] = str(self.runtime.output_root)
        if self.data is not None:
            result["data"]["subsets"] = list(self.data.subsets)
            for name in ("cache_dir", "prepared_dir"):
                result["data"][name] = str(getattr(self.data, name))
        return result


def _object(value, allowed: set[str], location: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{location}: expected a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{location}: unknown fields: {', '.join(sorted(unknown))}")
    return value


def _integer(value, location: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{location}: expected an integer >= {minimum}")


def _number(value, location: str, *, allow_zero: bool = False) -> None:
    if type(value) not in (int, float):
        raise ConfigError(f"{location}: expected a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (value == 0 and not allow_zero):
        comparison = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"{location}: expected a finite number {comparison}")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load JSON; resolve output paths against its nearest project root.

    This first schema supports configuration checks only. It does not yet
    describe a complete PPO experiment or load any models.
    """
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigError(f"Cannot read configuration {config_path}: {exc}") from exc

    raw = _object(raw, {f.name for f in fields(ExperimentConfig)}, "config")
    for required in ("schema_version", "experiment", "seeds"):
        if required not in raw:
            raise ConfigError(f"config: missing required field '{required}'")
    _integer(raw["schema_version"], "schema_version")
    if raw["schema_version"] != 1:
        raise ConfigError("schema_version: only version 1 is supported")
    if raw["experiment"] not in ("smoke", "followup"):
        raise ConfigError("experiment: expected 'smoke' or 'followup'")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("seeds: expected a nonempty list of integers")
    for index, seed in enumerate(seeds):
        _integer(seed, f"seeds[{index}]", minimum=0)
    if len(set(seeds)) != len(seeds):
        raise ConfigError("seeds: duplicate seeds are not allowed")

    training = TrainingConfig(**_object(
        raw.get("training", {}), {f.name for f in fields(TrainingConfig)}, "training"
    ))
    for name in ("round1_updates", "total_updates", "rollout_batch_size", "checkpoint_every"):
        _integer(getattr(training, name), f"training.{name}")
    _number(training.learning_rate, "training.learning_rate")
    _number(training.kl_coefficient, "training.kl_coefficient", allow_zero=True)
    if training.round1_updates >= training.total_updates:
        raise ConfigError("training.round1_updates: must be less than total_updates")

    runtime = RuntimeConfig(**_object(
        raw.get("runtime", {}), {f.name for f in fields(RuntimeConfig)}, "runtime"
    ))
    if runtime.device not in ("cpu", "cuda"):
        raise ConfigError("runtime.device: expected 'cpu' or 'cuda'")
    if type(runtime.allow_downloads) is not bool:
        raise ConfigError("runtime.allow_downloads: expected true or false")
    output = runtime.output_root
    if not isinstance(output, (str, Path)) or not str(output).strip():
        raise ConfigError("runtime.output_root: expected a nonempty path string")
    root = next((p for p in config_path.parents if (p / "pyproject.toml").is_file()), None)
    if root is None:
        raise ConfigError("Cannot locate project root: no parent pyproject.toml found")
    runtime = RuntimeConfig(runtime.device, (root / output).resolve(), runtime.allow_downloads)
    data = None
    if raw.get("data") is not None:
        values = _object(raw["data"], {f.name for f in fields(DataConfig)}, "data")
        data = DataConfig(**values)
        if not isinstance(data.revision, str) or not data.revision.strip():
            raise ConfigError("data.revision: expected a nonempty revision")
        if not isinstance(data.subsets, (list, tuple)) or not data.subsets:
            raise ConfigError("data.subsets: expected a nonempty list")
        if any(s not in HH_SUBSETS for s in data.subsets):
            raise ConfigError("data.subsets: unsupported HH-RLHF preference subset")
        if len(set(data.subsets)) != len(data.subsets):
            raise ConfigError("data.subsets: duplicate subsets")
        _integer(data.split_seed, "data.split_seed", minimum=0)
        counts = _object(data.minimum_prompts, set(COHORTS), "data.minimum_prompts")
        if set(counts) != set(COHORTS):
            raise ConfigError("data.minimum_prompts: provide all six cohort counts")
        for name, count in counts.items():
            _integer(count, f"data.minimum_prompts.{name}", minimum=0 if name == "validation" else 1)
        paths = {}
        for name in ("cache_dir", "prepared_dir"):
            value = getattr(data, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ConfigError(f"data.{name}: expected a nonempty path")
            paths[name] = (root / value).resolve()
        data = DataConfig(data.revision, tuple(sorted(data.subsets)), data.split_seed,
                          paths["cache_dir"], paths["prepared_dir"], dict(counts))
    return ExperimentConfig(1, raw["experiment"], tuple(seeds), training, runtime, data)
