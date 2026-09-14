"""Validated experiment, model-reference, and HH-RLHF preparation settings."""

import json
import math
import re
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Literal

ModelDtype = Literal["float32", "float16", "bfloat16"]


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
    ppo_epochs: int = 4
    minibatch_size: int = 1
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coefficient: float = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True

    def __post_init__(self):
        for name in ("round1_updates", "total_updates", "rollout_batch_size", "checkpoint_every",
                     "ppo_epochs", "minibatch_size"):
            _integer(getattr(self, name), f"training.{name}")
        for name in ("learning_rate", "clip_range", "value_clip_range", "max_grad_norm"):
            _number(getattr(self, name), f"training.{name}")
        for name in ("kl_coefficient", "value_coefficient", "gamma", "gae_lambda"):
            _number(getattr(self, name), f"training.{name}", allow_zero=True)
        if self.clip_range >= 1 or self.gamma > 1 or self.gae_lambda > 1:
            raise ConfigError("training: clip_range must be < 1; gamma and gae_lambda must be <= 1")
        if self.minibatch_size > self.rollout_batch_size:
            raise ConfigError("training.minibatch_size: must not exceed rollout_batch_size")
        if self.rollout_batch_size < 2 or self.rollout_batch_size % self.minibatch_size:
            raise ConfigError("training: TRL requires rollout_batch_size >= 2 and divisible by minibatch_size")
        if type(self.normalize_advantages) is not bool:
            raise ConfigError("training.normalize_advantages: expected true or false")
        if not self.normalize_advantages:
            raise ConfigError("training.normalize_advantages: TRL PPO requires true")
        if self.round1_updates >= self.total_updates:
            raise ConfigError("training.round1_updates: must be less than total_updates")


@dataclass(frozen=True)
class GenerationConfig:
    max_prompt_tokens: int = 512
    max_new_tokens: int = 256
    do_sample: bool = True


@dataclass(frozen=True)
class PolicyConfig:
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class ScoringConfig:
    max_tokens: int = 4096
    batch_size: int = 4


@dataclass(frozen=True)
class MemoryConfig:
    k: int = 8
    temperature: float = 0.1

    def __post_init__(self):
        _integer(self.k, "memory.k")
        _number(self.temperature, "memory.temperature")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    output_root: Path = Path("outputs")
    allow_downloads: bool = False
    dtype: ModelDtype = "float32"
    model_cache: Path = Path("model_cache")


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
class ModelReference:
    id: str
    revision: str = "main"


@dataclass(frozen=True)
class ModelsConfig:
    policy: ModelReference
    proxy: ModelReference
    judge: ModelReference


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: str
    seeds: tuple[int, ...]
    training: TrainingConfig
    runtime: RuntimeConfig
    data: DataConfig | None = None
    models: ModelsConfig | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def to_dict(self) -> dict:
        """Return JSON-compatible settings including all resolved defaults."""
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["runtime"]["output_root"] = str(self.runtime.output_root)
        result["runtime"]["model_cache"] = str(self.runtime.model_cache)
        result["policy"]["target_modules"] = list(self.policy.target_modules)
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


def _model_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ConfigError(f"{location}: expected a nonempty string without surrounding whitespace")
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
    runtime = RuntimeConfig(**_object(
        raw.get("runtime", {}), {f.name for f in fields(RuntimeConfig)}, "runtime"
    ))
    if not isinstance(runtime.device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", runtime.device):
        raise ConfigError("runtime.device: expected 'cpu', 'cuda', or 'cuda:<index>'")
    if runtime.dtype not in ("float32", "float16", "bfloat16"):
        raise ConfigError("runtime.dtype: expected float32, float16, or bfloat16")
    if runtime.device == "cpu" and runtime.dtype != "float32":
        raise ConfigError("runtime.dtype: CPU loading requires float32")
    if type(runtime.allow_downloads) is not bool:
        raise ConfigError("runtime.allow_downloads: expected true or false")
    output = runtime.output_root
    if not isinstance(output, (str, Path)) or not str(output).strip():
        raise ConfigError("runtime.output_root: expected a nonempty path string")
    root = next((p for p in config_path.parents if (p / "pyproject.toml").is_file()), None)
    if root is None:
        raise ConfigError("Cannot locate project root: no parent pyproject.toml found")
    if not isinstance(runtime.model_cache, (str, Path)) or not str(runtime.model_cache).strip():
        raise ConfigError("runtime.model_cache: expected a nonempty path")
    runtime = RuntimeConfig(runtime.device, (root / output).resolve(), runtime.allow_downloads,
                            runtime.dtype, (root / runtime.model_cache).resolve())
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
    models = None
    if raw.get("models") is not None:
        roles = {"policy", "proxy", "judge"}
        model_values = _object(raw["models"], roles, "models")
        if set(model_values) != roles:
            raise ConfigError("models: provide policy, proxy, and judge references")
        references = {}
        for role in ("policy", "proxy", "judge"):
            location = f"models.{role}"
            values = _object(model_values[role], {"id", "revision"}, location)
            model_id = _model_string(values.get("id"), f"{location}.id")
            revision = _model_string(values.get("revision", "main"), f"{location}.revision")
            references[role] = ModelReference(model_id, revision)
        models = ModelsConfig(**references)
    generation = GenerationConfig(**_object(
        raw.get("generation", {}), {f.name for f in fields(GenerationConfig)}, "generation"))
    _integer(generation.max_prompt_tokens, "generation.max_prompt_tokens")
    _integer(generation.max_new_tokens, "generation.max_new_tokens")
    if type(generation.do_sample) is not bool:
        raise ConfigError("generation.do_sample: expected true or false")
    scoring = ScoringConfig(**_object(raw.get("scoring", {}), {f.name for f in fields(ScoringConfig)}, "scoring"))
    _integer(scoring.max_tokens, "scoring.max_tokens")
    _integer(scoring.batch_size, "scoring.batch_size")
    if scoring.max_tokens > 16384:
        raise ConfigError("scoring.max_tokens: Skywork scoring limit is 16384")
    policy = PolicyConfig(**_object(raw.get("policy", {}), {f.name for f in fields(PolicyConfig)}, "policy"))
    _integer(policy.lora_rank, "policy.lora_rank")
    _integer(policy.lora_alpha, "policy.lora_alpha")
    targets = policy.target_modules
    if (not isinstance(targets, (list, tuple)) or not targets
            or any(t not in PolicyConfig().target_modules for t in targets)):
        raise ConfigError("policy.target_modules: expected supported Qwen projection names")
    if len(set(targets)) != len(targets):
        raise ConfigError("policy.target_modules: duplicate modules")
    policy = PolicyConfig(policy.lora_rank, policy.lora_alpha, tuple(targets))
    memory = MemoryConfig(**_object(raw.get("memory", {}), {f.name for f in fields(MemoryConfig)}, "memory"))
    return ExperimentConfig(1, raw["experiment"], tuple(seeds), training, runtime, data, models,
                            generation, scoring, policy, memory)
