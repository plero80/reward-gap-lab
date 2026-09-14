"""Separate GSM8K protocol settings; keep HH-RLHF configurations unchanged."""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from reward_gap.config import ExperimentConfig, load_config

COHORTS = ("calibration", "memory", "selection", "monitor", "refresh", "ppo")


@dataclass(frozen=True)
class GSMConfig:
    base: ExperimentConfig
    settings: dict

    def to_dict(self):
        return {"base": self.base.to_dict(), "gsm8k": self.settings}


def load_gsm_config(path) -> GSMConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    defaults = {"data_seed": 42, "run_seeds": [42], "updates": 100, "questions_per_update": 8,
                "responses_per_question": 2, "preparation_responses": 2,
                "monitor_every": 25, "checkpoint_every": 5, "policy_temperature": .7,
                "grading_budgets": [128, 256, 512], "grading_max_input": 4096,
                "format_penalty": .5, "length_penalty": .5, "k_values": [8, 16, 32],
                "similarity_temperatures": [.05, .1], "gap_quantile": .95,
                "evaluate_test": False, "test_limit": None,
                "dataset_revision": "main", "prepared_dir": "data/prepared/gsm8k",
                "cache_dir": "data/raw/gsm8k",
                "cohorts": {"calibration": 128, "memory": 512, "selection": 128,
                            "monitor": 128, "refresh": 512, "ppo": 5000}}
    if not isinstance(raw, dict) or set(raw) - (set(defaults) | {"base_config", "teacher_comparison"}) or "base_config" not in raw:
        raise ValueError("Invalid GSM8K configuration fields")
    root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), None)
    if root is None:
        raise ValueError("GSM8K configuration must be inside the project")
    values = {**defaults, **raw}
    base = load_config(root / values.pop("base_config"))
    for name in ("data_seed", "updates", "questions_per_update", "responses_per_question", "preparation_responses",
                 "monitor_every", "checkpoint_every", "grading_max_input"):
        if type(values[name]) is not int or values[name] < (0 if name == "data_seed" else 1):
            raise ValueError(f"Invalid GSM8K {name}")
    seeds = values["run_seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide unique nonnegative run_seeds")
    counts = values["cohorts"]
    if not isinstance(counts, dict) or set(counts) != set(COHORTS) or any(type(c) is not int or c < 1 for c in counts.values()):
        raise ValueError("Provide positive counts for all six GSM8K cohorts")
    for name in ("grading_budgets", "k_values"):
        if not isinstance(values[name], list) or not values[name] or any(type(x) is not int or x < 1 for x in values[name]):
            raise ValueError(f"Invalid {name}")
    if values["grading_budgets"] != sorted(set(values["grading_budgets"])):
        raise ValueError("Grading retry budgets must strictly increase")
    for name in ("format_penalty", "length_penalty", "policy_temperature", "gap_quantile"):
        v = values[name]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError(f"Invalid {name}")
    if values["policy_temperature"] <= 0 or not 0 < values["gap_quantile"] < 1:
        raise ValueError("Temperature must be positive and gap_quantile between 0 and 1")
    ts = values["similarity_temperatures"]
    if not isinstance(ts, list) or not ts or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in ts):
        raise ValueError("Invalid similarity_temperatures")
    if type(values["evaluate_test"]) is not bool or (values["test_limit"] is not None and
            (type(values["test_limit"]) is not int or values["test_limit"] < 1)):
        raise ValueError("Invalid test evaluation settings")
    if max(values["k_values"]) > counts["memory"] * values["preparation_responses"]:
        raise ValueError("Memory is too small for k_values")
    if counts["calibration"] * values["preparation_responses"] < 2:
        raise ValueError("Calibration needs at least two responses")
    if "teacher_comparison" in values:
        comparison = values["teacher_comparison"]
        if not isinstance(comparison, dict) or set(comparison) != {"teacher30", "k", "temperature"}:
            raise ValueError("teacher_comparison needs teacher30, k, and temperature")
        reference = comparison["teacher30"]
        if (not isinstance(reference, dict) or set(reference) != {"id", "revision"}
                or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in reference.values())):
            raise ValueError("teacher30 needs a model id and revision")
        if base.models is None or reference["id"] == base.models.judge.id:
            raise ValueError("The teacher comparison requires two different judge models")
        k, temperature = comparison["k"], comparison["temperature"]
        if type(k) is not int or not 1 <= k <= counts["memory"] * values["preparation_responses"]:
            raise ValueError("Invalid fixed teacher-comparison k")
        if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid fixed teacher-comparison temperature")
        if values["k_values"] != [k] or values["similarity_temperatures"] != [temperature]:
            raise ValueError("Primary teacher comparison requires the same single fixed k/temperature, without tuning")
    if values["questions_per_update"] > counts["ppo"]:
        raise ValueError("PPO pool is smaller than questions_per_update")
    for name in ("prepared_dir", "cache_dir"):
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"Invalid {name}")
        values[name] = str((root / values[name]).resolve())
    if not isinstance(values["dataset_revision"], str) or not values["dataset_revision"].strip():
        raise ValueError("Invalid dataset_revision")
    total = values["updates"]
    if total < 2:
        raise ValueError("At least two PPO updates are required")
    batch = values["questions_per_update"] * values["responses_per_question"]
    training = replace(base.training, total_updates=total, round1_updates=max(1, total // 2),
                       rollout_batch_size=batch, checkpoint_every=values["checkpoint_every"])
    return GSMConfig(replace(base, seeds=tuple(seeds), training=training), values)
