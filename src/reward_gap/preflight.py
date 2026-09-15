"""Check prepared inputs and real inference before spending time on training."""

import gc
import platform
import tempfile
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch.version import cuda as torch_cuda_version

from reward_gap.artifacts import atomic_write_json
from reward_gap.config import ExperimentConfig
from reward_gap.experiment import FollowupExperiment
from reward_gap.formatting import format_policy_batch
from reward_gap.policy import PPOActor
from reward_gap.scorers import RewardScorer


class PreflightError(ValueError):
    """A failed check, with the saved report location in the message."""


@dataclass(frozen=True)
class PreflightResult:
    report_path: Path
    state: str


def preflight(config: ExperimentConfig, *, actor_factory=None, scorer_factory=None) -> PreflightResult:
    """Generate one short training batch and score it without PPO or final labels.

    Each call gets a separate report directory. Factories support offline tests;
    normal calls load the configured models with the configured download policy.
    A pass verifies inference, not the peak memory requirement of PPO training.
    """
    root = config.runtime.output_root
    if not root.is_absolute():
        raise PreflightError("Use absolute output paths from load_config")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="preflight-", dir=root))
    report_path = folder / "report.json"
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {"schema_version": 1, "state": "running", "config": config.to_dict(),
              "python": platform.python_version(), "packages": packages, "checks": [],
              "scope": "input validation and inference only; no PPO update or final-evaluation labels"}
    actor = proxy = judge = None
    stage = "output_directory"
    def passed(name, details):
        report["checks"].append({"name": name, "state": "passed", "details": details})
        atomic_write_json(report_path, report)
    try:
        passed(stage, str(folder))  # Exercises the actual atomic artifact writer.
        stage = "prepared_data"
        experiment = FollowupExperiment(config, root / "preflight-validation")
        experiment.validate_inputs()
        passed(stage, {name: len(rows) for name, rows in experiment.cohorts.items()})

        stage = "device"
        device = torch.device(config.runtime.device)
        device_info = {"requested": str(device), "torch_cuda_build": torch_cuda_version}
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise PreflightError("CUDA is unavailable in this Python environment. "
                                     "Run on the GPU machine with a CUDA-enabled PyTorch build.")
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise PreflightError(f"CUDA device {index} does not exist")
            with torch.cuda.device(index):
                if config.runtime.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                    raise PreflightError("The selected GPU does not support bfloat16")
            device_info.update(name=torch.cuda.get_device_name(index),
                               total_memory_bytes=torch.cuda.get_device_properties(index).total_memory)
        elif device.type != "cpu":
            raise PreflightError("Preflight supports CPU or CUDA")
        passed(stage, device_info)

        stage = "policy_loading"
        actor = actor_factory(config.seeds[0]) if actor_factory else PPOActor.load(config, seed=config.seeds[0])
        if actor.tokenizer.pad_token_id == actor.tokenizer.eos_token_id:
            raise PreflightError("TRL PPO requires distinct padding and EOS token IDs")
        passed(stage, {"source": actor.source, "revision": actor.revision,
                       "effective_eos_ids": list(actor.eos_ids),
                       "response_contract": "primary-eos-contiguous-nonpad-v1"})

        stage = "prompt_formatting"
        # Check every prepared prompt, including template overhead and the full
        # configured answer reservation. No held-out answers are generated.
        for rows in experiment.cohorts.values():
            for record in rows:
                format_policy_batch(actor.tokenizer, [record],
                                    max_prompt_tokens=config.generation.max_prompt_tokens,
                                    max_new_tokens=config.generation.max_new_tokens,
                                    context_window=actor.context_window)
        passed(stage, "All prepared prompts fit the configured policy token limits")

        stage = "scorer_loading"
        load_scorer = scorer_factory or (lambda role: RewardScorer.load(config, role))
        proxy, judge = load_scorer("proxy"), load_scorer("judge")
        passed(stage, {role: {"source": scorer.loaded.source, "revision": scorer.loaded.revision}
                       for role, scorer in (("proxy", proxy), ("judge", judge))})

        stage = "generation_and_scoring"
        prompts = experiment.schedules[config.seeds[0]][0]
        original_generation = actor.generation
        actor.generation = replace(original_generation, max_new_tokens=min(16, original_generation.max_new_tokens))
        try:
            rollout = actor.generate(prompts, seed=config.seeds[0])
        finally:
            actor.generation = original_generation
        proxy_scores = proxy.score(prompts, rollout.answers, return_embeddings=True)
        judge_scores = judge.score(prompts, rollout.answers)
        expected = tuple(p.prompt_id for p in prompts)
        if rollout.prompt_ids != expected or proxy_scores.prompt_ids != expected or judge_scores.prompt_ids != expected:
            raise PreflightError("Generation/scoring changed prompt alignment")
        if proxy_scores.embeddings is None:
            raise PreflightError("Proxy scorer returned no memory embeddings")
        passed(stage, {"prompt_ids": list(expected), "answers": list(rollout.answers),
                       "response_lengths": list(rollout.response_lengths),
                       "proxy_scores": list(proxy_scores.scores), "judge_scores": list(judge_scores.scores),
                       "embedding_shape": list(proxy_scores.embeddings.shape)})
        report["state"] = "passed"
        atomic_write_json(report_path, report)
        return PreflightResult(report_path, "passed")
    except Exception as exc:
        report["state"] = "failed"
        report["checks"].append({"name": stage, "state": "failed", "error": str(exc)})
        atomic_write_json(report_path, report)
        raise PreflightError(f"{stage}: {exc}\nReport: {report_path}") from exc
    finally:
        del actor, proxy, judge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
