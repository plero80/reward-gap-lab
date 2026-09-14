"""TRL-backed PPO with project reward adapters and update-boundary checkpoints."""

import math
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import numpy as np
from datasets import Dataset
from transformers import TrainerControl, GenerationConfig as HFGenerationConfig
from trl.experimental.ppo import PPOConfig as TRLConfig, PPOTrainer as TRLTrainer
from peft import LoraConfig

from reward_gap.config import TrainingConfig
from reward_gap.data import PromptRecord
from reward_gap.policy import PPOActor, _seeded
from reward_gap.formatting import format_policy_batch
from reward_gap._trl_bridge import RewardBridge, RewardModel, ValueModel
from reward_gap.rewards import RewardBatch
from reward_gap.failures import SampleError


class PPOError(ValueError):
    """Invalid PPO inputs, incompatible checkpoints or failed updates."""


def _resolved_device(device: torch.device) -> torch.device:
    """Resolve an implicit CUDA index before comparing placement."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _policy_identity(actor: PPOActor) -> dict:
    adapter = actor.model.peft_config["default"]
    if not isinstance(adapter, LoraConfig) or not isinstance(adapter.target_modules, set):
        raise PPOError("Checkpoint requires the actor's named LoRA target modules")
    return {"source": actor.source, "revision": actor.revision,
            "context_window": actor.context_window,
            "base_dtype": str(next(actor.model.parameters()).dtype),
            "adapter": {"rank": adapter.r, "alpha": adapter.lora_alpha,
                        "targets": sorted(adapter.target_modules)},
            "tokenizer": {"template": actor.tokenizer.chat_template,
                          "pad": actor.tokenizer.pad_token_id, "eos": actor.eos_ids,
                          "size": len(actor.tokenizer)}}


def load_policy_checkpoint(actor: PPOActor, path: str | Path) -> dict:
    """Load only policy/value weights for held-out inference, without optimizer/RNG.

    Generation limits and device may differ from training; model, tokenizer,
    precision and adapter must match. No checkpoint or training state is changed.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise PPOError("Unsupported checkpoint schema")
    identity = payload.get("identity", {})
    if (not isinstance(identity, dict) or type(payload.get("update")) is not int or payload["update"] < 0
            or type(payload.get("trainer_seed")) is not int):
        raise PPOError("Invalid inference checkpoint metadata")
    if any(identity.get(key) != value for key, value in _policy_identity(actor).items()):
        raise PPOError("Inference checkpoint model, tokenizer or adapter differs")
    parameters = {name: parameter for name, parameter in actor.named_parameters() if parameter.requires_grad}
    saved = payload.get("parameters")
    if not isinstance(saved, dict) or set(saved) != set(parameters):
        raise PPOError("Checkpoint trainable parameter names differ")
    for name, parameter in parameters.items():
        value = saved[name]
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
            raise PPOError(f"Invalid checkpoint parameter: {name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved[name])
    return {"checkpoint": str(Path(path).resolve()), "update": payload["update"],
            "trainer_seed": payload["trainer_seed"], "identity": identity}


class RewardStrategy(Protocol):
    def score(self, prompts: Sequence[PromptRecord], answers: Sequence[str]) -> RewardBatch: ...


@dataclass(frozen=True)
class UpdateMetrics:
    """A scheduled batch; skipped batches have no optimizer step or mean reward."""
    update: int
    prompt_position: int
    rollout_seed: int
    mean_reward: float | None
    library_metrics: dict[str, float]
    skipped: bool = False
    skip_reason: str | None = None


class PPOTrainer:
    """Delegates generation, GAE, losses and optimization to pinned TRL PPO.

    experiment_id labels dataset/schedule/base artifacts; reward_id labels the
    strategy, calibration and memory snapshot. Callers must use distinct labels
    for changed artifacts. Checkpoints do not contain frozen model weights.
    """

    def __init__(self, actor: PPOActor, reward: RewardStrategy, config: TrainingConfig, *,
                 experiment_id: str, reward_id: str, seed: int = 42, temperature: float = 1.0):
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise PPOError("PPO temperature must be finite and positive")
        self.temperature = float(temperature)
        if not actor.generation.do_sample:
            raise PPOError("PPO requires sampled rollouts; greedy mode is for evaluation")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise PPOError("seed must be an integer in [0, 2**63)")
        if any(not isinstance(v, str) or not v.strip() for v in (experiment_id, reward_id)):
            raise PPOError("Provide experiment_id and reward_id for checkpoint compatibility")
        if any(p.dtype == torch.float16 for p in actor.parameters()):
            raise PPOError("PPO supports float32 or bfloat16 models; float16 needs loss scaling")
        self.actor, self.reward, self.config = actor, reward, config
        self.experiment_id, self.reward_id = experiment_id, reward_id
        self._parameters = {n: p for n, p in actor.named_parameters() if p.requires_grad}
        if (not self._parameters or any("lora_" not in n and not n.startswith("value_head.")
                                       for n in self._parameters)):
            raise PPOError("Only LoRA and the value head may be trainable")
        self.optimizer = torch.optim.AdamW(list(self._parameters.values()), lr=config.learning_rate,
                                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        for group in self.optimizer.param_groups:
            group["initial_lr"] = config.learning_rate
        if config.rollout_batch_size % config.minibatch_size:
            raise PPOError("TRL requires rollout_batch_size divisible by minibatch_size")
        if not config.normalize_advantages:
            raise PPOError("TRL PPO always normalizes advantages; set normalize_advantages=true")
        if actor.tokenizer.pad_token_id == actor.tokenizer.eos_token_id:
            raise PPOError("TRL requires distinct padding and primary EOS tokens")
        self.seed = seed
        self._backend: Any = None
        self._bridge = RewardBridge(actor.tokenizer, reward, eos_ids=actor.eos_ids)
        self.update_count = 0
        self.prompt_position = 0
        self._failed = False
        self._schedule = None

    def _initialize_backend(self, batch) -> None:
        if self._backend is not None:
            return
        cfg = self.config
        requested_device = self.actor.device
        # TRL requires an output directory even with all saving/reporting off.
        # Use a temporary directory, not a path dependent on a notebook's cwd.
        self._runtime_directory = tempfile.TemporaryDirectory(prefix="reward-gap-trl-")
        setattr(self.actor.model, "generation_config", HFGenerationConfig(
            pad_token_id=self.actor.tokenizer.pad_token_id,
            eos_token_id=self.actor.tokenizer.eos_token_id,
            bos_token_id=self.actor.tokenizer.bos_token_id,
            do_sample=True, temperature=1., top_k=0, top_p=1.,
        ))
        args = TRLConfig(
            output_dir=self._runtime_directory.name,
            per_device_train_batch_size=cfg.rollout_batch_size,
            gradient_accumulation_steps=1,
            num_mini_batches=cfg.rollout_batch_size // cfg.minibatch_size,
            total_episodes=cfg.rollout_batch_size,
            local_rollout_forward_batch_size=cfg.rollout_batch_size,
            num_sample_generations=0, response_length=self.actor.generation.max_new_tokens,
            stop_token="eos", temperature=self.temperature, num_ppo_epochs=cfg.ppo_epochs,
            whiten_rewards=False, kl_coef=cfg.kl_coefficient, kl_estimator="k1",
            cliprange=cfg.clip_range, cliprange_value=cfg.value_clip_range,
            vf_coef=cfg.value_coefficient, gamma=cfg.gamma, lam=cfg.gae_lambda,
            max_grad_norm=cfg.max_grad_norm, learning_rate=cfg.learning_rate,
            lr_scheduler_type="constant", optim="adamw_torch", weight_decay=0.,
            seed=self.seed, use_cpu=self.actor.device.type == "cpu", bf16=False, fp16=False,
            gradient_checkpointing=False, report_to="none", save_strategy="no",
            eval_strategy="no", disable_tqdm=True, dataloader_pin_memory=False,
            push_to_hub=False,
        )
        dataset = Dataset.from_dict({"input_ids": batch.input_ids.cpu().tolist()})
        # TRL annotates these as PreTrainedModel, but accepts PEFT models and
        # the nn.Module adapters implementing its backbone/score interface.
        self._backend = TRLTrainer(
            args=args, processing_class=self.actor.tokenizer, model=cast(Any, self.actor.model),
            ref_model=None, reward_model=cast(Any, RewardModel(self._bridge)),
            value_model=cast(Any, ValueModel(self.actor)), train_dataset=dataset, eval_dataset=dataset,
            optimizers=(self.optimizer, torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)),
        )
        if self._backend.accelerator.num_processes != 1:
            raise PPOError("The project adapter currently supports one process/device")
        selected_device = self._backend.accelerator.device
        if _resolved_device(selected_device) != _resolved_device(requested_device):
            raise PPOError(f"Accelerate selected a different device from the actor: "
                           f"selected={selected_device}, requested={requested_device}")
        # Qwen's functional attention dropout is not an nn.Dropout module.
        # Keep sampling and teacher-forced PPO probabilities on the same policy.
        setattr(self.actor.model.config, "attention_dropout", 0.0)
        for module in self.actor.model.modules():
            if hasattr(module, "attention_dropout"):
                setattr(module, "attention_dropout", 0.0)
        # TRL/Accelerate owns optimizer wrapping and scheduling from here on.
        self.optimizer = self._backend.optimizer
        if getattr(self, "_pending_scheduler", None) is not None:
            self._backend.lr_scheduler.load_state_dict(self._pending_scheduler)
            self._pending_scheduler = None

    def update(self, prompts: Sequence[PromptRecord], *, rollout_seed: int) -> UpdateMetrics:
        if self._failed:
            raise PPOError("Previous update failed; restore a checkpoint or construct a fresh trainer")
        if self.update_count >= self.config.total_updates:
            raise PPOError("Configured total_updates already completed")
        if len(prompts) != self.config.rollout_batch_size:
            raise PPOError("Prompt count must equal rollout_batch_size")
        if self._schedule is not None and (
                [asdict(p) for p in prompts] != self._schedule["prompts"][self.update_count]
                or rollout_seed != self._schedule["seeds"][self.update_count]):
            raise PPOError("Update differs from the fixed prompt/seed schedule")
        batch = format_policy_batch(
            self.actor.tokenizer, prompts, max_prompt_tokens=self.actor.generation.max_prompt_tokens,
            max_new_tokens=self.actor.generation.max_new_tokens, context_window=self.actor.context_window,
        )
        pad_id = self.actor.tokenizer.pad_token_id
        if not isinstance(pad_id, int):
            raise PPOError("TRL requires an integer padding token ID")
        if (batch.input_ids.eq(pad_id) & batch.attention_mask.bool()).any():
            raise PPOError("TRL cannot distinguish real PAD tokens inside prompts from padding")
        numpy_state = np.random.get_state()
        try:
            # TRL owns sampling and numpy minibatch shuffling. Isolate their seed
            # while preserving reproducibility across checkpoint reconstruction.
            with _seeded(rollout_seed, self.actor.device):
                np.random.seed(rollout_seed % 2**32)
                self._initialize_backend(batch)
                self._bridge.strategy = self.reward
                self._bridge.bind(prompts, batch)
                self._backend.dataloader = [{"input_ids": batch.input_ids}]
                self._backend.control = TrainerControl()
                self._backend.state.log_history.clear()
                # Construction seeds TRL internally; the caller's rollout seed
                # must govern this update whether or not a backend was rebuilt.
                with _seeded(rollout_seed, self.actor.device):
                    np.random.seed(rollout_seed % 2**32)
                    self._backend.train()
            if any(not torch.isfinite(p).all() for p in self._parameters.values()):
                raise PPOError("TRL produced nonfinite trainable parameters")
        except SampleError as exc:
            # Our one-batch TRL call scores all rollouts before any optimizer
            # step. Only failures originating in that bridge may be skipped.
            if self._bridge.sample_failure is not exc or not getattr(self.reward, "recover_sample_failures", False):
                self._failed = True
                raise
            self.update_count += 1
            self.prompt_position += len(prompts)
            self._bridge.sample_failure = None
            return UpdateMetrics(self.update_count, self.prompt_position, rollout_seed,
                                 None, {}, skipped=True, skip_reason=str(exc))
        except BaseException:
            self._failed = True
            raise
        finally:
            np.random.set_state(numpy_state)
            self.optimizer.zero_grad(set_to_none=True)
        self.update_count += 1
        self.prompt_position += len(prompts)
        rewards = [v for result in self._bridge.batches for v in result.rewards]
        logged = self._backend.state.log_history[-1]
        metrics = {k: float(v) for k, v in logged.items()
                   if k not in ("eps", "epoch", "step") and isinstance(v, (int, float)) and math.isfinite(v)}
        return UpdateMetrics(self.update_count, self.prompt_position, rollout_seed,
                             sum(rewards) / len(rewards), metrics)

    def train(self, prompt_batches: Sequence[Sequence[PromptRecord]], rollout_seeds: Sequence[int], *,
              checkpoint_dir: str | Path | None = None, until_update: int | None = None) -> list[UpdateMetrics]:
        """Run a fixed schedule; update_count includes explicitly skipped batches."""
        if len(prompt_batches) != self.config.total_updates or len(rollout_seeds) != len(prompt_batches):
            raise PPOError("Provide the full prompt and seed schedule for total_updates")
        if any(len(batch) != self.config.rollout_batch_size for batch in prompt_batches):
            raise PPOError("All scheduled batches must match rollout_batch_size")
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in rollout_seeds):
            raise PPOError("Invalid scheduled seed")
        schedule = {"prompts": [[asdict(p) for p in batch] for batch in prompt_batches], "seeds": list(rollout_seeds)}
        if self._schedule is not None and self._schedule != schedule:
            raise PPOError("Prompt/seed schedule differs from the saved schedule")
        if self._schedule is None and self.update_count:
            raise PPOError("Cannot attach a schedule after manual updates")
        stop = self.config.total_updates if until_update is None else until_update
        if type(stop) is not int or not self.update_count <= stop <= self.config.total_updates:
            raise PPOError("Invalid stopping update")
        self._schedule = schedule
        metrics = []
        while self.update_count < stop:
            i = self.update_count
            metrics.append(self.update(prompt_batches[i], rollout_seed=rollout_seeds[i]))
            if checkpoint_dir is not None and (self.update_count % self.config.checkpoint_every == 0
                                               or self.update_count == stop):
                self.save_checkpoint(Path(checkpoint_dir) / f"update_{self.update_count:06d}.pt")
        return metrics

    def _identity(self) -> dict:
        return {"experiment_id": self.experiment_id, "reward_id": self.reward_id,
                **({"sampling_temperature": self.temperature} if self.temperature != 1.0 else {}),
                "training": asdict(self.config), "generation": asdict(self.actor.generation),
                "device": str(self.actor.device), **_policy_identity(self.actor),
                "packages": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "accelerate", "numpy")}}

    def save_checkpoint(self, path: str | Path, *, replace_existing: bool = False) -> Path:
        """Publish complete state; opt into atomic replacement for rolling recovery only."""
        if type(replace_existing) is not bool:
            raise PPOError("replace_existing must be boolean")
        if self._failed:
            raise PPOError("Cannot checkpoint a partially failed update")
        payload = {"schema_version": 2, "identity": self._identity(),
                   "parameters": {n: p.detach().cpu().clone() for n, p in self._parameters.items()},
                   "optimizer": self.optimizer.state_dict(), "update": self.update_count,
                   "prompt_position": self.prompt_position, "schedule": self._schedule,
                   "trainer_seed": self.seed, "torch_rng": torch.get_rng_state(),
                   "scheduler": self._backend.lr_scheduler.state_dict() if self._backend else getattr(self, "_pending_scheduler", None),
                   "python_rng": random.getstate(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if self.actor.device.type == "cuda" else [],
                   "forked_from": getattr(self, "forked_from", None)}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=destination.parent, prefix=".ppo-", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if replace_existing:
                os.replace(name, destination)
            else:
                os.link(name, destination)
        finally:
            Path(name).unlink(missing_ok=True)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume into an independently loaded, matching base model and reward setup."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self._restore_checkpoint(payload)

    def fork_checkpoint(self, path: str | Path, *, expected_reward_id: str) -> None:
        """Explicitly fork a known source reward into this trainer's target reward.

        All other identities, trainable weights, optimizer/scheduler and progress
        are restored exactly as for ordinary resume. This does not train or save.
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self._restore_checkpoint(payload, expected_reward_id=expected_reward_id)
        self.forked_from = {"checkpoint": str(Path(path).resolve()), "reward_id": expected_reward_id}

    def _restore_checkpoint(self, payload: dict, *, expected_reward_id: str | None = None) -> None:
        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") != 2):
            raise PPOError("Unsupported checkpoint schema")
        expected = self._identity()
        if expected_reward_id is not None:
            if not isinstance(expected_reward_id, str) or not expected_reward_id.strip():
                raise PPOError("Provide the expected source reward identity for a fork")
            expected["reward_id"] = expected_reward_id
        if payload.get("identity") != expected:
            raise PPOError("Checkpoint configuration, model or experiment/reward identity differs")
        update, position = payload.get("update"), payload.get("prompt_position")
        if (type(update) is not int or not 0 <= update <= self.config.total_updates
                or type(position) is not int or position != update * self.config.rollout_batch_size):
            raise PPOError("Invalid checkpoint progress")
        saved = payload.get("parameters")
        if not isinstance(saved, dict) or set(saved) != set(self._parameters):
            raise PPOError("Checkpoint trainable parameter names differ")
        for name, parameter in self._parameters.items():
            value = saved[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                raise PPOError(f"Invalid checkpoint parameter: {name}")
        optimizer_state = payload.get("optimizer")
        if (not isinstance(optimizer_state, dict)
                or optimizer_state.get("param_groups") != self.optimizer.state_dict()["param_groups"]):
            raise PPOError("Checkpoint optimizer settings differ")
        restored_optimizer = torch.optim.AdamW(list(self._parameters.values()), lr=self.config.learning_rate,
                                               betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        restored_optimizer.load_state_dict(optimizer_state)
        for parameter, state in restored_optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq", "step"):
                value = state.get(name)
                if (not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                        or (name != "step" and value.shape != parameter.shape)):
                    raise PPOError("Invalid checkpoint optimizer state")
        try:
            self._failed = True
            with torch.no_grad():
                for name, parameter in self._parameters.items():
                    parameter.copy_(saved[name])
            self.optimizer = restored_optimizer
            self.seed = payload["trainer_seed"]
            self._backend = None
            self._pending_scheduler = payload["scheduler"]
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
            if self.actor.device.type == "cuda":
                if len(payload["cuda_rng"]) != torch.cuda.device_count():
                    raise PPOError("Checkpoint CUDA device count differs")
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
            self._schedule = payload["schedule"]
            self.forked_from = payload.get("forked_from")
            self.update_count, self.prompt_position = update, position
            self.optimizer.zero_grad(set_to_none=True)
            self._failed = False
        except BaseException:
            self._failed = True
            raise

    def release(self) -> None:
        """Release trainer/optimizer resources after a coordinator stage is saved.

        The caller may retain the actor. This trainer must not be reused without
        restoring a checkpoint; optimizer moments have been released.
        """
        if self._backend is not None:
            self._backend.accelerator.free_memory()
            self._backend = None
        self.optimizer.state.clear()
        directory = getattr(self, "_runtime_directory", None)
        if directory is not None:
            directory.cleanup()
        self._failed = True
