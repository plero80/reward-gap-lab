"""Single-device PPO over sampled answers, with update-boundary checkpoints."""

import math
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import torch
from peft import LoraConfig

from reward_gap.config import TrainingConfig
from reward_gap.data import PromptRecord
from reward_gap.policy import PPOActor, RolloutBatch
from reward_gap.rewards import RewardBatch


class PPOError(ValueError):
    """Invalid PPO inputs, incompatible checkpoints or failed updates."""


class RewardStrategy(Protocol):
    def score(self, prompts: Sequence[PromptRecord], answers: Sequence[str]) -> RewardBatch: ...


@dataclass(frozen=True)
class UpdateMetrics:
    update: int
    prompt_position: int
    rollout_seed: int
    mean_reward: float
    mean_sampled_kl: float
    mean_response_length: float
    eos_fraction: float
    policy_loss: float
    value_loss: float
    clip_fraction: float
    grad_norm: float
    optimizer_steps: int


def _mask(mask: torch.Tensor) -> None:
    if mask.ndim != 2 or mask.dtype != torch.bool or min(mask.shape) == 0:
        raise PPOError("Expected a nonempty boolean response mask")
    if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise PPOError("Response masks must be nonempty contiguous prefixes")


@torch.no_grad()
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, *,
                gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for finite answers. EOS AND length-limit endings have zero bootstrap."""
    _mask(mask)
    if rewards.shape != mask.shape or values.shape != mask.shape:
        raise PPOError("Rewards, values and masks must have matching shapes")
    for value in (gamma, gae_lambda):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise PPOError("GAE settings must be in [0, 1]")
    if not torch.isfinite(rewards[mask]).all() or not torch.isfinite(values[mask]).all():
        raise PPOError("Nonfinite GAE inputs")
    rewards = rewards.float().masked_fill(~mask, 0)
    values = values.float().masked_fill(~mask, 0)
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[:, 0])
    next_value = torch.zeros_like(carry)
    for t in reversed(range(mask.shape[1])):
        carry = (rewards[:, t] + gamma * next_value - values[:, t]
                 + gamma * gae_lambda * carry).masked_fill(~mask[:, t], 0)
        advantages[:, t] = carry
        next_value = values[:, t]
    returns = (advantages + values).masked_fill(~mask, 0)
    return advantages, returns


def ppo_loss(log_probs: torch.Tensor, values: torch.Tensor, old_log_probs: torch.Tensor,
             old_values: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor,
             mask: torch.Tensor, config: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token-mean clipped policy and value losses, excluding all padding."""
    _mask(mask)
    tensors = (log_probs, values, old_log_probs, old_values, advantages, returns)
    if any(t.shape != mask.shape for t in tensors):
        raise PPOError("PPO tensors must match the response mask")
    if any(not torch.isfinite(t[mask]).all() for t in tensors):
        raise PPOError("Nonfinite PPO inputs")
    logp, value = log_probs[mask], values[mask]
    old_logp, old_value, advantage, target = (t[mask].detach() for t in tensors[2:])
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(ratio * advantage,
                            ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * advantage).mean()
    clipped_value = old_value + (value - old_value).clamp(-config.value_clip_range, config.value_clip_range)
    critic = 0.5 * torch.maximum((value - target).square(), (clipped_value - target).square()).mean()
    clipped = ((ratio - 1).abs() > config.clip_range).float().mean()
    return policy, critic, clipped


def _select(rollout: RolloutBatch, indices: list[int]) -> RolloutBatch:
    return replace(rollout,
                   prompt_ids=tuple(rollout.prompt_ids[i] for i in indices),
                   answers=tuple(rollout.answers[i] for i in indices),
                   sequences=rollout.sequences[indices], attention_mask=rollout.attention_mask[indices],
                   response_mask=rollout.response_mask[indices],
                   prompt_token_counts=tuple(rollout.prompt_token_counts[i] for i in indices),
                   response_lengths=tuple(rollout.response_lengths[i] for i in indices),
                   finish_reasons=tuple(rollout.finish_reasons[i] for i in indices))


class PPOTrainer:
    """Updates LoRA and critic only. One optimizer step per sequence minibatch.

    experiment_id labels dataset/schedule/base artifacts; reward_id labels the
    strategy, calibration and memory snapshot. Callers must use distinct labels
    for changed artifacts. Checkpoints do not contain frozen model weights.
    """

    def __init__(self, actor: PPOActor, reward: RewardStrategy, config: TrainingConfig, *,
                 experiment_id: str, reward_id: str, seed: int = 42):
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
        self._shuffle = torch.Generator(device="cpu").manual_seed(seed)
        self.update_count = 0
        self.prompt_position = 0
        self._failed = False
        self._schedule = None

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
        rollout = self.actor.generate(prompts, seed=rollout_seed)
        if not rollout.sampled:
            raise PPOError("PPO requires sampled rollouts")
        scored = self.reward.score(prompts, rollout.answers)
        if scored.prompt_ids != rollout.prompt_ids or len(scored.rewards) != len(prompts):
            raise PPOError("Reward batch does not match rollout")
        with torch.no_grad():
            old = self.actor.statistics(rollout)
            reference = self.actor.reference_log_probs(rollout)
            mask = old.response_mask
            scalar = torch.tensor(scored.rewards, device=self.actor.device, dtype=torch.float32)
            if not torch.isfinite(scalar).all():
                raise PPOError("Nonfinite answer rewards")
            sampled_kl = (old.log_probs - reference).masked_fill(~mask, 0)
            rewards = -self.config.kl_coefficient * sampled_kl
            ends = mask.sum(1) - 1
            rewards[torch.arange(len(prompts), device=self.actor.device), ends] += scalar
            advantages, returns = compute_gae(rewards, old.values, mask,
                                              gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
            if self.config.normalize_advantages:
                active = advantages[mask]
                advantages = ((advantages - active.mean()) / active.std(unbiased=False).clamp_min(1e-8)).masked_fill(~mask, 0)
        sums = [0., 0., 0., 0.]
        steps = 0
        try:
            for _ in range(self.config.ppo_epochs):
                order = torch.randperm(len(prompts), generator=self._shuffle).tolist()
                for start in range(0, len(prompts), self.config.minibatch_size):
                    ids = order[start:start + self.config.minibatch_size]
                    self.optimizer.zero_grad(set_to_none=True)
                    current = self.actor.statistics(_select(rollout, ids))
                    policy, critic, clipped = ppo_loss(
                        current.log_probs, current.values, old.log_probs[ids], old.values[ids],
                        advantages[ids], returns[ids], mask[ids], self.config)
                    loss = policy + self.config.value_coefficient * critic
                    if not torch.isfinite(loss):
                        raise PPOError("Nonfinite PPO loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(list(self._parameters.values()),
                                                          self.config.max_grad_norm, error_if_nonfinite=True)
                    self.optimizer.step()
                    if any(not torch.isfinite(p).all() for p in self._parameters.values()):
                        raise PPOError("Optimizer produced nonfinite parameters")
                    for i, value in enumerate((policy, critic, clipped, norm)):
                        sums[i] += float(value.detach())
                    steps += 1
        except BaseException:
            # Some optimizer steps may have succeeded. Never label that a completed update.
            self._failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
        self.update_count += 1
        self.prompt_position += len(prompts)
        return UpdateMetrics(self.update_count, self.prompt_position, rollout_seed, float(scalar.mean()),
                             float(sampled_kl[mask].mean()), sum(rollout.response_lengths) / len(prompts),
                             rollout.finish_reasons.count("eos") / len(prompts),
                             sums[0] / steps, sums[1] / steps, sums[2] / steps, sums[3] / steps, steps)

    def train(self, prompt_batches: Sequence[Sequence[PromptRecord]], rollout_seeds: Sequence[int], *,
              checkpoint_dir: str | Path | None = None, until_update: int | None = None) -> list[UpdateMetrics]:
        """Run a full fixed schedule, resuming at update_count; optionally pause early."""
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
        adapter = self.actor.model.peft_config["default"]
        if not isinstance(adapter, LoraConfig) or not isinstance(adapter.target_modules, set):
            raise PPOError("Checkpoint requires the actor's named LoRA target modules")
        return {"experiment_id": self.experiment_id, "reward_id": self.reward_id,
                "training": asdict(self.config), "generation": asdict(self.actor.generation),
                "source": self.actor.source, "revision": self.actor.revision,
                "device": str(self.actor.device), "context_window": self.actor.context_window,
                "base_dtype": str(next(self.actor.model.parameters()).dtype),
                "adapter": {"rank": adapter.r, "alpha": adapter.lora_alpha,
                            "targets": sorted(adapter.target_modules)},
                "tokenizer": {"template": self.actor.tokenizer.chat_template,
                              "pad": self.actor.tokenizer.pad_token_id, "eos": self.actor.eos_ids,
                              "size": len(self.actor.tokenizer)},
                "packages": {name: version(name) for name in ("torch", "transformers", "peft")}}

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically publish complete training state; refuse to replace a checkpoint."""
        if self._failed:
            raise PPOError("Cannot checkpoint a partially failed update")
        payload = {"schema_version": 1, "identity": self._identity(),
                   "parameters": {n: p.detach().cpu().clone() for n, p in self._parameters.items()},
                   "optimizer": self.optimizer.state_dict(), "update": self.update_count,
                   "prompt_position": self.prompt_position, "schedule": self._schedule,
                   "shuffle_rng": self._shuffle.get_state(), "torch_rng": torch.get_rng_state(),
                   "python_rng": random.getstate(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if self.actor.device.type == "cuda" else []}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=destination.parent, prefix=".ppo-", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(name, destination)
        finally:
            Path(name).unlink(missing_ok=True)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume into an independently loaded, matching base model and reward setup."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") != 1):
            raise PPOError("Unsupported checkpoint schema")
        if payload.get("identity") != self._identity():
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
            self._shuffle.set_state(payload["shuffle_rng"])
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
            if self.actor.device.type == "cuda":
                if len(payload["cuda_rng"]) != torch.cuda.device_count():
                    raise PPOError("Checkpoint CUDA device count differs")
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
            self._schedule = payload["schedule"]
            self.update_count, self.prompt_position = update, position
            self.optimizer.zero_grad(set_to_none=True)
            self._failed = False
        except BaseException:
            self._failed = True
            raise
