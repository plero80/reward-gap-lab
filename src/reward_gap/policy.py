"""Qwen2 LoRA actor and token statistics; PPO optimization is separate."""

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
import math
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import GenerationConfig as HFGenerationConfig, Qwen2ForCausalLM

from reward_gap.config import ExperimentConfig, GenerationConfig, PolicyConfig
from reward_gap.data import PromptRecord
from reward_gap.formatting import format_policy_batch
from reward_gap.models import LoadedModel, load_experiment_model


class PolicyError(ValueError):
    """Unsupported policy configuration or malformed rollout."""


@dataclass(frozen=True)
class RolloutBatch:
    prompt_ids: tuple[str, ...]
    answers: tuple[str, ...]
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    prompt_width: int
    prompt_token_counts: tuple[int, ...]
    response_lengths: tuple[int, ...]
    finish_reasons: tuple[Literal["eos", "length"], ...]
    sampled: bool
    seed: int

    @property
    def response_ids(self) -> torch.Tensor:
        """Generated suffix including EOS and masked trailing padding."""
        return self.sequences[:, self.prompt_width:]


@dataclass(frozen=True)
class PolicyStatistics:
    # All token fields align with rollout.response_ids, shape [batch, response width].
    log_probs: torch.Tensor
    values: torch.Tensor
    response_mask: torch.Tensor
    # Value of the state after the last action; zero when EOS terminated it.
    bootstrap_values: torch.Tensor


@contextmanager
def _seeded(seed: int, device: torch.device):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise PolicyError("seed must be an integer in [0, 2**63)")
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        for index in devices:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
        yield


class PPOActor(torch.nn.Module):
    """Own a frozen Qwen2 base plus trainable LoRA and a float32 value head.

    Attaching PEFT modifies the supplied base model in place: do not share it
    with another actor. Dropout is disabled even during gradient computation
    so PPO compares the same policy distribution used during generation.
    """

    def __init__(self, loaded: LoadedModel, *, policy: PolicyConfig = PolicyConfig(),
                 generation: GenerationConfig = GenerationConfig(), seed: int = 42):
        super().__init__()
        if loaded.role != "policy" or not isinstance(loaded.model, Qwen2ForCausalLM):
            raise PolicyError("PPOActor supports the loaded Qwen2 causal-LM policy")
        if isinstance(loaded.model, PeftModel) or hasattr(loaded.model, "peft_config"):
            raise PolicyError("Policy already has an adapter; load a fresh base for a new actor")
        for name, value in (("lora_rank", policy.lora_rank), ("lora_alpha", policy.lora_alpha),
                            ("max_prompt_tokens", generation.max_prompt_tokens), ("max_new_tokens", generation.max_new_tokens)):
            if type(value) is not int or value <= 0:
                raise PolicyError(f"{name} must be a positive integer")
        if type(generation.do_sample) is not bool:
            raise PolicyError("do_sample must be a boolean")
        if (not policy.target_modules or any(t not in PolicyConfig().target_modules for t in policy.target_modules)
                or len(set(policy.target_modules)) != len(policy.target_modules)):
            raise PolicyError("Choose unique supported Qwen LoRA target modules")
        self.tokenizer = loaded.tokenizer
        self.source = loaded.source
        self.revision = loaded.revision
        self.generation = generation
        self.context_window = loaded.model.config.max_position_embeddings
        if type(self.context_window) is not int or self.context_window <= 0:
            raise PolicyError("Policy must declare a positive context window")
        if loaded.tokenizer.pad_token_id is None or loaded.model.config.pad_token_id != loaded.tokenizer.pad_token_id:
            raise PolicyError("Policy and tokenizer must share a padding token")
        eos = loaded.model.generation_config.eos_token_id
        if eos is None:
            eos = loaded.tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else eos
        if not isinstance(eos_ids, (list, tuple)) or not eos_ids or any(type(token) is not int or token < 0 for token in eos_ids):
            raise PolicyError("Policy needs valid EOS token IDs")
        self.eos_ids: tuple[int, ...] = tuple(int(token) for token in eos_ids)
        loaded.model.requires_grad_(False)
        with _seeded(seed, loaded.model.device):
            self.model = get_peft_model(loaded.model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=policy.lora_rank, lora_alpha=policy.lora_alpha,
                target_modules=list(policy.target_modules), lora_dropout=0.0, bias="none",
                init_lora_weights=True,
            ))
            self.value_head = torch.nn.Linear(loaded.model.config.hidden_size, 1,
                                             device=loaded.model.device, dtype=torch.float32)
            torch.nn.init.zeros_(self.value_head.weight)
            torch.nn.init.zeros_(self.value_head.bias)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.value_head.weight.device

    @classmethod
    def load(cls, config: ExperimentConfig, *, seed: int) -> "PPOActor":
        return cls(load_experiment_model(config, "policy"), policy=config.policy,
                   generation=config.generation, seed=seed)

    @torch.no_grad()
    def generate(self, prompts: Sequence[PromptRecord], *, seed: int, temperature: float = 1.0) -> RolloutBatch:
        """Generate once; retain original sampled IDs including the first EOS.

        Sampling uses the full softmax at the requested temperature (default 1).
        No inherited Qwen top-k/top-p/repetition penalties alter it. The
        statistics helpers describe temperature-1 logits; TRL computes its
        own temperature-adjusted training probabilities.
        Greedy mode is available for evaluation and labeled sampled=False.
        """
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise PolicyError("Generation temperature must be finite and positive")
        self.model.eval()
        batch = format_policy_batch(
            self.tokenizer, prompts, max_prompt_tokens=self.generation.max_prompt_tokens,
            max_new_tokens=self.generation.max_new_tokens, context_window=self.context_window,
        ).to(self.device)
        settings = HFGenerationConfig(
            max_new_tokens=self.generation.max_new_tokens, do_sample=self.generation.do_sample,
            top_k=0, top_p=1.0, temperature=temperature, num_beams=1,
            eos_token_id=list(self.eos_ids), pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True,
        )
        with _seeded(seed, self.device):
            sequences = self.model.generate(**batch.model_inputs(), generation_config=settings)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise PolicyError("Generation must return a tensor of token sequences")
        width = batch.input_ids.shape[1]
        if sequences.shape[0] != len(prompts) or not torch.equal(sequences[:, :width], batch.input_ids):
            raise PolicyError("Generation did not preserve the input prompt tokens")
        suffix = sequences[:, width:]
        if suffix.shape[1] == 0:
            from reward_gap.failures import SampleError
            raise SampleError("Generation returned no answer tokens")
        if suffix.shape[1] > self.generation.max_new_tokens:
            raise PolicyError("Generation returned an invalid response length")
        eos = torch.zeros_like(suffix, dtype=torch.bool)
        for token in self.eos_ids:
            eos |= suffix.eq(token)
        # Include the first EOS action; exclude everything generated after it.
        valid = (eos.long().cumsum(dim=1) - eos.long()).eq(0)
        lengths = tuple(int(n) for n in valid.sum(dim=1).tolist())
        reasons: tuple[Literal["eos", "length"], ...] = tuple("eos" if ended else "length" for ended in eos.any(dim=1).tolist())
        attention = torch.cat((batch.attention_mask, valid.long()), dim=1)
        response_mask = torch.cat((torch.zeros_like(batch.attention_mask, dtype=torch.bool), valid), dim=1)
        answers: list[str] = []
        for i, length in enumerate(lengths):
            text = self.tokenizer.decode(suffix[i, :length].tolist(), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            if not isinstance(text, str):
                raise PolicyError("Tokenizer must decode one answer string per sequence")
            answers.append(text)
        return RolloutBatch(batch.prompt_ids, tuple(answers), sequences, attention, response_mask, width,
                            batch.token_counts, lengths, reasons, self.generation.do_sample, seed)

    def _inputs(self, rollout: RolloutBatch) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if (rollout.sequences.ndim != 2 or rollout.attention_mask.shape != rollout.sequences.shape
                or rollout.response_mask.shape != rollout.sequences.shape
                or not 1 <= rollout.prompt_width < rollout.sequences.shape[1]):
            raise PolicyError("Invalid rollout token shapes or prompt width")
        if rollout.sequences.shape[1] > self.context_window:
            raise PolicyError("Rollout exceeds policy context window")
        size = rollout.sequences.shape[0]
        if size == 0 or any(len(items) != size for items in (
            rollout.prompt_ids, rollout.answers, rollout.prompt_token_counts,
            rollout.response_lengths, rollout.finish_reasons,
        )):
            raise PolicyError("Rollout metadata must have one entry per sequence")
        if rollout.sequences.dtype != torch.long or rollout.response_mask.dtype != torch.bool:
            raise PolicyError("Rollout needs int64 token IDs and a boolean response mask")
        device = self.device
        ids = rollout.sequences.to(device)
        mask = rollout.attention_mask.to(device)
        response = rollout.response_mask.to(device).bool()
        if not ((mask == 0) | (mask == 1)).all().item():
            raise PolicyError("Attention mask must contain only zeros and ones")
        if (response[:, :rollout.prompt_width].any().item()
                or (response & ~mask.bool()).any().item()
                or not response.any(dim=1).all().item()):
            raise PolicyError("Response masks must select attended generated tokens only")
        suffix_mask = response[:, rollout.prompt_width:]
        lengths = suffix_mask.sum(dim=1)
        expected = torch.arange(suffix_mask.shape[1], device=device)[None, :] < lengths[:, None]
        if (not torch.equal(suffix_mask, expected)
                or not torch.equal(mask[:, rollout.prompt_width:].bool(), expected)
                or tuple(lengths.tolist()) != rollout.response_lengths):
            raise PolicyError("Generated-token masks and response lengths disagree")
        prefix = mask[:, :rollout.prompt_width]
        if (not prefix[:, -1].eq(1).all().item()
                or (prefix[:, 1:] < prefix[:, :-1]).any().item()
                or tuple(prefix.sum(dim=1).tolist()) != rollout.prompt_token_counts):
            raise PolicyError("Prompt masks must describe left-padded prompts")
        suffix = ids[:, rollout.prompt_width:]
        eos = torch.zeros_like(suffix, dtype=torch.bool)
        for token in self.eos_ids:
            eos |= suffix.eq(token)
        for i, reason in enumerate(rollout.finish_reasons):
            length = int(lengths[i].item())
            ended = eos[i, length - 1].item()
            if reason not in ("eos", "length") or (reason == "eos") != ended or eos[i, :length - 1].any().item():
                raise PolicyError("Finish reason disagrees with generated EOS tokens")
        # Ignore left padding in position numbering, matching generate().
        positions = mask.long().cumsum(dim=1) - 1
        positions.masked_fill_(~mask.bool(), 0)
        return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}, suffix_mask

    @staticmethod
    def _log_probs(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        selected = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        result = selected - torch.logsumexp(logits, dim=-1)
        if not torch.isfinite(result[mask]).all().item():
            raise PolicyError("Nonfinite response log probabilities")
        return result.masked_fill(~mask, 0.0)

    def statistics(self, rollout: RolloutBatch) -> PolicyStatistics:
        """Recompute differentiable response log probabilities and pre-action values.

        Call under torch.no_grad() when recording old-policy rollout statistics.
        The first response token is predicted by the final prompt position.
        Never reconstruct PPO token inputs by decoding and retokenizing answers.
        """
        self.model.eval()
        inputs, mask = self._inputs(rollout)
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        start = rollout.prompt_width - 1
        log_probs = self._log_probs(outputs.logits[:, start:-1], inputs["input_ids"][:, rollout.prompt_width:], mask)
        hidden = outputs.hidden_states[-1]
        all_values = self.value_head(hidden.to(self.value_head.weight.dtype)).squeeze(-1)
        values = all_values[:, start:-1].masked_fill(~mask, 0.0)
        ends = rollout.prompt_width + mask.sum(dim=1) - 1
        bootstrap = all_values[torch.arange(len(rollout.prompt_ids), device=all_values.device), ends]
        terminated = torch.tensor([r == "eos" for r in rollout.finish_reasons], device=all_values.device)
        bootstrap = bootstrap.masked_fill(terminated, 0.0)
        if not torch.isfinite(values[mask]).all().item() or not torch.isfinite(bootstrap).all().item():
            raise PolicyError("Nonfinite value-head predictions")
        return PolicyStatistics(log_probs, values, mask, bootstrap)

    @torch.no_grad()
    def reference_log_probs(self, rollout: RolloutBatch) -> torch.Tensor:
        """Score original base-policy probabilities with LoRA temporarily disabled."""
        self.model.eval()
        inputs, mask = self._inputs(rollout)
        with self.model.disable_adapter():
            outputs = self.model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
        return self._log_probs(outputs.logits[:, rollout.prompt_width - 1:-1],
                               inputs["input_ids"][:, rollout.prompt_width:], mask)
