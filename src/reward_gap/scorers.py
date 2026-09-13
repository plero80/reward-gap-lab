"""Frozen Qwen3 reward scoring for the selected Skywork proxy and judge."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from reward_gap.data import PromptRecord
from reward_gap.formatting import format_reward_batch
from reward_gap.models import LoadedModel, load_experiment_model

if TYPE_CHECKING:
    import torch
    from reward_gap.config import ExperimentConfig


class ScoringError(ValueError):
    """Invalid scorer settings or unusable model outputs."""


@dataclass(frozen=True)
class ScoreBatch:
    prompt_ids: tuple[str, ...]
    scores: tuple[float, ...]
    token_counts: tuple[int, ...]
    role: Literal["proxy", "judge"]
    source: str
    revision: str | None
    embeddings: "torch.Tensor | None" = None
    embedding_pooling: str | None = None


class RewardScorer:
    """Wrap one loaded scalar Qwen3 reward model, with no optimizer or updates.

    Both roles return raw scalar logits. Only the proxy supplies embeddings,
    so memory keys cannot accidentally mix proxy and judge representations.
    """

    def __init__(self, loaded: LoadedModel, *, role: Literal["proxy", "judge"],
                 max_tokens: int = 4096, batch_size: int = 4):
        from transformers import Qwen3ForSequenceClassification

        if role not in ("proxy", "judge"):
            raise ScoringError("Scorer role must be proxy or judge")
        for name, value in (("max_tokens", max_tokens), ("batch_size", batch_size)):
            if type(value) is not int or value <= 0:
                raise ScoringError(f"{name}: expected a positive integer")
        if max_tokens > 16384:
            raise ScoringError("Skywork scoring limit is 16384 tokens")
        if loaded.role != "reward" or not isinstance(loaded.model, Qwen3ForSequenceClassification):
            raise ScoringError("RewardScorer supports loaded Qwen3 sequence-classification reward models")
        if loaded.model.config.num_labels != 1:
            raise ScoringError("RewardScorer requires one scalar output per answer")
        if loaded.tokenizer.pad_token_id is None or loaded.model.config.pad_token_id != loaded.tokenizer.pad_token_id:
            raise ScoringError("Model and tokenizer must share a valid padding token")
        context_window = loaded.model.config.max_position_embeddings
        if type(context_window) is not int or context_window <= 0:
            raise ScoringError("Model must declare a positive context window")
        self.loaded = loaded
        self.role: Literal["proxy", "judge"] = role
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.context_window = context_window
        self.loaded.model.requires_grad_(False)
        self.loaded.model.eval()

    @classmethod
    def load(cls, config: "ExperimentConfig", role: Literal["proxy", "judge"]) -> "RewardScorer":
        """Load just one configured role using models.py's cache/device rules."""
        if role not in ("proxy", "judge"):
            raise ScoringError("Scorer role must be proxy or judge")
        loaded = load_experiment_model(config, role)
        return cls(loaded, role=role, max_tokens=config.scoring.max_tokens,
                   batch_size=config.scoring.batch_size)

    def score(self, prompts: Sequence[PromptRecord], answers: Sequence[str], *,
              return_embeddings: bool = False) -> ScoreBatch:
        """Score complete answers in input order, in bounded microbatches.

        No truncation, sigmoid, normalization of scores, or file writes.
        Empty answer strings remain scoreable. Formatting errors propagate
        with their prompt context; invalid scores/embeddings fail explicitly.
        """
        import torch

        if type(return_embeddings) is not bool:
            raise ScoringError("return_embeddings must be a boolean")
        if return_embeddings and self.role != "proxy":
            raise ScoringError("Memory embeddings must come from the proxy, not the judge")
        if not prompts or isinstance(answers, str) or len(prompts) != len(answers):
            raise ScoringError("Provide a nonempty prompt batch and exactly one answer per prompt")
        model = self.loaded.model
        # Reassert inference behavior if an external caller changed the mode.
        model.requires_grad_(False)
        model.eval()
        scores: list[float] = []
        counts: list[int] = []
        vectors: list[torch.Tensor] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            batch = format_reward_batch(
                self.loaded.tokenizer, chunk, answers[start:start + self.batch_size],
                max_tokens=self.max_tokens, context_window=self.context_window,
            ).to(model.device)
            with torch.inference_mode():
                outputs = model(**batch.model_inputs(), use_cache=False,
                                output_hidden_states=return_embeddings, return_dict=True)
                logits = outputs.logits
                if not isinstance(logits, torch.Tensor) or logits.shape != (len(chunk), 1):
                    raise ScoringError("Reward model must return logits with shape [batch, 1]")
                raw = logits[:, 0].float()
                if not torch.isfinite(raw).all().item():
                    raise ScoringError(f"Nonfinite reward score in batch starting at prompt {start}")
                scores.extend(raw.cpu().tolist())
                counts.extend(batch.token_counts)
                if return_embeddings:
                    if not outputs.hidden_states:
                        raise ScoringError("Proxy did not return hidden states for memory embeddings")
                    hidden = outputs.hidden_states[-1]
                    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[:2] != batch.input_ids.shape:
                        raise ScoringError("Proxy hidden states do not match the formatted batch")
                    # Match Qwen3's scalar-head pooling: rightmost token != PAD.
                    # The mask also ensures padding positions can never be used.
                    valid = batch.attention_mask.bool() & batch.input_ids.ne(model.config.pad_token_id)
                    if not valid.any(dim=1).all().item():
                        raise ScoringError("Cannot pool a conversation with no non-padding tokens")
                    positions = torch.arange(valid.shape[1], device=hidden.device).expand_as(valid)
                    last = positions.masked_fill(~valid, -1).max(dim=1).values
                    pooled = hidden[torch.arange(len(chunk), device=hidden.device), last].float()
                    norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
                    if not torch.isfinite(pooled).all().item() or not torch.isfinite(norm).all().item() or (norm <= 0).any().item():
                        raise ScoringError("Proxy embedding has a nonfinite or zero norm")
                    vectors.append((pooled / norm).cpu())
                # Do not keep all-layer hidden states alive between microbatches.
                del outputs
        embeddings = torch.cat(vectors, dim=0) if return_embeddings else None
        return ScoreBatch(tuple(r.prompt_id for r in prompts), tuple(scores), tuple(counts),
                          self.role, self.loaded.source, self.loaded.revision, embeddings,
                          "last_non_pad_final_hidden_l2_v1" if return_embeddings else None)
