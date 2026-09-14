"""Model-shaped adapters for TRL PPO; no PPO math or optimizer updates."""

from collections import defaultdict, deque
from types import SimpleNamespace

import torch
from reward_gap.failures import SampleError


class RewardBridge(torch.nn.Module):
    """Translate TRL's policy-token batches back to original prompt/answer pairs."""

    def __init__(self, tokenizer, strategy, *, eos_ids=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.eos_ids = tuple(eos_ids) if eos_ids is not None else (tokenizer.eos_token_id,)
        self.width = 0
        self.records = defaultdict(deque)
        self.batches = []
        self.sample_failure = None

    def bind(self, prompts, batch):
        self.width = batch.input_ids.shape[1]
        self.records.clear()
        self.batches.clear()
        self.sample_failure = None
        # TRL replaces left-padding IDs with zero before calling the backbone.
        ids = batch.input_ids.masked_fill(~batch.attention_mask.bool(), 0).cpu().tolist()
        for record, row in zip(prompts, ids, strict=True):
            self.records[tuple(row)].append(record)

    def forward(self, input_ids, attention_mask, **kwargs):
        records, answers, lengths, reasons = [], [], [], []
        score_rollouts = getattr(self.strategy, "score_rollouts", None)
        for row, mask in zip(input_ids, attention_mask, strict=True):
            key = tuple(row[:self.width].cpu().tolist())
            if not self.records[key]:
                raise ValueError("TRL reward input does not match the bound prompt batch")
            records.append(self.records[key].popleft())
            suffix = row[self.width:][mask[self.width:].bool()]
            lengths.append(len(suffix))
            reasons.append("eos" if any(suffix.eq(eos).any().item() for eos in self.eos_ids) else "length")
            if score_rollouts is not None:
                # Native TRL treats the first PAD as the end of the response.
                # A sampled PAD inside a completion cannot silently become a
                # different candidate/length-penalty decision in our adapter.
                response_mask = mask[self.width:].bool()
                expected_mask = torch.arange(len(response_mask), device=mask.device) < len(suffix)
                if (not torch.equal(response_mask, expected_mask) or not len(suffix)
                        or (reasons[-1] == "length" and len(suffix) != len(response_mask))):
                    error = "Unexpected PAD inside a completion; cannot establish its EOS/length status"
                    if getattr(self.strategy, "recover_sample_failures", False):
                        self.sample_failure = SampleError(error)
                        raise self.sample_failure
                    raise ValueError(error)
            answers.append(self.tokenizer.decode(suffix.tolist(), skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False))
        try:
            result = (score_rollouts(records, answers, response_lengths=lengths, finish_reasons=reasons)
                      if score_rollouts is not None else self.strategy.score(records, answers))
        except SampleError as exc:
            self.sample_failure = exc
            raise
        if result.prompt_ids != tuple(p.prompt_id for p in records) or len(result.rewards) != len(records):
            raise ValueError("Reward results do not match TRL's prompt batch")
        scores = torch.tensor(result.rewards, device=input_ids.device, dtype=torch.float32)
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite answer rewards")
        self.batches.append(result)
        # TRL pools the score at the end of the response. A scalar repeated
        # along positions adapts an answer-level strategy without retokenizing.
        return SimpleNamespace(hidden_states=(scores[:, None, None].expand(-1, input_ids.shape[1], 1),))


class RewardModel(torch.nn.Module):
    base_model_prefix = "backbone"

    def __init__(self, bridge):
        super().__init__()
        self.backbone = bridge
        self.score = torch.nn.Identity()


class ValueScore(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden):
        return self.head(hidden.to(self.head.weight.dtype))


class ValueModel(torch.nn.Module):
    base_model_prefix = "backbone"

    def __init__(self, actor):
        super().__init__()
        # LoRA has already been attached in-place to these Qwen layers. Reuse
        # them without computing the vocabulary-sized LM logits for the critic.
        self.backbone = actor.model.get_base_model().model
        self.score = ValueScore(actor.value_head)
