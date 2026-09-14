"""Model-shaped adapters for TRL PPO; no PPO math or optimizer updates."""

from collections import defaultdict, deque
from types import SimpleNamespace

import torch


class RewardBridge(torch.nn.Module):
    """Translate TRL's policy-token batches back to original prompt/answer pairs."""

    def __init__(self, tokenizer, strategy):
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.width = 0
        self.records = defaultdict(deque)
        self.batches = []

    def bind(self, prompts, batch):
        self.width = batch.input_ids.shape[1]
        self.records.clear()
        self.batches.clear()
        # TRL replaces left-padding IDs with zero before calling the backbone.
        ids = batch.input_ids.masked_fill(~batch.attention_mask.bool(), 0).cpu().tolist()
        for record, row in zip(prompts, ids, strict=True):
            self.records[tuple(row)].append(record)

    def forward(self, input_ids, attention_mask, **kwargs):
        records, answers = [], []
        for row, mask in zip(input_ids, attention_mask, strict=True):
            key = tuple(row[:self.width].cpu().tolist())
            if not self.records[key]:
                raise ValueError("TRL reward input does not match the bound prompt batch")
            records.append(self.records[key].popleft())
            suffix = row[self.width:][mask[self.width:].bool()]
            answers.append(self.tokenizer.decode(suffix.tolist(), skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False))
        result = self.strategy.score(records, answers)
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
