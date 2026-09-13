"""Format HH-RLHF prompts for Qwen policy generation and Skywork rewards.

No model loading, generation, scoring, or file writes occur here. Tokenizers
are supplied by models.py; tensors are created on CPU and can be moved later.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from reward_gap.data import Message, PromptRecord

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


class FormattingError(ValueError):
    """A conversation cannot be formatted within the requested constraints."""


@dataclass(frozen=True)
class FormattedBatch:
    prompt_ids: tuple[str, ...]
    texts: tuple[str, ...]
    token_counts: tuple[int, ...]
    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"

    def model_inputs(self) -> dict[str, "torch.Tensor"]:
        """Only the arguments passed to model.forward() or model.generate()."""
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}

    def to(self, device: "str | torch.device") -> "FormattedBatch":
        """Return a batch with tensors moved to the caller's selected device."""
        return replace(self, input_ids=self.input_ids.to(device),
                       attention_mask=self.attention_mask.to(device))


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise FormattingError(f"{name}: expected a positive integer")


def _conversation(record: PromptRecord, *, reward: bool) -> list[dict[str, str]]:
    if not isinstance(record, PromptRecord):
        raise FormattingError("Expected PromptRecord objects")
    if not isinstance(record.prompt_id, str) or not record.prompt_id.strip():
        raise FormattingError("Prompt record needs a nonempty prompt_id")
    if not record.messages:
        raise FormattingError(f"{record.prompt_id}: conversation is empty")
    messages = []
    for message in record.messages:
        if not isinstance(message, Message) or not isinstance(message.content, str) or not message.content.strip():
            raise FormattingError(f"{record.prompt_id}: expected nonempty Message text")
        messages.append({"role": message.role, "content": message.content})
    if reward and any(m["role"] == "system" for m in messages):
        raise FormattingError(f"{record.prompt_id}: Skywork reward inputs must not include system messages")
    offset = 1 if messages[0]["role"] == "system" else 0
    turns = messages[offset:]
    if not turns or len(turns) % 2 != 1:
        raise FormattingError(f"{record.prompt_id}: prompt must end with a user turn")
    for index, message in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise FormattingError(f"{record.prompt_id}: expected {expected} at conversation turn {index + 1}")
    return messages


def _encode(tokenizer: "PreTrainedTokenizerBase", messages: list[dict[str, str]], *,
            generate: bool, prompt_id: str) -> tuple[str, list[int]]:
    from jinja2 import TemplateError

    if not tokenizer.chat_template:
        raise FormattingError("Tokenizer has no chat template; select a supported chat checkpoint")
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generate,
        )
        if not isinstance(text, str) or not text:
            raise FormattingError(f"{prompt_id}: chat template returned no text")
        # The template already supplied its control tokens. Do not add them twice.
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    except (ValueError, TypeError, TemplateError) as exc:
        raise FormattingError(f"{prompt_id}: cannot apply tokenizer chat template: {exc}") from exc
    if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
        raise FormattingError(f"{prompt_id}: tokenizer returned no valid token IDs")
    return text, ids


def _batch(tokenizer: "PreTrainedTokenizerBase", records: Sequence[PromptRecord],
           texts: list[str], rows: list[list[int]], *, left_padding: bool) -> FormattedBatch:
    import torch

    pad_id = tokenizer.pad_token_id
    if type(pad_id) is not int or pad_id < 0:
        raise FormattingError("Tokenizer needs a valid padding token; configure it in models.py")
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        start = width - len(row) if left_padding else 0
        input_ids[index, start:start + len(row)] = torch.tensor(row, dtype=torch.long)
        # Construct masks from lengths, not token equality: EOS may equal PAD.
        attention_mask[index, start:start + len(row)] = 1
    return FormattedBatch(tuple(r.prompt_id for r in records), tuple(texts),
                          tuple(map(len, rows)), input_ids, attention_mask)


def format_policy_batch(tokenizer: "PreTrainedTokenizerBase", prompts: Sequence[PromptRecord], *,
                        max_prompt_tokens: int, max_new_tokens: int,
                        context_window: int) -> FormattedBatch:
    """Apply the policy chat template and left-pad for batched generation.

    Reserve max_new_tokens within the model's context window. Every length
    includes chat-template control tokens; overlength prompts raise errors.
    Duplicate prompt IDs are allowed because schedules can repeat prompts.
    """
    for name, value in (("max_prompt_tokens", max_prompt_tokens), ("max_new_tokens", max_new_tokens),
                        ("context_window", context_window)):
        _positive(value, name)
    if not prompts:
        raise FormattingError("Policy batch must contain at least one prompt")
    if max_new_tokens >= context_window:
        raise FormattingError("max_new_tokens must leave room for a prompt in context_window")
    limit = min(max_prompt_tokens, context_window - max_new_tokens)
    texts, rows = [], []
    for record in prompts:
        messages = _conversation(record, reward=False)
        text, ids = _encode(tokenizer, messages, generate=True, prompt_id=record.prompt_id)
        if len(ids) > limit:
            raise FormattingError(f"{record.prompt_id}: policy prompt has {len(ids)} tokens; limit is {limit} "
                                  f"with {max_new_tokens} tokens reserved for the answer. No truncation applied.")
        texts.append(text)
        rows.append(ids)
    return _batch(tokenizer, prompts, texts, rows, left_padding=True)


def format_reward_batch(tokenizer: "PreTrainedTokenizerBase", prompts: Sequence[PromptRecord],
                        answers: Sequence[str], *, max_tokens: int,
                        context_window: int) -> FormattedBatch:
    """Format complete conversations for one Skywork reward model, right-padded.

    Supply this scorer's tokenizer, not the policy tokenizer. Append the raw
    generated answer without a new generation prefix. System messages are
    rejected rather than silently deleted. Empty generated answers remain
    scoreable, so failed/empty completions are not dropped from evaluation.
    """
    _positive(max_tokens, "max_tokens")
    _positive(context_window, "context_window")
    if not prompts:
        raise FormattingError("Reward batch must contain at least one prompt")
    if isinstance(answers, str) or len(prompts) != len(answers):
        raise FormattingError("Provide exactly one answer string per prompt")
    # Skywork-Reward-V2 was trained with sequences up to 16,384 tokens.
    limit = min(max_tokens, context_window, 16384)
    texts, rows = [], []
    for record, answer in zip(prompts, answers):
        if not isinstance(answer, str):
            raise FormattingError("Each generated answer must be a string")
        messages = _conversation(record, reward=True)
        messages.append({"role": "assistant", "content": answer})
        text, ids = _encode(tokenizer, messages, generate=False, prompt_id=record.prompt_id)
        if len(ids) > limit:
            raise FormattingError(f"{record.prompt_id}: reward conversation has {len(ids)} tokens; "
                                  f"limit is {limit}. Full answers are required; no truncation applied.")
        texts.append(text)
        rows.append(ids)
    return _batch(tokenizer, prompts, texts, rows, left_padding=False)
