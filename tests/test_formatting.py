"""Chat formatting checks; ordinary tests use a tiny offline tokenizer."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from reward_gap.data import Message, PromptRecord
from reward_gap.formatting import FormattingError, format_policy_batch, format_reward_batch


@pytest.fixture
def tokenizer():
    vocab = {token: i for i, token in enumerate(
        ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[USER]", "[ASSISTANT]", "[SYSTEM]",
         "hello", "world", "answer", "earlier", "followup"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    result = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                                    eos_token="[EOS]", unk_token="[UNK]",
                                    additional_special_tokens=["[USER]", "[ASSISTANT]", "[SYSTEM]"])
    result.chat_template = (
        "{{ bos_token }}{% for m in messages %}"
        "{{ '[' + m['role'].upper() + ']' }}{{ m['content'] }}{{ eos_token }}"
        "{% endfor %}{% if add_generation_prompt %}[ASSISTANT]{% endif %}"
    )
    return result


def prompt(text="hello", name="p1"):
    return PromptRecord(name, text, (Message("user", text),))


def policy(tokenizer, prompts, **kwargs):
    return format_policy_batch(tokenizer, prompts, **{
        "max_prompt_tokens": 64, "max_new_tokens": 8, "context_window": 128, **kwargs})


def reward(tokenizer, prompts, answers, **kwargs):
    return format_reward_batch(tokenizer, prompts, answers, **{
        "max_tokens": 64, "context_window": 128, **kwargs})


def test_policy_prefix_left_padding_and_no_duplicate_bos(tokenizer):
    records = [prompt(), prompt("hello world", "p2")]
    original_side = tokenizer.padding_side
    batch = policy(tokenizer, records)
    assert batch.prompt_ids == ("p1", "p2")
    assert batch.texts[0].endswith("[ASSISTANT]")
    assert batch.attention_mask[0, 0].item() == 0
    assert batch.attention_mask[1, 0].item() == 1
    assert tokenizer.padding_side == original_side
    for index, record in enumerate(records):
        expected = tokenizer.apply_chat_template(
            [{"role": "user", "content": record.messages[0].content}],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        actual = batch.input_ids[index][batch.attention_mask[index].bool()].tolist()
        assert actual == expected
        assert actual.count(tokenizer.bos_token_id) == 1
        assert len(actual) == batch.token_counts[index]
    assert batch.to("cpu").model_inputs().keys() == {"input_ids", "attention_mask"}


def test_reward_full_answer_right_padding_and_context(tokenizer):
    long_prompt = PromptRecord("multi", "hello", (
        Message("user", "hello"), Message("assistant", "earlier"), Message("user", "followup")))
    before = long_prompt.messages
    batch = reward(tokenizer, [prompt(), long_prompt], ["answer", "answer world"])
    assert "earlier" in batch.texts[1]
    assert batch.texts[1].endswith("[ASSISTANT]answer world[EOS]")
    assert batch.attention_mask[0, -1].item() == 0
    assert batch.attention_mask[1, -1].item() == 1
    assert long_prompt.messages == before


def test_masks_keep_real_eos_even_when_padding_uses_eos(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    batch = reward(tokenizer, [prompt(), prompt("hello world", "p2")], ["", "answer"])
    assert batch.attention_mask.sum(dim=1).tolist() == list(batch.token_counts)
    actual = batch.input_ids[0][batch.attention_mask[0].bool()]
    assert actual[-1].item() == tokenizer.eos_token_id
    assert batch.attention_mask[0, -1].item() == 0


def test_boundary_and_reservation(tokenizer):
    record = prompt()
    count = policy(tokenizer, [record]).token_counts[0]
    assert policy(tokenizer, [record], max_prompt_tokens=count, context_window=count + 8).token_counts == (count,)
    with pytest.raises(FormattingError, match="No truncation"):
        policy(tokenizer, [record], max_prompt_tokens=count - 1)
    with pytest.raises(FormattingError, match="reserved"):
        policy(tokenizer, [record], context_window=count + 7)


def test_reward_overlength_does_not_truncate_answer(tokenizer):
    record = prompt()
    count = reward(tokenizer, [record], ["answer world"]).token_counts[0]
    assert reward(tokenizer, [record], ["answer world"], max_tokens=count).token_counts == (count,)
    with pytest.raises(FormattingError, match="Full answers"):
        reward(tokenizer, [record], ["answer world"], max_tokens=count - 1)
    with pytest.raises(FormattingError, match="Full answers"):
        reward(tokenizer, [record], ["answer world"], context_window=count - 1)


def test_skywork_limit_applies_even_with_large_model_context(tokenizer):
    with pytest.raises(FormattingError, match="16384"):
        reward(tokenizer, [prompt()], ["answer " * 16384], max_tokens=32768, context_window=32768)


def test_reward_rejects_system_without_silently_removing_it(tokenizer):
    record = PromptRecord("system", "hello", (Message("system", "instructions"), Message("user", "hello")))
    assert "instructions" in policy(tokenizer, [record]).texts[0]
    with pytest.raises(FormattingError, match="must not include system"):
        reward(tokenizer, [record], ["answer"])


@pytest.mark.parametrize("messages", [
    (), (Message("assistant", "answer"),), (Message("user", ""),),
    (Message("user", "hello"), Message("assistant", "answer")),
    (Message("user", "hello"), Message("system", "middle"), Message("user", "world")),
])
def test_malformed_conversations_fail(tokenizer, messages):
    with pytest.raises(FormattingError):
        policy(tokenizer, [PromptRecord("invalid", "group", messages)])


@pytest.mark.parametrize("kwargs", [{"max_prompt_tokens": 0}, {"max_new_tokens": True},
                                    {"context_window": -1}, {"max_new_tokens": 128}])
def test_invalid_limits_fail(tokenizer, kwargs):
    with pytest.raises(FormattingError):
        policy(tokenizer, [prompt()], **kwargs)


def test_batch_validation_and_missing_template(tokenizer):
    with pytest.raises(FormattingError, match="at least one"):
        policy(tokenizer, [])
    with pytest.raises(FormattingError, match="exactly one"):
        reward(tokenizer, [prompt()], [])
    with pytest.raises(FormattingError, match="exactly one"):
        reward(tokenizer, [prompt()], "answer")
    tokenizer.chat_template = None
    with pytest.raises(FormattingError, match="no chat template"):
        policy(tokenizer, [prompt()])


def test_repeated_schedule_prompts_are_allowed(tokenizer):
    batch = policy(tokenizer, [prompt(), prompt()])
    assert batch.prompt_ids == ("p1", "p1")
    assert torch.equal(batch.input_ids[0], batch.input_ids[1])


@pytest.mark.parametrize("name,is_policy", [
    ("Qwen/Qwen2.5-0.5B-Instruct", True),
    ("Skywork/Skywork-Reward-V2-Qwen3-0.6B", False),
    ("Skywork/Skywork-Reward-V2-Qwen3-4B", False),
])
def test_real_cached_templates_without_network(name, is_policy):
    cache = Path(__file__).resolve().parents[1] / "model_cache"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache, local_files_only=True)
    except OSError:
        pytest.skip("Optional selected-model tokenizer is not cached; no download performed")
    record = prompt("Explain gravity.")
    messages = [{"role": "user", "content": record.messages[0].content}]
    if is_policy:
        batch = policy(tokenizer, [record], max_prompt_tokens=512, context_window=32768)
        assert batch.texts[0].endswith("<|im_start|>assistant\n")
    else:
        answer = "Masses attract each other."
        messages.append({"role": "assistant", "content": answer})
        batch = reward(tokenizer, [record], [answer], max_tokens=4096, context_window=32768)
        assert "<|im_start|>system" not in batch.texts[0]
        assert answer in batch.texts[0]
        assert batch.texts[0].endswith("<|im_end|>\n")
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=is_policy, return_dict=False)
    assert batch.input_ids[0].tolist() == expected
