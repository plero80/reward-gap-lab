from copy import deepcopy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

from reward_gap.config import GenerationConfig, PolicyConfig
from reward_gap.data import Message, PromptRecord
from reward_gap.models import LoadedModel
from reward_gap.policy import PPOActor, PolicyError


def nonpad_logits(module, inputs, output):
    # Random tiny models give PAD ordinary probability. Normal integration
    # fixtures generate valid completions; dedicated scripted cases test PAD.
    result = output.clone()
    result[..., 0] = -1e4
    return result


@pytest.fixture
def loaded():
    special = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[USER]", "[ASSISTANT]"]
    vocab = {token: i for i, token in enumerate(special + sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], pad_token="[PAD]", bos_token="[BOS]",
                               eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[USER]", "[ASSISTANT]"]})
    tokenizer.chat_template = (
        "{{ bos_token }}{% for m in messages %}{{ '[' + m['role'].upper() + ']' }}"
        "{{ m['content'] }}{{ eos_token }}{% endfor %}"
        "{% if add_generation_prompt %}[ASSISTANT]{% endif %}"
    )
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=128, tie_word_embeddings=True,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         bos_token_id=tokenizer.bos_token_id, attention_dropout=0.2)
    with torch.random.fork_rng():
        torch.manual_seed(5)
        model = Qwen2ForCausalLM(config)
    model.lm_head.register_forward_hook(nonpad_logits)
    return LoadedModel(model, tokenizer, "tiny-local-qwen2", None, "policy")


def actor_for(loaded, *, do_sample=True):
    return PPOActor(loaded, policy=PolicyConfig(lora_rank=2, lora_alpha=4, target_modules=("q_proj", "v_proj")),
                    generation=GenerationConfig(max_prompt_tokens=64, max_new_tokens=4, do_sample=do_sample), seed=42)


def prompts():
    return [PromptRecord("short", "hi", (Message("user", "hi"),)),
            PromptRecord("long", "hello world", (Message("user", "hello world"),))]


def scripted_generation(actor, monkeypatch, suffix):
    def generate(**kwargs):
        settings = kwargs["generation_config"]
        assert settings.top_k == 0 and settings.top_p == 1 and settings.temperature == 1
        ids = kwargs["input_ids"]
        return torch.cat((ids, torch.tensor(suffix, device=ids.device)), dim=1)
    monkeypatch.setattr(actor.model, "generate", generate)


def test_only_lora_and_value_head_are_trainable_and_rng_preserved(loaded):
    before = torch.random.get_rng_state().clone()
    actor = actor_for(loaded)
    assert torch.equal(before, torch.random.get_rng_state())
    trainable = [name for name, p in actor.named_parameters() if p.requires_grad]
    assert any("lora_" in name for name in trainable)
    assert "value_head.weight" in trainable and "value_head.bias" in trainable
    assert all("lora_" in name or name.startswith("value_head.") for name in trainable)


@pytest.mark.parametrize("pad_is_eos", [False, True])
def test_eos_masks_empty_completion_and_length_stop(loaded, monkeypatch, pad_is_eos):
    if pad_is_eos:
        loaded.tokenizer.pad_token = loaded.tokenizer.eos_token
        loaded.model.config.pad_token_id = loaded.tokenizer.pad_token_id
        with pytest.raises(PolicyError, match="distinct from PAD"):
            actor_for(loaded)
        return
    actor = actor_for(loaded)
    eos, pad = loaded.tokenizer.eos_token_id, loaded.tokenizer.pad_token_id
    token = loaded.tokenizer.encode("a", add_special_tokens=False)[0]
    scripted_generation(actor, monkeypatch, [[eos, pad, pad, pad], [token, token, token, token]])
    batch = actor.generate(prompts(), seed=9)
    assert batch.answers == ("", "aaaa")
    assert batch.response_lengths == (1, 4)
    assert batch.finish_reasons == ("eos", "length")
    assert batch.response_mask[:, batch.prompt_width:].tolist() == [[True, False, False, False], [True] * 4]
    assert not batch.response_mask[:, :batch.prompt_width].any()
    assert batch.attention_mask[0, 0] == 0
    assert not batch.sequences.is_inference()


def test_sampling_repeatable_and_rng_restored(loaded):
    actor = actor_for(loaded)
    before = torch.random.get_rng_state().clone()
    first = actor.generate(prompts(), seed=123)
    second = actor.generate(prompts(), seed=123)
    assert torch.equal(first.sequences, second.sequences)
    assert torch.equal(before, torch.random.get_rng_state())
    assert first.sampled is True
    assert all(1 <= n <= 4 for n in first.response_lengths)


def test_greedy_generation_is_marked_for_evaluation(loaded):
    actor = actor_for(loaded, do_sample=False)
    assert actor.generate(prompts(), seed=1).sampled is False


def test_statistics_alignment_and_bootstrap(loaded, monkeypatch):
    actor = actor_for(loaded)
    token = loaded.tokenizer.encode("a", add_special_tokens=False)[0]
    eos, pad = loaded.tokenizer.eos_token_id, loaded.tokenizer.pad_token_id
    scripted_generation(actor, monkeypatch, [[token, eos, pad, pad], [token] * 4])
    rollout = actor.generate(prompts(), seed=1)
    with torch.no_grad():
        actor.value_head.bias.fill_(2.0)
    stats = actor.statistics(rollout)
    inputs, mask = actor._inputs(rollout)
    native = actor.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    for row in range(2):
        for j in range(rollout.response_lengths[row]):
            position = rollout.prompt_width + j - 1
            expected = torch.log_softmax(native.logits[row, position].float(), dim=-1)[rollout.response_ids[row, j]]
            assert stats.log_probs[row, j].item() == pytest.approx(expected.item(), abs=1e-6)
    assert stats.values[mask].tolist() == [2.0] * 6
    assert stats.values[~mask].eq(0).all()
    assert stats.log_probs[~mask].eq(0).all()
    assert stats.bootstrap_values.tolist() == [0.0, 2.0]


def test_gradients_update_only_adapter_and_value_head(loaded, monkeypatch):
    actor = actor_for(loaded)
    token = loaded.tokenizer.encode("a", add_special_tokens=False)[0]
    scripted_generation(actor, monkeypatch, [[token] * 4, [token] * 4])
    rollout = actor.generate(prompts(), seed=7)
    frozen = {name: p.detach().clone() for name, p in actor.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=0.01)
    stats = actor.statistics(rollout)
    loss = -stats.log_probs[stats.response_mask].mean() + (stats.values[stats.response_mask] - 1).square().mean()
    loss.backward()
    assert actor.value_head.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in actor.named_parameters() if "lora_B" in name)
    assert all(p.grad is None for name, p in actor.named_parameters() if name in frozen)
    optimizer.step()
    assert all(torch.equal(p, frozen[name]) for name, p in actor.named_parameters() if name in frozen)


def test_disabled_adapter_matches_original_base_after_adapter_changes(loaded, monkeypatch):
    base = deepcopy(loaded.model).eval()
    actor = actor_for(loaded)
    token = loaded.tokenizer.encode("a", add_special_tokens=False)[0]
    scripted_generation(actor, monkeypatch, [[token] * 4, [token] * 4])
    rollout = actor.generate(prompts(), seed=1)
    inputs, mask = actor._inputs(rollout)
    with torch.no_grad():
        output = base(**inputs, use_cache=False, return_dict=True)
        expected = actor._log_probs(output.logits[:, rollout.prompt_width - 1:-1], rollout.response_ids, mask)
        for name, parameter in actor.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.5)
    reference = actor.reference_log_probs(rollout)
    assert not reference.requires_grad
    assert torch.allclose(reference, expected, atol=1e-6)
    active = actor.statistics(rollout).log_probs
    assert not torch.allclose(active, reference, atol=1e-6)
    assert torch.allclose(actor.reference_log_probs(rollout), expected, atol=1e-6)


def test_single_and_left_padded_batch_statistics_match(loaded, monkeypatch):
    actor = actor_for(loaded)
    token = loaded.tokenizer.encode("a", add_special_tokens=False)[0]
    scripted_generation(actor, monkeypatch, [[token] * 4, [token] * 4])
    batch = actor.generate(prompts(), seed=1)
    batch_stats = actor.statistics(batch)
    for i, prompt in enumerate(prompts()):
        scripted_generation(actor, monkeypatch, [[token] * 4])
        single = actor.generate([prompt], seed=1)
        assert torch.allclose(actor.statistics(single).log_probs[0], batch_stats.log_probs[i], atol=1e-6)


def test_invalid_rollout_and_double_adapter_fail(loaded, monkeypatch):
    actor = actor_for(loaded)
    with pytest.raises(PolicyError, match="already has an adapter"):
        actor_for(loaded)
    token = loaded.tokenizer.encode("a", add_special_tokens=False)[0]
    scripted_generation(actor, monkeypatch, [[token] * 4, [token] * 4])
    rollout = actor.generate(prompts(), seed=1)
    with pytest.raises(PolicyError, match="shapes"):
        actor.statistics(replace(rollout, prompt_width=0))
    with pytest.raises(PolicyError, match="response lengths"):
        actor.statistics(replace(rollout, response_lengths=(1, 1)))
    with pytest.raises(PolicyError, match="Finish reason"):
        actor.statistics(replace(rollout, finish_reasons=("eos", "eos")))


def test_recomputed_log_probs_match_distribution_used_for_sampling(loaded, monkeypatch):
    actor = actor_for(loaded)
    original = actor.model.generate
    captured = []
    def capture(**kwargs):
        result = original(**kwargs, return_dict_in_generate=True, output_scores=True)
        captured.append(result)
        return result.sequences
    monkeypatch.setattr(actor.model, "generate", capture)
    rollout = actor.generate(prompts(), seed=12)
    with torch.no_grad():
        stats = actor.statistics(rollout)
    for step, logits in enumerate(captured[0].scores):
        expected = torch.log_softmax(logits.float(), dim=-1).gather(1, rollout.response_ids[:, step:step + 1]).squeeze(1)
        active = stats.response_mask[:, step]
        assert torch.allclose(stats.log_probs[active, step], expected[active], atol=1e-6)


def test_primary_eos_matches_ppo_despite_pretrained_eos_list(loaded, monkeypatch):
    alternative = loaded.tokenizer.encode("!", add_special_tokens=False)[0]
    loaded.model.generation_config.eos_token_id = [loaded.tokenizer.eos_token_id, alternative]
    actor = actor_for(loaded)
    eos, pad = loaded.tokenizer.eos_token_id, loaded.tokenizer.pad_token_id
    token = loaded.tokenizer.encode("a", add_special_tokens=False)[0]
    scripted_generation(actor, monkeypatch, [[alternative, token, eos, pad], [token, alternative, token, eos]])
    rollout = actor.generate(prompts(), seed=1)
    assert actor.eos_ids == (eos,)
    assert rollout.response_lengths == (3, 4)
    assert rollout.finish_reasons == ("eos", "eos")
    assert actor.statistics(rollout).bootstrap_values.eq(0).all()


def test_standalone_generation_rejects_pad_before_eos(loaded, monkeypatch):
    from reward_gap.failures import SampleError
    actor = actor_for(loaded)
    pad, eos = actor.tokenizer.pad_token_id, actor.tokenizer.eos_token_id
    token = actor.tokenizer.encode("a", add_special_tokens=False)[0]
    scripted_generation(actor, monkeypatch, [[token, pad, token, eos], [token, token, token, eos]])
    with pytest.raises(SampleError, match="PAD"):
        actor.generate(prompts(), seed=1)
