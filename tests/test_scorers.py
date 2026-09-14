from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3ForSequenceClassification

from reward_gap.config import load_config
from reward_gap.data import Message, PromptRecord
from reward_gap.formatting import FormattingError, format_reward_batch
from reward_gap.models import LoadedModel
from reward_gap.scorers import RewardScorer, ScoringError


@pytest.fixture
def loaded():
    special = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[USER]", "[ASSISTANT]"]
    vocab = {token: index for index, token in enumerate(special + sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], pad_token="[PAD]", bos_token="[BOS]",
                               eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[USER]", "[ASSISTANT]"]})
    tokenizer.chat_template = (
        "{{ bos_token }}{% for m in messages %}"
        "{{ '[' + m['role'].upper() + ']' }}{{ m['content'] }}{{ eos_token }}{% endfor %}"
    )
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=256, num_labels=1,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         attention_dropout=0.2)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = Qwen3ForSequenceClassification(config)
    return LoadedModel(model, tokenizer, "tiny-local-qwen3", None, "reward")


def prompts():
    return [PromptRecord("short", "hello", (Message("user", "hello"),)),
            PromptRecord("long", "hello world", (Message("user", "hello world"),))]


def test_real_scorer_integrates_with_both_reward_strategies(loaded):
    from reward_gap.calibration import FrozenCalibration, ScoreScale
    from reward_gap.memory import GapMemory, MemoryContext
    from reward_gap.rewards import KNNReward, ProxyReward

    loaded = replace(loaded, revision="tiny-test-v1")
    scorer = RewardScorer(loaded, role="proxy")
    records, answers = prompts(), ["answer", "longer answer"]
    scored = scorer.score(records, answers, return_embeddings=True)
    scale = ScoreScale(loaded.source, loaded.revision, 0., 1.)
    calibration = FrozenCalibration("test-calibration", scale, scale)
    context = MemoryContext(scored.source, scored.revision, scored.embedding_pooling, calibration.calibration_id)
    memory = GapMemory(["first", "second"], scored.embeddings, [0.5, -0.25], context=context, k=1)
    baseline = ProxyReward(scorer, calibration).score(records, answers)
    corrected = KNNReward(scorer, calibration, memory).score(records, answers)
    assert baseline.rewards == pytest.approx(scored.scores)
    assert corrected.rewards == pytest.approx((scored.scores[0] - 0.5, scored.scores[1] + 0.25))
    assert corrected.neighbors.neighbor_ids == (("first",), ("second",))


def test_batched_scores_and_embeddings_match_single_answers(loaded):
    scorer = RewardScorer(loaded, role="proxy", batch_size=2)
    answers = ["answer", "a longer answer"]
    batch = scorer.score(prompts(), answers, return_embeddings=True)
    assert batch.prompt_ids == ("short", "long")
    assert batch.role == "proxy"
    assert batch.source == loaded.source
    assert batch.embeddings.shape == (2, 16)
    assert batch.embeddings.device.type == "cpu"
    assert batch.embeddings.dtype == torch.float32
    assert not batch.embeddings.requires_grad
    assert torch.allclose(torch.linalg.vector_norm(batch.embeddings, dim=1), torch.ones(2), atol=1e-6)
    for index, record in enumerate(prompts()):
        single = scorer.score([record], [answers[index]], return_embeddings=True)
        assert batch.scores[index] == pytest.approx(single.scores[0], abs=1e-6)
        assert batch.token_counts[index] == single.token_counts[0]
        assert torch.allclose(batch.embeddings[index], single.embeddings[0], atol=1e-6)


def test_native_scalar_output_and_embedding_pooling_agree(loaded):
    scorer = RewardScorer(loaded, role="proxy")
    answers = ["answer", "longer answer"]
    formatted = format_reward_batch(loaded.tokenizer, prompts(), answers, max_tokens=4096, context_window=256)
    with torch.inference_mode():
        native = loaded.model(**formatted.model_inputs(), output_hidden_states=True, use_cache=False)
        last = formatted.attention_mask.sum(dim=1) - 1
        vectors = native.hidden_states[-1][torch.arange(2), last]
        expected = torch.nn.functional.normalize(vectors.float(), dim=1)
    result = scorer.score(prompts(), answers, return_embeddings=True)
    assert result.scores == pytest.approx(native.logits[:, 0].tolist(), abs=1e-6)
    assert torch.allclose(result.embeddings, expected, atol=1e-6)
    assert result.embedding_pooling == "last_non_pad_final_hidden_l2_v1"


def test_scorer_stays_frozen_and_disables_gradient_recording(loaded, monkeypatch):
    scorer = RewardScorer(loaded, role="proxy")
    before = {name: tensor.clone() for name, tensor in loaded.model.state_dict().items()}
    original = loaded.model.forward
    calls = []
    def forward(*args, **kwargs):
        calls.append(kwargs)
        assert torch.is_inference_mode_enabled()
        assert not loaded.model.training
        return original(*args, **kwargs)
    monkeypatch.setattr(loaded.model, "forward", forward)
    loaded.model.train()
    loaded.model.requires_grad_(True)
    first = scorer.score(prompts(), ["answer", "answer"])
    second = scorer.score(prompts(), ["answer", "answer"])
    assert first.scores == second.scores
    assert all(call["use_cache"] is False and call["output_hidden_states"] is False for call in calls)
    assert first.embeddings is None and first.embedding_pooling is None
    assert not any(p.requires_grad or p.grad is not None for p in loaded.model.parameters())
    assert all(torch.equal(tensor, loaded.model.state_dict()[name]) for name, tensor in before.items())


def test_microbatch_order_repeated_prompts_and_empty_answer(loaded):
    scorer = RewardScorer(loaded, role="proxy", batch_size=1)
    records = [prompts()[1], prompts()[0], prompts()[1]]
    answers = ["long", "", "other"]
    small = scorer.score(records, answers, return_embeddings=True)
    larger = RewardScorer(loaded, role="proxy", batch_size=3).score(records, answers, return_embeddings=True)
    assert small.prompt_ids == ("long", "short", "long")
    assert small.scores == pytest.approx(larger.scores, abs=1e-6)
    assert torch.allclose(small.embeddings, larger.embeddings, atol=1e-6)


def test_judge_scores_but_cannot_supply_proxy_memory(loaded):
    scorer = RewardScorer(loaded, role="judge")
    assert len(scorer.score(prompts(), ["one", "two"]).scores) == 2
    with pytest.raises(ScoringError, match="proxy, not the judge"):
        scorer.score(prompts(), ["one", "two"], return_embeddings=True)


def test_bad_input_fails_before_forward(loaded, monkeypatch):
    scorer = RewardScorer(loaded, role="proxy", max_tokens=1)
    def never(*args, **kwargs):
        raise AssertionError("Model must not run for invalid input")
    monkeypatch.setattr(loaded.model, "forward", never)
    with pytest.raises(ScoringError, match="exactly one"):
        scorer.score(prompts(), ["one"])
    with pytest.raises(ScoringError, match="nonempty"):
        scorer.score([], [])
    with pytest.raises(FormattingError, match="no truncation"):
        scorer.score(prompts(), ["one", "two"])


@pytest.mark.parametrize("kind", ["nan", "shape", "zero_embedding"])
def test_invalid_model_outputs_fail(loaded, monkeypatch, kind):
    scorer = RewardScorer(loaded, role="proxy")
    def broken(**kwargs):
        ids = kwargs["input_ids"]
        logits = torch.zeros((ids.shape[0], 2 if kind == "shape" else 1))
        if kind == "nan":
            logits[:] = float("nan")
        return SimpleNamespace(logits=logits, hidden_states=(torch.zeros((*ids.shape, 16)),))
    monkeypatch.setattr(loaded.model, "forward", broken)
    with pytest.raises(ScoringError):
        scorer.score(prompts(), ["one", "two"], return_embeddings=kind == "zero_embedding")


@pytest.mark.parametrize("settings", [{"batch_size": 0}, {"max_tokens": 16385}, {"role": "policy"}])
def test_invalid_settings(loaded, settings):
    with pytest.raises(ScoringError):
        RewardScorer(loaded, **{"role": "proxy", **settings})


def test_causal_policy_cannot_be_wrapped_as_reward(loaded):
    with pytest.raises(ScoringError, match="reward models"):
        RewardScorer(replace(loaded, role="policy"), role="proxy")


def test_load_uses_configured_role_and_settings(loaded, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.json")
    calls = []
    def load(actual_config, role):
        calls.append((actual_config, role))
        return loaded
    monkeypatch.setattr("reward_gap.scorers.load_experiment_model", load)
    scorer = RewardScorer.load(config, "judge")
    assert calls == [(config, "judge")]
    assert scorer.max_tokens == config.scoring.max_tokens
    assert scorer.batch_size == config.scoring.batch_size
