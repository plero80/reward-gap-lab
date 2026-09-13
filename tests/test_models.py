"""Offline model-loading checks using tiny locally generated checkpoints."""
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch", reason="Install the models extra to run model tests")
pytest.importorskip("transformers", reason="Install the models extra to run model tests")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification, PreTrainedTokenizerFast

from reward_gap.models import ModelLoadError, ModelSpec, LoadOptions, load_policy_model, load_reward_model


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    def create(name, role="policy", labels=1, padding=True):
        path = tmp_path / name
        config = GPT2Config(vocab_size=7, n_positions=16, n_embd=8, n_layer=1, n_head=1,
                            bos_token_id=1, eos_token_id=2, pad_token_id=0 if padding else None,
                            num_labels=labels)
        model = GPT2LMHeadModel(config) if role == "policy" else GPT2ForSequenceClassification(config)
        model.save_pretrained(path)
        backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2,
                                       "[UNK]": 3, "hello": 4, "world": 5, "!": 6}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="[BOS]",
                                            eos_token="[EOS]", unk_token="[UNK]",
                                            pad_token="[PAD]" if padding else None)
        tokenizer.save_pretrained(path)
        return path, model
    return tmp_path, create


def test_policy_real_local_checkpoint_and_generation(checkpoint, monkeypatch):
    root, create = checkpoint
    path, original = create("policy")
    monkeypatch.chdir(root.parent)
    loaded = load_policy_model(ModelSpec(Path("policy")), project_root=root)
    assert loaded.source == str(path)
    assert loaded.revision is None
    assert loaded.tokenizer.padding_side == "left"
    assert not loaded.model.training
    assert not any(p.requires_grad for p in loaded.model.parameters())
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k, v in original.state_dict().items())
    inputs = loaded.tokenizer(["hello", "hello world"], return_tensors="pt", padding=True)
    with torch.inference_mode():
        assert loaded.model(**inputs).logits.shape == (2, 2, 7)
        assert loaded.model.generate(**inputs, max_new_tokens=1, do_sample=False).shape == (2, 3)


def test_reward_real_scalar_checkpoint(checkpoint):
    root, create = checkpoint
    path, _ = create("reward", role="reward")
    loaded = load_reward_model(ModelSpec(path), project_root=root)
    assert not loaded.model.training
    assert not any(p.requires_grad for p in loaded.model.parameters())
    inputs = loaded.tokenizer(["hello", "hello world"], return_tensors="pt", padding=True)
    with torch.inference_mode():
        scores = loaded.model(**inputs).logits
    assert scores.shape == (2, 1)
    assert torch.isfinite(scores).all()


def test_multiclass_reward_is_rejected(checkpoint):
    root, create = checkpoint
    path, _ = create("classifier", role="reward", labels=2)
    with pytest.raises(ModelLoadError, match="single scalar"):
        load_reward_model(ModelSpec(path), project_root=root)


def test_language_model_is_not_accepted_as_reward(checkpoint):
    root, create = checkpoint
    path, _ = create("lm")
    with pytest.raises(ModelLoadError, match="sequence-classification"):
        load_reward_model(ModelSpec(path), project_root=root)


def test_missing_reward_head_cannot_be_randomly_initialized(checkpoint):
    from safetensors.torch import load_file, save_file
    root, create = checkpoint
    path, _ = create("broken_reward", role="reward")
    weights = load_file(path / "model.safetensors")
    del weights["score.weight"]
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ModelLoadError, match="did not load exactly"):
        load_reward_model(ModelSpec(path), project_root=root)


def test_padding_requires_explicit_opt_in(checkpoint):
    root, create = checkpoint
    path, _ = create("unpadded", padding=False)
    with pytest.raises(ModelLoadError, match="no padding token"):
        load_policy_model(ModelSpec(path), project_root=root)
    loaded = load_policy_model(ModelSpec(path), LoadOptions(pad_to_eos=True), project_root=root)
    assert loaded.tokenizer.pad_token_id == loaded.model.config.pad_token_id == 2


def test_padding_mismatch_rejected(checkpoint):
    root, create = checkpoint
    path, _ = create("badpad")
    config = json.loads((path / "config.json").read_text())
    config["pad_token_id"] = 2
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ModelLoadError, match="disagree"):
        load_policy_model(ModelSpec(path), project_root=root)


@pytest.mark.parametrize("options, message", [
    (LoadOptions(dtype="float16"), "CPU loading"),
    (LoadOptions(dtype="wrong"), "dtype must"),
    (LoadOptions(device="meta"), "Only CPU"),
    (LoadOptions(allow_downloads="false"), "booleans"),
])
def test_invalid_options_fail(checkpoint, options, message):
    root, create = checkpoint
    path, _ = create("policy")
    with pytest.raises(ModelLoadError, match=message):
        load_policy_model(ModelSpec(path), options, project_root=root)


def test_missing_local_directory_has_clear_error(checkpoint):
    root, _ = checkpoint
    with pytest.raises(ModelLoadError, match="does not exist"):
        load_policy_model(ModelSpec(Path("missing")), project_root=root)


def test_offline_hub_failure_never_uses_network(checkpoint, monkeypatch):
    import socket
    root, _ = checkpoint
    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network access")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    with pytest.raises(ModelLoadError, match="Cannot load"):
        load_policy_model(ModelSpec("reward-gap-nonexistent/offline-checkpoint"),
                          LoadOptions(cache_dir=root / "empty_cache"), project_root=root)


@pytest.mark.parametrize("role", ["policy", "reward"])
def test_selected_qwen_architectures_on_tiny_local_checkpoints(checkpoint, role):
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForSequenceClassification, Qwen2Tokenizer
    root, create = checkpoint
    path, _ = create("qwen", role=role)
    common = dict(vocab_size=12, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                  pad_token_id=0, bos_token_id=1, eos_token_id=2)
    if role == "policy":
        original = Qwen2ForCausalLM(Qwen2Config(**common, tie_word_embeddings=True))
        loader = load_policy_model
    else:
        original = Qwen3ForSequenceClassification(Qwen3Config(**common, head_dim=4, num_labels=1))
        loader = load_reward_model
    original.save_pretrained(path)
    # Use the actual Qwen tokenizer architecture with a tiny character BPE.
    vocab = {token: i for i, token in enumerate(
        ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "h", "e", "l", "o", "\u0120", "w", "r", "d"])}
    Qwen2Tokenizer(vocab=vocab, merges=[], pad_token="[PAD]", bos_token="[BOS]",
                   eos_token="[EOS]", unk_token="[UNK]").save_pretrained(path)
    loaded = loader(ModelSpec(path), project_root=root)
    inputs = loaded.tokenizer(["hello", "hello world"], return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = loaded.model(**inputs).logits
        if role == "policy":
            assert logits.shape == (2, inputs.input_ids.shape[1], 12)
            assert loaded.model.generate(**inputs, max_new_tokens=1, do_sample=False).shape == (2, inputs.input_ids.shape[1] + 1)
        else:
            assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()
    assert not any(p.requires_grad for p in loaded.model.parameters())


@pytest.mark.parametrize("role", ["policy", "proxy", "judge"])
def test_experiment_role_routes_configured_settings(monkeypatch, role):
    from reward_gap.config import load_config
    from reward_gap.models import load_experiment_model
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke_gpu.json")
    received = []
    def capture(spec, options, **kwargs):
        received.append((spec, options))
        return "loaded"
    def wrong_loader(*args, **kwargs):
        raise AssertionError("Wrong loader for selected role")
    monkeypatch.setattr("reward_gap.models.load_policy_model", capture if role == "policy" else wrong_loader)
    monkeypatch.setattr("reward_gap.models.load_reward_model", wrong_loader if role == "policy" else capture)
    assert load_experiment_model(config, role) == "loaded"
    spec, options = received[0]
    assert spec.source == getattr(config.models, role).id
    assert options.device == "cuda:0"
    assert options.dtype == "bfloat16"
    assert options.allow_downloads is True
    assert options.cache_dir == config.runtime.model_cache
