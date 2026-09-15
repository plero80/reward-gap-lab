import json
from pathlib import Path

import pytest

from reward_gap.config import ConfigError, load_config


@pytest.fixture
def config_file(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    folder = tmp_path / "configs"
    folder.mkdir()
    path = folder / "smoke.json"
    path.write_text(json.dumps({
        "schema_version": 1, "experiment": "smoke", "seeds": [42]
    }), encoding="utf-8")
    return path


def test_defaults_and_paths_do_not_depend_on_working_directory(config_file, monkeypatch, tmp_path):
    monkeypatch.chdir(config_file.parent)
    config = load_config(config_file)
    assert config.seeds == (42,)
    assert config.training.total_updates == 4
    assert config.runtime.output_root == tmp_path / "outputs"
    assert not config.runtime.output_root.exists()
    saved = config.to_dict()
    assert saved["training"]["learning_rate"] == 3e-6
    json.dumps(saved, allow_nan=False)
    config_file.write_text(json.dumps(saved), encoding="utf-8")
    assert load_config(config_file) == config


@pytest.mark.parametrize("patch, message", [
    ({"trainig": {}}, "unknown fields: trainig"),
    ({"training": {"batch_sze": 2}}, "training: unknown fields"),
    ({"runtime": {"gpu": True}}, "runtime: unknown fields"),
    ({"schema_version": 2}, "schema_version"),
    ({"experiment": "typo"}, "experiment"),
    ({"seeds": []}, "seeds"),
    ({"seeds": [42, 42]}, "duplicate"),
    ({"seeds": [True]}, "seeds"),
    ({"training": {"rollout_batch_size": 0}}, "rollout_batch_size"),
    ({"training": {"total_updates": 2}}, "round1_updates"),
    ({"training": {"learning_rate": "0.001"}}, "learning_rate"),
    ({"training": {"learning_rate": float("nan")}}, "learning_rate"),
    ({"training": {"kl_coefficient": -1}}, "kl_coefficient"),
    ({"memory": {"k": 0}}, "memory.k"),
    ({"memory": {"temperature": 0}}, "memory.temperature"),
    ({"training": {"ppo_epochs": 0}}, "ppo_epochs"),
    ({"training": {"minibatch_size": 3}}, "minibatch_size"),
    ({"training": {"clip_range": 1}}, "clip_range"),
    ({"training": {"value_clip_range": -1}}, "value_clip_range"),
    ({"training": {"value_coefficient": -1}}, "value_coefficient"),
    ({"training": {"gamma": 1.1}}, "gamma"),
    ({"training": {"gae_lambda": True}}, "gae_lambda"),
    ({"training": {"max_grad_norm": 0}}, "max_grad_norm"),
    ({"training": {"normalize_advantages": "true"}}, "normalize_advantages"),
    ({"training": {"normalize_advantages": False}}, "normalize_advantages"),
    ({"training": {"rollout_batch_size": 1}}, "rollout_batch_size"),
    ({"training": {"rollout_batch_size": 3, "minibatch_size": 2}}, "divisible"),
    ({"runtime": {"allow_downloads": "false"}}, "allow_downloads"),
    ({"runtime": {"device": "typo"}}, "runtime.device"),
    ({"runtime": {"output_root": ""}}, "output_root"),
    ({"generation": {"suppress_pad_token": "true"}}, "suppress_pad_token"),
])
def test_reject_invalid_settings(config_file, patch, message):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw.update(patch)
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


def test_pad_suppression_is_explicit_and_roundtrips(config_file):
    original = load_config(config_file)
    assert not original.generation.suppress_pad_token
    assert "suppress_pad_token" not in original.to_dict()["generation"]
    raw = original.to_dict()
    raw["generation"]["suppress_pad_token"] = True
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    changed = load_config(config_file)
    assert changed.generation.suppress_pad_token
    assert changed.to_dict()["generation"]["suppress_pad_token"] is True


@pytest.mark.parametrize("contents, message", [
    ("{", "Cannot read configuration"),
    ("[]", "expected a JSON object"),
    ("{}", "missing required field"),
])
def test_bad_document(config_file, contents, message):
    config_file.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


def test_checked_in_smoke_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke.json")
    assert config.experiment == "smoke"
    assert config.runtime.device == "cpu"
    assert config.models.policy.id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.models.proxy.id == "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    assert config.models.judge.id == "Skywork/Skywork-Reward-V2-Qwen3-4B"


def model_references():
    return {role: {"id": f"organization/{role}"} for role in ("policy", "proxy", "judge")}


def test_model_references_roundtrip(config_file):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["models"] = model_references()
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(config_file)
    assert config.models.policy.revision == "main"
    saved = config.to_dict()
    assert saved["models"]["judge"] == {"id": "organization/judge", "revision": "main"}
    config_file.write_text(json.dumps(saved), encoding="utf-8")
    assert load_config(config_file) == config


@pytest.mark.parametrize("models, message", [
    ({}, "provide policy, proxy, and judge"),
    ({**model_references(), "actor": {}}, "unknown fields"),
    ({**model_references(), "policy": {"id": ""}}, "models.policy.id"),
    ({**model_references(), "proxy": {"id": "x", "revison": "main"}}, "unknown fields"),
    ({**model_references(), "judge": {"id": "x", "revision": 12}}, "models.judge.revision"),
    ({**model_references(), "policy": {"revision": "main"}}, "models.policy.id"),
])
def test_invalid_model_references(config_file, models, message):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["models"] = models
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


def test_gpu_smoke_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke_gpu.json")
    assert config.runtime.device == "cuda:0"
    assert config.runtime.dtype == "bfloat16"
    assert config.runtime.allow_downloads is True
    assert config.runtime.model_cache == root / "model_cache"
    assert config.models == load_config(root / "configs" / "smoke.json").models


@pytest.mark.parametrize("runtime, error", [
    ({"device": "cpu", "dtype": "bfloat16"}, "CPU loading requires"),
    ({"device": "cuda:-1"}, "runtime.device"),
    ({"device": "cuda:0", "dtype": "int8"}, "runtime.dtype"),
    ({"model_cache": ""}, "runtime.model_cache"),
])
def test_invalid_model_runtime(config_file, runtime, error):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["runtime"] = runtime
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=error):
        load_config(config_file)


@pytest.mark.parametrize("patch, error", [
    ({"generation": {"max_new_tokens": 0}}, "generation.max_new_tokens"),
    ({"generation": {"max_prompt_tokens": True}}, "generation.max_prompt_tokens"),
    ({"generation": {"max_tokens": 512}}, "unknown fields"),
    ({"scoring": {"max_tokens": 16385}}, "Skywork scoring limit"),
    ({"scoring": {"max_tokens": -1}}, "scoring.max_tokens"),
    ({"scoring": {"batch_size": 0}}, "scoring.batch_size"),
])
def test_formatting_settings_are_validated(config_file, patch, error):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw.update(patch)
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=error):
        load_config(config_file)


@pytest.mark.parametrize("patch, error", [
    ({"policy": {"lora_rank": 0}}, "policy.lora_rank"),
    ({"policy": {"lora_alpha": True}}, "policy.lora_alpha"),
    ({"policy": {"target_modules": ["wrong"]}}, "projection names"),
    ({"policy": {"target_modules": ["q_proj", "q_proj"]}}, "duplicate modules"),
    ({"policy": {"target_modules": []}}, "projection names"),
    ({"generation": {"do_sample": "false"}}, "generation.do_sample"),
])
def test_policy_settings_are_validated(config_file, patch, error):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw.update(patch)
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=error):
        load_config(config_file)
