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
    ({"runtime": {"allow_downloads": "false"}}, "allow_downloads"),
    ({"runtime": {"device": "typo"}}, "runtime.device"),
    ({"runtime": {"output_root": ""}}, "output_root"),
])
def test_reject_invalid_settings(config_file, patch, message):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw.update(patch)
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


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
