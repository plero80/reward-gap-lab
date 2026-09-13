import json
from pathlib import Path

import pytest

from reward_gap.artifacts import atomic_write_json, save_resolved_config
from reward_gap.config import load_config


def test_save_resolved_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke.json")
    path = save_resolved_config(config, tmp_path / "run")
    assert path.name == "resolved_config.json"
    assert json.loads(path.read_text(encoding="utf-8")) == config.to_dict()
    assert list(path.parent.iterdir()) == [path]


def test_replaces_existing_json(tmp_path):
    path = tmp_path / "status.json"
    atomic_write_json(path, {"state": "running"})
    atomic_write_json(path, {"state": "completed"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "completed"}


@pytest.mark.parametrize("bad_data", [{"value": object()}, {"value": float("nan")}])
def test_invalid_json_preserves_previous_file(tmp_path, bad_data):
    path = tmp_path / "saved.json"
    atomic_write_json(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        atomic_write_json(path, bad_data)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "saved.json"
    atomic_write_json(path, {"original": True})
    original = path.read_bytes()

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {"new": True}
        raise PermissionError("simulated replacement failure")

    monkeypatch.setattr("reward_gap.artifacts.os.replace", fail_replace)
    with pytest.raises(PermissionError, match="simulated"):
        atomic_write_json(path, {"new": True})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
