import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reward_gap import cli
from reward_gap.config import load_config


@pytest.fixture
def config(tmp_path, monkeypatch):
    settings = load_config("configs/smoke.json")
    settings = replace(settings, runtime=replace(settings.runtime, output_root=tmp_path))
    monkeypatch.setattr(cli, "load_config", lambda path: settings)
    return settings


def test_status_reads_saved_state(config, capsys):
    folder = config.runtime.output_root / "run"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"state": "paused"}))
    cli.main(["status", "--config", "unused", "--run-name", "run"])
    assert json.loads(capsys.readouterr().out) == {"state": "paused"}


def test_status_rejects_path_outside_outputs(config, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["status", "--config", "unused", "--run-name", "../outside"])
    assert error.value.code == 1
    assert "inside output_root" in capsys.readouterr().err


def test_run_passes_pause_boundary(config, monkeypatch, capsys):
    pytest.importorskip("trl")
    from reward_gap import experiment
    received = []
    class Runner:
        def __init__(self, settings, folder):
            assert settings is config
            received.append(folder)
        def run(self, *, until):
            received.append(until)
            return SimpleNamespace(state="paused", status_path="status.json")
    monkeypatch.setattr(experiment, "FollowupExperiment", Runner)
    cli.main(["run", "--config", "unused", "--run-name", "run", "--until", "round1"])
    assert received == [config.runtime.output_root / "run", "round1"]
    assert "paused" in capsys.readouterr().out


def test_preflight_failure_returns_nonzero_exit(config, monkeypatch, capsys):
    pytest.importorskip("trl")
    from reward_gap import preflight
    def fail(settings):
        raise preflight.PreflightError("CUDA unavailable; report.json")
    monkeypatch.setattr(preflight, "preflight", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["preflight", "--config", "unused"])
    assert error.value.code == 1
    assert "CUDA unavailable" in capsys.readouterr().err
