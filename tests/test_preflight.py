import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from test_experiment import setup, loaded, prepared
from test_scorers import loaded as scorer_loaded
from reward_gap.preflight import PreflightError, preflight
from reward_gap.scorers import RewardScorer


def test_real_preflight_scores_same_answers_without_changing_weights(setup, scorer_loaded):
    config, factory, _ = setup
    experiment = factory()
    actor = experiment.actor_factory(7)
    config = replace(config, generation=actor.generation)
    original = actor.generation
    before = {name: param.detach().clone() for name, param in actor.named_parameters()}
    seen = []
    class TrackedScorer(RewardScorer):
        def score(self, prompts, answers, **kwargs):
            seen.append((self.role, tuple(p.prompt_id for p in prompts), answers))
            return super().score(prompts, answers, **kwargs)
    result = preflight(config, actor_factory=lambda seed: actor,
                       scorer_factory=lambda role: TrackedScorer(scorer_loaded, role=role))
    report = json.loads(result.report_path.read_text())
    assert result.state == report["state"] == "passed"
    assert seen[0][1:] == seen[1][1:]
    assert all(pid.startswith("training-") for pid in seen[0][1])
    assert all(torch.equal(param, before[name]) for name, param in actor.named_parameters())
    assert actor.generation == original
    assert not list(config.runtime.output_root.rglob("*.pt"))
    assert not (config.runtime.output_root / "preflight-validation").exists()


def test_unavailable_cuda_fails_before_loading_models(setup, monkeypatch):
    config, factory, calls = setup
    config = replace(config, runtime=replace(config.runtime, device="cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PreflightError, match="CUDA is unavailable"):
        preflight(config, actor_factory=factory().actor_factory)
    assert calls == []
    report = json.loads(next(config.runtime.output_root.glob("preflight-*/report.json")).read_text())
    assert report["state"] == "failed"
    assert report["checks"][-1]["name"] == "device"
    assert report["checks"][1]["name"] == "prepared_data"


def test_bad_schedule_fails_before_loading_models(setup):
    config, factory, calls = setup
    (config.data.prepared_dir / "training_schedule_seed7.json").write_text('[["unknown"]]')
    with pytest.raises(PreflightError, match="schedule"):
        preflight(config, actor_factory=factory().actor_factory)
    assert calls == []


def test_overlength_prompts_fail_before_scorer_loading(setup):
    config, factory, calls = setup
    actor = factory().actor_factory(7)
    config = replace(config, generation=replace(actor.generation, max_prompt_tokens=1))
    with pytest.raises(PreflightError, match="prompt_formatting"):
        preflight(config, actor_factory=lambda seed: actor,
                  scorer_factory=lambda role: pytest.fail("Must validate prompt lengths first"))


def test_changed_preparation_metadata_is_rejected(setup):
    config, factory, calls = setup
    (config.data.prepared_dir / "input_manifest.json").write_text(
        json.dumps({"schema_version": 1, "split_seed": config.data.split_seed + 1}))
    with pytest.raises(PreflightError, match="split_seed"):
        preflight(config, actor_factory=factory().actor_factory)
    assert calls == []
