from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from test_policy import loaded, actor_for
from test_evaluation import prepared, Scorer
from reward_gap.config import ExperimentConfig, TrainingConfig, RuntimeConfig, DataConfig, MemoryConfig
from reward_gap.experiment import FollowupExperiment, ExperimentError
from reward_gap.ppo import PPOTrainer


@pytest.fixture
def setup(loaded, prepared, tmp_path):
    (prepared / "input_manifest.json").write_text(json.dumps({"schema_version": 1}))
    schedule = [["training-0", "training-1"], ["training-1", "training-2"]]
    (prepared / "training_schedule_seed7.json").write_text(json.dumps(schedule))
    config = ExperimentConfig(1, "smoke", (7,),
                              TrainingConfig(round1_updates=1, total_updates=2, ppo_epochs=1,
                                             checkpoint_every=1, learning_rate=.001),
                              RuntimeConfig(output_root=tmp_path / "outputs"),
                              DataConfig(prepared_dir=prepared), memory=MemoryConfig(k=1))
    calls = []
    def actor(seed):
        calls.append(("actor", seed))
        return actor_for(deepcopy(loaded))
    class TrackedScorer(Scorer):
        def __init__(self, role):
            super().__init__(role)
            self.loaded = SimpleNamespace(source=role, revision="p1" if role == "proxy" else "j1")
        def score(self, records, answers, **kwargs):
            calls.append((self.role, tuple(r.prompt_id for r in records)))
            return super().score(records, answers, **kwargs)
    def factory(run_name="run"):
        return FollowupExperiment(config, config.runtime.output_root / run_name,
                                  actor_factory=actor, scorer_factory=TrackedScorer)
    return config, factory, calls


def test_full_experiment_retains_four_checkpoints_and_evaluates_last(setup, monkeypatch):
    config, factory, calls = setup
    original = PPOTrainer.update
    def update(self, *args, **kwargs):
        calls.append(("update", self.reward_id))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPOTrainer, "update", update)
    experiment = factory()
    result = experiment.run()
    assert result.state == "completed"
    checkpoints = sorted(p.relative_to(result.run_dir).as_posix() for p in result.run_dir.rglob("*.pt"))
    assert checkpoints == ["seed-7/corrected_round1.pt", "seed-7/iterative/final.pt",
                           "seed-7/raw/final.pt", "seed-7/static/final.pt"]
    summary = json.loads(result.summary_path.read_text())
    assert set(summary["results"]) == {"seed-7/raw", "seed-7/static", "seed-7/iterative"}
    final_label_indices = [i for i, (role, value) in enumerate(calls)
                           if role == "judge" and any(p.startswith("final_evaluation") for p in value)]
    assert final_label_indices and min(final_label_indices) > max(i for i, c in enumerate(calls) if c[0] == "update")
    assert len([c for c in calls if c[0] == "update"]) == 5
    m0 = json.loads(Path(experiment.status["stages"]["initial-memory"]["result"]["memory"]).read_text())
    m1 = json.loads(Path(experiment.status["stages"]["seed-7/refresh"]["result"]["memory"]).read_text())
    assert len(m0["example_ids"]) == 3 and len(m1["example_ids"]) == 6
    count = len(calls)
    assert factory().run().state == "completed"
    assert len(calls) == count  # Completed runs do not reload any models.


def test_pause_at_round1_and_resume(setup):
    config, factory, calls = setup
    first = factory().run(until="round1")
    assert first.state == "paused"
    assert not any(role == "judge" and any(p.startswith("refresh") for p in value)
                   for role, value in calls if role != "actor")
    assert len(list(first.run_dir.rglob("*.pt"))) == 2
    assert factory().run().state == "completed"
    assert len(list(first.run_dir.rglob("*.pt"))) == 4


def test_failed_training_resumes_latest_checkpoint(setup, monkeypatch):
    config, factory, calls = setup
    # Three updates in round 1 force at least one rolling recovery save.
    config = replace(config, training=replace(config.training, round1_updates=3, total_updates=4))
    path = config.data.prepared_dir / "training_schedule_seed7.json"
    path.write_text(json.dumps([["training-0", "training-1"]] * 4))
    experiment = factory()
    experiment.config = config
    original = PPOTrainer.update
    failed = False
    seen = []
    def interrupted(self, *args, **kwargs):
        nonlocal failed
        seen.append((self.reward_id, self.update_count))
        if self.reward_id.startswith("proxy/") and self.update_count == 1 and not failed:
            failed = True
            raise RuntimeError("test interruption")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPOTrainer, "update", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        experiment.run(until="round1")
    assert list(experiment.run_dir.glob("seed-7/raw/*.pt")) == [experiment.run_dir / "seed-7/raw/latest.pt"]
    assert experiment.run(until="round1").state == "paused"
    assert len([c for c in seen if c[0].startswith("proxy/") and c[1] == 0]) == 1
    assert not list(experiment.run_dir.rglob("latest.pt"))


def test_resume_rejects_changed_config_or_inputs(setup):
    config, factory, calls = setup
    experiment = factory()
    experiment.run(until="round1")
    changed = factory()
    changed.config = replace(config, memory=MemoryConfig(k=2))
    before = len(calls)
    with pytest.raises(ExperimentError, match="changed"):
        changed.run()
    assert len(calls) == before


def test_overlap_rejected_before_model_loading(setup):
    config, factory, calls = setup
    path = config.data.prepared_dir / "training.json"
    rows = json.loads(path.read_text())
    rows[0]["conversation_group"] = "question final_evaluation 0"
    rows[0]["messages"][0]["content"] = "question final_evaluation 0"
    path.write_text(json.dumps(rows))
    with pytest.raises(ExperimentError, match="overlap"):
        factory().run()
    assert not calls


def test_two_seed_training_phase_does_not_read_final_labels(setup):
    config, factory, calls = setup
    config = replace(config, seeds=(7, 8))
    (config.data.prepared_dir / "training_schedule_seed8.json").write_text(
        (config.data.prepared_dir / "training_schedule_seed7.json").read_text())
    experiment = factory()
    experiment.config = config
    result = experiment.run(until="training")
    assert result.state == "paused"
    assert len(list(result.run_dir.rglob("*.pt"))) == 8
    assert not any(role == "judge" and any(p.startswith("final_evaluation") for p in value)
                   for role, value in calls if role != "actor")
    assert len([1 for role, value in calls if role == "judge" and value[0].startswith("calibration")]) == 2


def test_completed_run_rejects_missing_final_artifacts(setup):
    config, factory, calls = setup
    result = factory().run()
    (result.run_dir / "seed-7/static/final.pt").unlink()
    before = len(calls)
    with pytest.raises(ExperimentError, match="missing required artifacts"):
        factory().run()
    assert len(calls) == before
