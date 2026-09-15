import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from test_experiment import setup, loaded, prepared
from reward_gap.hh import HHExperiment
from reward_gap.hh_analysis import HHAnalysis, diagnostics
from reward_gap.experiment import ExperimentError
from reward_gap.ppo import PPOTrainer


def make(setup):
    config, factory, _ = setup
    source = factory()
    return HHExperiment(config, config.runtime.output_root / "unified",
                        actor_factory=source.actor_factory, scorer_factory=source.scorer_factory)


def test_unified_hh_trains_once_compares_memories_on_identical_answers(setup, monkeypatch):
    config, _, calls = setup
    original = PPOTrainer.update
    def update(self, *args, **kwargs):
        calls.append(("update", self.reward_id))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPOTrainer, "update", update)
    exp = make(setup)
    result = exp.run()
    assert result.state == "completed"
    summary = json.loads(result.summary_path.read_text())
    assert len(summary["results"]) == 6
    assert len([c for c in calls if c[0] == "update"]) == 5
    assert summary["predictor_label_counts"] == {"7": {"frozen": 3, "updated": 6}}
    for r in summary["results"].values():
        assert len(r["predictors"]) == 8
        assert {m["count"] for m in r["predictors"].values()} == {3}
        assert r["predictors"]["knn_M0"]["memory_id"] == "M0"
        assert r["predictors"]["knn_M1"]["memory_id"] == "M1-seed-7"
        assert len(json.loads(Path(r["answers"]).read_text())["vectors"]) == 3
    final_calls = [i for i, (role, ids) in enumerate(calls) if role == "judge" and any(p.startswith("final_evaluation") for p in ids)]
    assert min(final_calls) > max(i for i, c in enumerate(calls) if c[0] == "update")
    assert (result.run_dir / "seed-7/raw/round1.pt").is_file()
    assert (result.run_dir / "seed-7/initial.pt").is_file()
    assert (result.run_dir / "report.html").is_file()
    before = len(calls)
    assert make(setup).run().state == "completed"
    assert len(calls) == before


def test_unified_pause_then_analysis_only_does_not_repeat_training(setup, monkeypatch):
    exp = make(setup)
    assert exp.run(until="round1").state == "paused"
    assert not (exp.run_dir / "analysis").exists()
    assert make(setup).run(until="training").state == "paused"
    monkeypatch.setattr(PPOTrainer, "update", lambda *a, **k: pytest.fail("Training repeated"))
    assert make(setup).run().state == "completed"


def test_evaluation_only_on_legacy_coordinator(setup, monkeypatch):
    config, factory, _ = setup
    source = factory()
    source.run(until="training")
    before = {str(p): p.read_bytes() for p in source.run_dir.rglob("*") if p.is_file()}
    monkeypatch.setattr(PPOTrainer, "update", lambda *a, **k: pytest.fail("PPO not permitted"))
    analysis = HHAnalysis(source, config.runtime.output_root / "legacy-analysis")
    summary = analysis.run_analysis()
    assert any("proxy-round1" in reason for reason in summary["missing_checkpoints"])
    assert len(summary["results"]) == 5
    assert before == {str(p): p.read_bytes() for p in source.run_dir.rglob("*") if p.is_file()}
    path = Path(source.status["stages"]["initial-memory"]["result"]["memory"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ExperimentError, match="source or protocol changed"):
        HHAnalysis(source, analysis.run_dir).run_analysis()


def test_analysis_reuses_answers_after_metric_interruption(setup, monkeypatch):
    exp = make(setup)
    exp.run(until="training")
    analysis = HHAnalysis(exp, exp.run_dir / "analysis")
    original = analysis._measure
    monkeypatch.setattr(analysis, "_measure", lambda *a: (_ for _ in ()).throw(RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        analysis.run_analysis()
    answers = list(analysis.run_dir.rglob("answers.json"))
    before = {str(p): p.read_bytes() for p in answers}
    monkeypatch.setattr(analysis, "_measure", original)
    analysis.run_analysis()
    assert all(Path(p).read_bytes() == value for p, value in before.items())


def test_detector_explicitly_marks_all_positive_constant_baseline():
    result = diagnostics([0., 0., 2.], [0., 0., 0.], theta=1., cutoff=0.)
    assert result["flags_every_answer"] and result["recall"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["false_positive"] == 2 and result["true_positive"] == 1


def test_unified_rejects_old_protocol_before_loading_models(setup):
    exp = make(setup)
    exp.run(until="round1")
    protocol = exp.run_dir / "hh_protocol.json"
    protocol.write_text('{}')
    before = len(setup[2])
    with pytest.raises(ExperimentError, match="protocol changed"):
        make(setup).run()
    assert len(setup[2]) == before


def test_update_one_and_round_one_are_retained_separately(setup):
    from dataclasses import replace
    exp = make(setup)
    exp.config = replace(exp.config, training=replace(exp.config.training, round1_updates=2, total_updates=3))
    (exp.prepared / "training_schedule_seed7.json").write_text(json.dumps([["training-0", "training-1"]] * 3))
    exp.run(until="round1")
    for name, expected in (("initial.pt", 0), ("raw/update-1.pt", 1), ("static/update-1.pt", 1),
                           ("raw/round1.pt", 2), ("corrected_round1.pt", 2)):
        payload = torch.load(exp.run_dir / "seed-7" / name, weights_only=True)
        assert payload["update"] == expected
        assert payload["identity"]["response_contract"] == "primary-eos-contiguous-nonpad-v1"
