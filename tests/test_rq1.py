import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("matplotlib")

from test_experiment import setup, loaded, prepared
from reward_gap.experiment import ExperimentError
from reward_gap.rq1 import RQ1Experiment, load_plan


def make(setup, name="rq1", plan=None):
    config, factory, _ = setup
    source = factory()
    return RQ1Experiment(config, config.runtime.output_root / name,
                         plan=plan or load_plan("configs/rq1.json"),
                         actor_factory=source.actor_factory, scorer_factory=source.scorer_factory)


def test_end_to_end_rq1_saves_five_predictors_and_freezes_before_test(setup, monkeypatch):
    config, _, calls = setup
    experiment = make(setup)
    original = experiment._fit
    def fit(*args, **kwargs):
        assert not any(role == "judge" and any(p.startswith("final_evaluation") for p in ids)
                       for role, ids in calls)
        return original(*args, **kwargs)
    monkeypatch.setattr(experiment, "_fit", fit)
    result = experiment.run()
    summary = json.loads(result.summary_path.read_text())
    assert result.state == "completed"
    assert set(summary["results"]["initial"]["metrics"]) == {"knn", "zero_gap", "mean_gap", "ridge_gap", "judge_student"}
    assert summary["judge_labels"] == {"calibration": 3, "predictor_training": 3, "validation": 3, "final_evaluation": 3}
    assert not summary["distribution_shift_evaluated"]
    assert not list(result.run_dir.rglob("*.pt"))
    assert (result.run_dir / "prediction_vs_actual.png").stat().st_size > 1000
    assert (result.run_dir / "checkpoint_metrics.pdf").is_file()
    count = len(calls)
    assert make(setup).run().state == "completed"
    assert len(calls) == count


def test_rq1_retries_failed_test_without_refitting(setup, monkeypatch):
    experiment = make(setup)
    original = experiment._test
    monkeypatch.setattr(experiment, "_test", lambda *a: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.run()
    fitted = Path(experiment.status["stages"]["fit-predictors"]["result"]["predictors"])
    before = fitted.read_bytes()
    monkeypatch.setattr(experiment, "_test", original)
    assert experiment.run().state == "completed"
    assert fitted.read_bytes() == before


def test_changed_plan_cannot_reuse_test_run(setup):
    make(setup).run()
    plan = load_plan("configs/rq1.json")
    plan["theta"] = 2.
    with pytest.raises(ExperimentError, match="plan changed"):
        make(setup, plan=plan).run()


def test_multiple_answers_keep_groups_and_unique_example_ids(setup):
    plan = load_plan("configs/rq1.json")
    plan["answers_per_prompt"] = 2
    experiment = make(setup, plan=plan)
    experiment.run()
    data = json.loads(Path(experiment.status["stages"]["predictor-training"]["result"]["answers"]).read_text())
    assert len(data["rows"]) == len({r["example_id"] for r in data["rows"]}) == 6
    assert len({r["conversation_group"] for r in data["rows"]}) == 3


def test_shift_uses_saved_proxy_and_corrected_policies(setup):
    config, factory, calls = setup
    trained = factory("ppo").run(until="training")
    plan = load_plan("configs/rq1.json")
    plan["checkpoints"] = [{"label": branch, "path": str(trained.run_dir / "seed-7" / branch / "final.pt")}
                           for branch in ("raw", "static")]
    result = make(setup, plan=plan).run()
    summary = json.loads(result.summary_path.read_text())
    assert set(summary["results"]) == {"initial", "raw", "static"}
    assert summary["distribution_shift_evaluated"]
    assert summary["results"]["raw"]["policy"]["update"] == 2
    assert summary["judge_labels"]["final_evaluation"] == 9
    rows = [json.loads(Path(r["predictions"]).read_text()) for r in summary["results"].values()]
    assert all([r["prompt_id"] for r in group] == [r["prompt_id"] for r in rows[0]] for group in rows)
    assert all([r["generation_seed"] for r in group] == [r["generation_seed"] for r in rows[0]] for group in rows)


def test_checkpoint_training_overlap_is_rejected_before_loading(setup, tmp_path):
    config, _, calls = setup
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "final.pt"
    checkpoint.touch()
    cohorts = {name: [] for name in ("calibration", "initial_memory", "training", "refresh", "validation", "final_evaluation")}
    cohorts["training"] = [{"conversation_group": "question final_evaluation 0"}]
    (source / "inputs.json").write_text(json.dumps({"cohorts": cohorts}))
    plan = load_plan("configs/rq1.json")
    plan["checkpoints"] = [{"label": "leaky", "path": str(checkpoint)}]
    with pytest.raises(ExperimentError, match="overlap"):
        make(setup, plan=plan).run()
    assert calls == []


@pytest.mark.parametrize("override", [{"theta": -1}, {"answers_per_prompt": 0},
                                     {"ridge_alphas": [0]}, {"unknown": True}])
def test_plan_rejects_invalid_scientific_settings(tmp_path, override):
    (tmp_path / "pyproject.toml").touch()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ExperimentError):
        load_plan(path)
