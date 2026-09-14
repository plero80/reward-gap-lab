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
                         plan=plan or load_plan("configs/rq1_initial.json"),
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
    plan = load_plan("configs/rq1_initial.json")
    plan["theta"] = 2.
    with pytest.raises(ExperimentError, match="plan changed"):
        make(setup, plan=plan).run()


def test_multiple_answers_keep_groups_and_unique_example_ids(setup):
    plan = load_plan("configs/rq1_initial.json")
    plan["answers_per_prompt"] = 2
    experiment = make(setup, plan=plan)
    experiment.run()
    data = json.loads(Path(experiment.status["stages"]["predictor-training"]["result"]["answers"]).read_text())
    assert len(data["rows"]) == len({r["example_id"] for r in data["rows"]}) == 6
    assert len({r["conversation_group"] for r in data["rows"]}) == 3


def test_shift_uses_saved_proxy_and_corrected_policies(setup):
    config, factory, calls = setup
    trained = factory("ppo").run(until="training")
    plan = load_plan("configs/rq1_initial.json")
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
    plan = load_plan("configs/rq1_initial.json")
    plan["checkpoints"] = [{"label": "leaky", "path": str(checkpoint)}]
    with pytest.raises(ExperimentError, match="overlap"):
        make(setup, plan=plan).run()
    assert calls == []


@pytest.mark.parametrize("override", [{"theta": -1}, {"answers_per_prompt": 0},
                                     {"ridge_alphas": [0]}, {"unknown": True}, {"train_ppo": 1},
                                     {"train_ppo": True, "ppo_evaluation_updates": [2, 1]},
                                     {"train_ppo": True, "ppo_evaluation_updates": [1, 1]},
                                     {"train_ppo": True, "ppo_evaluation_updates": [True]},
                                     {"ppo_evaluation_updates": [1]},
                                     {"train_ppo": True, "checkpoints": [{"label": "external", "path": "x.pt"}]}])
def test_plan_rejects_invalid_scientific_settings(tmp_path, override):
    (tmp_path / "pyproject.toml").touch()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ExperimentError):
        load_plan(path)


def integrated(setup):
    return make(setup, plan=load_plan("configs/rq1.json"))


def test_integrated_ppo_uses_frozen_memory_equal_starts_and_disjoint_prompts(setup, monkeypatch):
    from reward_gap.ppo import PPOTrainer
    config, _, calls = setup
    original = PPOTrainer.update
    starts, schedules, memories = [], {}, []
    experiment = integrated(setup)
    frozen = {}
    original_fit = experiment._fit
    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        path = Path(result["predictors"])
        fitted = json.loads(path.read_text())
        frozen.update({path: path.read_bytes(), Path(fitted["memory"]): Path(fitted["memory"]).read_bytes()})
        return result
    monkeypatch.setattr(experiment, "_fit", fit)
    def update(self, prompts, *, rollout_seed):
        assert frozen and all(p.read_bytes() == value for p, value in frozen.items())
        assert all(p.prompt_id.startswith("training") for p in prompts)
        arm = self.reward_id.split("/")[0]
        schedules.setdefault(arm, []).append(([p.prompt_id for p in prompts], rollout_seed))
        if self.update_count == 0:
            starts.append({n: p.detach().clone() for n, p in self.actor.named_parameters() if p.requires_grad})
        if arm == "knn":
            memories.append(self.reward.memory)
        return original(self, prompts, rollout_seed=rollout_seed)
    monkeypatch.setattr(PPOTrainer, "update", update)
    result = experiment.run()
    summary = json.loads(result.summary_path.read_text())
    assert set(summary["results"]) == {"initial", "proxy-update-1", "proxy-update-2", "corrected-update-1", "corrected-update-2"}
    assert summary["distribution_shift_evaluated"] and summary["ppo_training"]["performed"]
    assert summary["ppo_training"]["evaluation_updates"] == [1, 2]
    assert summary["judge_labels"]["final_evaluation"] == 15
    assert len(starts) == 2 and schedules["proxy"] == schedules["knn"]
    for name in starts[0]:
        torch.testing.assert_close(starts[0][name], starts[1][name], rtol=0, atol=0)
    assert memories and all(m is memories[0] for m in memories)
    assert all(p.read_bytes() == value for p, value in frozen.items())
    assert sorted(p.relative_to(result.run_dir).as_posix() for p in result.run_dir.rglob("*.pt")) == [
        "seed-7/rq1-corrected/final.pt", "seed-7/rq1-proxy/final.pt"]
    rows = [json.loads(Path(r["predictions"]).read_text()) for r in summary["results"].values()]
    assert all([r["prompt_id"] for r in batch] == [r["prompt_id"] for r in rows[0]] for batch in rows)
    assert all([r["generation_seed"] for r in batch] == [r["generation_seed"] for r in rows[0]] for batch in rows)
    assert not summary["results"]["proxy-update-1"]["policy"]["checkpoint_retained"]
    before = len(calls)
    assert integrated(setup).run().state == "completed" and len(calls) == before


def test_integrated_resume_reuses_boundary_after_failed_evaluation(setup, monkeypatch):
    from reward_gap.ppo import PPOTrainer
    experiment = integrated(setup)
    original = experiment._test
    updates = []
    original_update = PPOTrainer.update
    def update(self, *args, **kwargs):
        updates.append((self.reward_id, self.update_count))
        return original_update(self, *args, **kwargs)
    monkeypatch.setattr(PPOTrainer, "update", update)
    def test(folder, data, fitted, policy):
        if policy["label"] == "proxy-update-1":
            raise RuntimeError("evaluation interrupted")
        return original(folder, data, fitted, policy)
    monkeypatch.setattr(experiment, "_test", test)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        experiment.run()
    predictor = Path(experiment.status["stages"]["fit-predictors"]["result"]["predictors"])
    snapshot = predictor.read_bytes()
    assert len(updates) == 1
    assert integrated(setup).run().state == "completed"
    assert len(updates) == 4  # Already completed first update was not rerun.
    assert predictor.read_bytes() == snapshot


def test_integrated_training_failure_resumes_prior_boundary(setup, monkeypatch):
    from reward_gap.ppo import PPOTrainer
    original = PPOTrainer.update
    updates, failed = [], False
    def update(self, *args, **kwargs):
        nonlocal failed
        if self.reward_id.startswith("knn/") and self.update_count == 1 and not failed:
            failed = True
            raise RuntimeError("training interrupted")
        updates.append((self.reward_id, self.update_count))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPOTrainer, "update", update)
    with pytest.raises(RuntimeError, match="training interrupted"):
        integrated(setup).run()
    result = integrated(setup).run()
    assert result.state == "completed" and len(updates) == 4
    assert len(list(result.run_dir.rglob("*.pt"))) == 2


def test_integrated_plan_rejects_update_outside_budget_before_loading(setup):
    _, _, calls = setup
    plan = load_plan("configs/rq1.json")
    plan["ppo_evaluation_updates"] = [3]
    with pytest.raises(ExperimentError, match="exceeds"):
        make(setup, plan=plan)
    assert calls == []
