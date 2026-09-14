"""Matched-teacher invariants and actual CPU TRL/mixture-of-experts checks."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from test_gsm8k_integration import setup, loaded
from reward_gap.gsm8k.config import load_gsm_config
from reward_gap.gsm8k.teachers import TeacherExperiment
from reward_gap.ppo import PPOTrainer


def make(setup):
    config, original, calls = setup
    source = original()
    config = replace(config, settings={**config.settings, "teacher_comparison": {
        "teacher30": {"id": "test-30b", "revision": "v30"}, "k": 1, "temperature": .05},
        "k_values": [1], "similarity_temperatures": [.05]})
    live, inputs = [], {}
    class Teacher:
        def __init__(self, name):
            self.name, self.phase = name, ""
            self.loaded = SimpleNamespace(source=f"test-{name}", revision=f"version-{name}")
        def score(self, records, answers, *, return_embeddings=False):
            assert not return_embeddings
            from reward_gap.scorers import ScoreBatch
            inputs.setdefault(self.name, []).extend((p.prompt_id, a) for p, a in zip(records, answers, strict=True))
            numbers = [int(p.prompt_id.rsplit("-", 1)[1]) for p in records]
            grades = tuple(float(1 + n if self.name == "4b" else 5 - n) for n in numbers)
            return ScoreBatch(tuple(p.prompt_id for p in records), grades, (10,) * len(records),
                              "judge", self.loaded.source, self.loaded.revision)
    def teacher(name):
        assert all(ref() is None for ref in live)  # Teachers never coexist.
        result = Teacher(name)
        live.append(weakref.ref(result))
        return result
    def factory(cfg=None):
        return TeacherExperiment(cfg or config, config.base.runtime.output_root / "teachers",
            actor_factory=source.actor_factory, scorer_factory=source.scorer_factory, teacher_factory=teacher)
    return config, factory, live, inputs


def test_teacher_preparation_shares_exact_inputs_vectors_and_proxy_scale(setup):
    _, factory, live, inputs = make(setup)
    experiment = factory()
    result = experiment.run(until="preparation")
    assert result.state == "paused" and all(ref() is None for ref in live)
    assert inputs["4b"] == inputs["30b"]
    c4, m4, _ = experiment.preparations[7]["4b"]
    c30, m30, _ = experiment.preparations[7]["30b"]
    assert c4.proxy == c30.proxy and c4.judge.source != c30.judge.source
    assert c4.calibration_id != c30.calibration_id
    assert torch.equal(m4.vectors, m30.vectors)
    assert [r["gap"] for r in m4.rows] != [r["gap"] for r in m30.rows]
    query = torch.tensor([[1., 2.]])
    _, n4 = m4.predict(query, ["new"], context=m4.context)
    _, n30 = m30.predict(query, ["new"], context=m30.context)
    assert n4 == n30
    blind = json.loads((result.run_dir / "blinded_review.json").read_text())
    assert blind and "teacher" not in blind[0] and "teacher_identity" not in blind[0]
    blind[0]["notes"] = "Reviewer notes must survive resume."
    (result.run_dir / "blinded_review.json").write_text(json.dumps(blind))
    assert len(json.loads((result.run_dir / "teacher_checks.json").read_text())) == 2
    count = len(inputs["30b"])
    assert factory().run(until="preparation").state == "paused"
    assert len(inputs["30b"]) == count
    assert json.loads((result.run_dir / "blinded_review.json").read_text())[0]["notes"] == "Reviewer notes must survive resume."
    assert not list(result.run_dir.rglob("*.pt"))


def test_actual_teacher_ppo_has_equal_starts_scoped_judge_and_paired_final_results(setup, monkeypatch):
    _, factory, live, inputs = make(setup)
    original = PPOTrainer.update
    starts, visits = [], {}
    def update(self, prompts, *, rollout_seed):
        proxy_calls = len([c for c in setup[2] if c[0] == "proxy"])
        teacher_calls = len(inputs.get("4b", []))
        current = [ref() for ref in live if ref() is not None]
        if self.reward_id.startswith("judge4/"):
            assert len(current) == 1 and current[0].name == "4b"
        else:
            assert not current
        if self.update_count == 0:
            starts.append({n: p.detach().clone() for n, p in self.actor.named_parameters() if p.requires_grad})
        visits.setdefault(self.reward_id.split("/")[0], []).append(([p.prompt_id for p in prompts], rollout_seed))
        result = original(self, prompts, rollout_seed=rollout_seed)
        if self.reward_id.startswith("judge4/"):
            assert len(inputs["4b"]) - teacher_calls == len(prompts)
            assert len([c for c in setup[2] if c[0] == "proxy"]) == proxy_calls
        else:
            assert len(inputs.get("4b", [])) == teacher_calls
        return result
    monkeypatch.setattr(PPOTrainer, "update", update)
    experiment = factory()
    result = experiment.run()
    assert result.state == "completed" and len(starts) == 4
    assert all(ref() is None for ref in live)
    for state in starts[1:]:
        for name in starts[0]:
            torch.testing.assert_close(state[name], starts[0][name], atol=0, rtol=0)
    assert visits["proxy"] == visits["judge4"] == visits["knn4"] == visits["knn30"]
    summary = json.loads(result.summary_path.read_text())
    assert set(summary["aggregate_final"]) == {"base", "proxy", "judge4", "knn4", "knn30"}
    assert summary["paired_knn30_minus_knn4"][0]["seed"] == 7
    assert summary["paired_difference_sample_std"] is None
    assert all(r["teacher_grades"] == "not_computed" for r in summary["results"])
    assert sorted(p.relative_to(result.run_dir).as_posix() for p in result.run_dir.rglob("*.pt")) == [
        "seed-7/judge4/final.pt", "seed-7/knn30/final.pt", "seed-7/knn4/final.pt", "seed-7/proxy/final.pt"]
    judge_rows = json.loads((result.run_dir / "seed-7/judge4/rollouts/update-000001.json").read_text())
    calibration = experiment.preparations[7]["4b"][0]
    for row in judge_rows:
        assert row["arm"] == "judge" and row["condition"] == "judge4"
        assert row["teacher"]["source"] == "test-4b"
        assert row["task_reward"] == pytest.approx((row["raw_grade"] - calibration.judge.mean) / calibration.judge.std)
        assert row["reward"] == pytest.approx(row["task_reward"] - row["format_penalty"] - row["length_penalty"])
    for arm in ("knn4", "knn30"):
        rows = json.loads((result.run_dir / f"seed-7/{arm}/rollouts/update-000001.json").read_text())
        assert all(r["condition"] == arm and r["teacher"]["source"] == ("test-4b" if arm == "knn4" else "test-30b") for r in rows)
    before = len(inputs["30b"])
    assert factory().run().state == "completed" and len(inputs["30b"]) == before


@pytest.mark.parametrize("interrupted_arm", ["knn30", "judge4"])
def test_teacher_resume_retains_preparation_and_rejects_changed_teacher(setup, monkeypatch, interrupted_arm):
    config, factory, live, inputs = make(setup)
    original = PPOTrainer.update
    seen, failed = [], False
    def update(self, *args, **kwargs):
        nonlocal failed
        if self.reward_id.startswith(f"{interrupted_arm}/") and self.update_count == 1 and not failed:
            failed = True
            raise RuntimeError("interrupted")
        seen.append((self.reward_id, self.update_count))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPOTrainer, "update", update)
    with pytest.raises(RuntimeError, match="interrupted"):
        factory().run()
    assert all(ref() is None for ref in live)
    count = len(inputs["30b"])
    assert factory().run().state == "completed" and len(seen) == 8
    assert len(inputs["30b"]) == count
    assert all(ref() is None for ref in live)
    changed = replace(config, settings={**config.settings, "teacher_comparison": {
        **config.settings["teacher_comparison"], "teacher30": {"id": "other-teacher", "revision": "v2"}}})
    with pytest.raises(ValueError, match="protocol or data changed"):
        factory(changed).run()


@pytest.mark.parametrize("preset", ["smoke", "pilot", "full"])
def test_teacher_presets_use_fixed_retrieval(preset):
    config = load_gsm_config(Path(__file__).parents[1] / f"configs/gsm8k_teachers_{preset}.json")
    assert config.settings["k_values"] == [32] and config.settings["similarity_temperatures"] == [.05]
    assert config.settings["teacher_comparison"]["teacher30"]["id"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert config.settings["evaluate_test"] == (preset == "full")


def test_local_qwen3_moe_loads_through_actual_model_loader(loaded, tmp_path):
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig
    from reward_gap.models import ModelSpec, LoadOptions, load_policy_model
    config = Qwen3MoeConfig(vocab_size=len(loaded.tokenizer), hidden_size=16, intermediate_size=32,
                          moe_intermediate_size=8, num_experts=4, num_experts_per_tok=2, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=2048,
                          pad_token_id=loaded.tokenizer.pad_token_id, eos_token_id=loaded.tokenizer.eos_token_id)
    model = AutoModelForCausalLM.from_config(config)
    path = tmp_path / "tiny-moe"
    model.save_pretrained(path)
    loaded.tokenizer.save_pretrained(path)
    actual = load_policy_model(ModelSpec(path), LoadOptions(device="cpu", dtype="float32", allow_downloads=False))
    output = actual.model.generate(input_ids=torch.tensor([[5, 8, 9]]), max_new_tokens=2, do_sample=False)
    assert output.shape[1] > 3
    assert actual.model.config.model_type == "qwen3_moe"
    assert all(not p.requires_grad for p in actual.model.parameters())


@pytest.mark.parametrize("change", [
    {"k": 7}, {"temperature": 0}, {"teacher30": {"id": "x", "revision": ""}},
    {"teacher30": {"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "main"}},
])
def test_comparison_rejects_unmatched_settings(tmp_path, change):
    root = Path(__file__).parents[1]
    raw = json.loads((root / "configs/gsm8k_teachers_smoke.json").read_text())
    raw["base_config"] = str(root / "configs/gsm8k_models.json")
    raw["teacher_comparison"].update(change)
    (tmp_path / "pyproject.toml").touch()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_gsm_config(path)


def test_cli_selects_matched_teacher_runner(setup, monkeypatch, capsys):
    from reward_gap import cli
    from reward_gap.gsm8k import config as settings_module, teachers
    config, _, _, _ = make(setup)
    received = []
    class Runner:
        def __init__(self, settings, folder):
            assert settings is config
            received.append(folder)
        def run(self, *, until):
            received.append(until)
            return SimpleNamespace(state="paused", summary_path="summary.json")
    monkeypatch.setattr(settings_module, "load_gsm_config", lambda path: config)
    monkeypatch.setattr(teachers, "TeacherExperiment", Runner)
    cli.main(["gsm8k-run", "--config", "unused", "--run-name", "matched", "--until", "preparation"])
    assert received == [config.base.runtime.output_root / "matched", "preparation"]
    assert "paused" in capsys.readouterr().out


def test_grading_exception_records_failed_attempt_and_unknown_output_tokens(loaded, tmp_path, monkeypatch):
    from test_gsm8k_integration import make_grader
    grader, prompt = make_grader(loaded, tmp_path)
    def fail(**kwargs):
        raise RuntimeError("simulated generation failure")
    monkeypatch.setattr(loaded.model, "generate", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        grader.score([prompt], ["5"])
    event = json.loads((tmp_path / "grading_cost.jsonl").read_text())
    assert event["valid_grade"] is False and event["output_tokens_known"] is False
    assert event["attempt"] == 1 and event["model"] == loaded.source
    assert not list(tmp_path.glob("grade_cache/*.json"))
