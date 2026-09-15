"""Recover individual output failures without manufacturing labels or PPO rewards."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from test_gsm8k_integration import setup, loaded, make_grader
from test_gsm8k_teachers import make as teacher_setup
from test_ppo import trainer, prompts, Reward, assert_state_equal
from reward_gap.failures import SampleError
from reward_gap.calibration import FrozenCalibration
from reward_gap.gsm8k.data import Question
from reward_gap.gsm8k.experiment import read
from reward_gap.gsm8k.memory import QuestionMemory


def test_partial_grading_keeps_next_question_and_bounds_each_retry(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    other = Question("other", "Other question?", "#### 2", "2")
    grader.questions[other.id] = other
    calls = []
    def generate(**kwargs):
        ids = kwargs["input_ids"]
        calls.append(len(ids))
        texts = [loaded.tokenizer.decode(row, skip_special_tokens=False) for row in ids]
        rows = [loaded.tokenizer.encode("SCORE: 4" if "Other question?" in text else "no grade", add_special_tokens=False)
                + [loaded.tokenizer.eos_token_id] for text in texts]
        suffix = torch.full((len(rows), max(map(len, rows))), loaded.tokenizer.pad_token_id, dtype=torch.long)
        for i, row in enumerate(rows):
            suffix[i, :len(row)] = torch.tensor(row)
        return torch.cat((ids, suffix), dim=1)
    monkeypatch.setattr(loaded.model, "generate", generate)
    batches, errors = grader.score_partial([prompt, other.prompt()], ["5", "2"], return_embeddings=True)
    assert batches[0] is None and errors[0]
    assert batches[1].scores == (4.,) and errors[1] is None
    assert calls == [2, 1]
    assert len(list((tmp_path / "grade_cache").glob("*.json"))) == 1


def test_grade_timeout_retries_but_cuda_error_is_fatal(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    attempts = []
    def generate(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("temporary timeout")
        tokens = torch.tensor([loaded.tokenizer.encode("SCORE: 4", add_special_tokens=False)])
        return torch.cat((kwargs["input_ids"], tokens), dim=1)
    monkeypatch.setattr(loaded.model, "generate", generate)
    assert grader.score([prompt], ["answer"]).scores == (4.,)
    assert len(attempts) == 2
    def oom(**kwargs):
        raise torch.OutOfMemoryError("GPU exhausted")
    monkeypatch.setattr(loaded.model, "generate", oom)
    with pytest.raises(torch.OutOfMemoryError, match="GPU exhausted"):
        grader.score_partial([prompt], ["uncached answer"])


def deterministic_actor(experiment, *, missing=None, empty=False):
    def generate(records, **kwargs):
        if missing and any(p.prompt_id == missing for p in records):
            raise SampleError("No answer tokens")
        answers = tuple("" if empty else rf"\boxed{{{experiment.questions[p.prompt_id].gold}}}" for p in records)
        return SimpleNamespace(prompt_ids=tuple(p.prompt_id for p in records), answers=answers,
                               response_lengths=(1 if empty else 5,) * len(records), finish_reasons=("eos",) * len(records))
    return SimpleNamespace(generation=experiment.config.generation, generate=generate)


def initialize(factory):
    experiment = factory()
    experiment.run_dir.mkdir(parents=True)
    experiment._open_gsm()
    experiment.proxy, experiment.judge = experiment._scorer("proxy"), experiment._scorer("judge")
    return experiment


def test_failed_generation_isolated_and_kept_in_evaluation_denominator(setup):
    _, factory, _ = setup
    exp = initialize(factory)
    actor = deterministic_actor(exp)
    prepared = exp._prepare_seed(exp.run_dir / "cal", actor, 7)
    cal = FrozenCalibration.load(prepared["calibration"])
    built = exp._build_memory(exp.run_dir / "mem", actor, 7, cal)
    memory = QuestionMemory.from_dict(read(built["memory"]))
    result = exp._evaluate_math(exp.run_dir / "eval", deterministic_actor(exp, missing="final-1"),
                               7, "base", 0, "final", cal, memory, prepared["theta"], 0.)
    assert result["metrics"]["count"] == 3
    assert result["metrics"]["numeric_match"] == pytest.approx(2 / 3)
    assert result["metrics"]["graded_count"] == 2 and result["metrics"]["failed_count"] == 1
    failed = next(r for r in read(result["rows"]) if r["question_id"] == "final-1")
    assert failed["unresolved"] and failed["raw_judge"] is None and failed["answer"] == ""
    assert "final-1" in (exp.run_dir / "sample_failures.jsonl").read_text()


def test_empty_eos_answer_is_a_real_observation_not_an_exception(setup):
    _, factory, _ = setup
    exp = initialize(factory)
    data = exp._labels(deterministic_actor(exp, empty=True), "monitor", 7, 1)
    assert len(data["rows"]) == 3 and not data["failed_rows"]
    assert all(r["empty_answer"] and r["unresolved"] and not r["numeric_match"] for r in data["rows"])


def test_calibration_drops_only_ungraded_pair_and_preserves_embedding_alignment(setup, monkeypatch):
    _, factory, _ = setup
    exp = initialize(factory)
    original = exp.judge.score
    def score(records, answers, **kwargs):
        if any(p.prompt_id.endswith("-1") for p in records):
            raise SampleError("Unparseable grade for middle example")
        return original(records, answers, **kwargs)
    monkeypatch.setattr(exp.judge, "score", score)
    data = exp._labels(deterministic_actor(exp), "calibration", 7, 1)
    assert [r["question_id"] for r in data["rows"]] == ["calibration-0", "calibration-2"]
    assert data["embeddings"] == [[1., 1.], [1., 3.]]
    assert data["failed_rows"][0]["question_id"] == "calibration-1"
    FrozenCalibration.fit(exp._batch(data, "proxy"), exp._batch(data, "judge"), calibration_id="test")


def test_all_missing_final_grades_still_finishes_and_reports_numeric_outcomes(setup):
    _, factory, _ = setup
    exp = factory()
    original = exp.scorer_factory
    def scorers(role):
        scorer = original(role)
        score = scorer.score
        def sometimes(records, answers, **kwargs):
            if role == "judge" and "/evaluation/final/" in scorer.phase:
                raise SampleError("No parseable grade")
            return score(records, answers, **kwargs)
        scorer.score = sometimes
        return scorer
    exp.scorer_factory = scorers
    result = exp.run()
    assert result.state == "completed"
    summary = read(result.summary_path)
    for evaluation in summary["results"]:
        if evaluation["cohort"] == "final":
            metrics = evaluation["metrics"]
            assert metrics["count"] == 3 and metrics["failed_count"] == 3
            assert metrics["judge"] is None and metrics["gap_prediction"]["count"] == 0
            assert isinstance(metrics["numeric_match"], float)
    assert "NaN" not in result.summary_path.read_text()


def test_skip_before_optimization_preserves_weights_optimizer_and_resume(loaded, tmp_path):
    class RecoverableReward(Reward):
        recover_sample_failures = True
        fail = False
        def score(self, records, answers):
            if self.fail:
                raise SampleError("Unparseable grade")
            return super().score(records, answers)
    run = trainer(loaded)
    run.config = replace(run.config, total_updates=3)
    run.reward = RecoverableReward()
    batches, seeds = [prompts()] * 3, [7, 8, 9]
    run.train(batches, seeds, until_update=1)
    before = {n: p.detach().clone() for n, p in run.actor.named_parameters()}
    optimizer = deepcopy(run.optimizer.state_dict())
    scheduler = deepcopy(run._backend.lr_scheduler.state_dict())
    run.reward.fail = True
    skipped = run.train(batches, seeds, until_update=2)[0]
    assert skipped.skipped and skipped.mean_reward is None and skipped.update == 2
    assert not run._failed and not run._bridge.batches
    assert_state_equal(optimizer, run.optimizer.state_dict())
    assert_state_equal(scheduler, run._backend.lr_scheduler.state_dict())
    for name, parameter in run.actor.named_parameters():
        torch.testing.assert_close(before[name], parameter, atol=0, rtol=0)
    path = run.save_checkpoint(tmp_path / "resume.pt")
    run.reward.fail = False
    run.train(batches, seeds)
    expected = {n: p.detach().clone() for n, p in run.actor.named_parameters()}
    restored = trainer(loaded)
    restored.config = replace(restored.config, total_updates=3)
    restored.reward = RecoverableReward()
    restored.load_checkpoint(path)
    assert not restored.train(batches, seeds)[0].skipped
    for name, parameter in restored.actor.named_parameters():
        torch.testing.assert_close(expected[name], parameter, atol=0, rtol=0)
    run.release()
    restored.release()


def test_teacher_failure_filters_both_memories_to_identical_examples(setup):
    _, factory, _, _ = teacher_setup(setup)
    exp = factory()
    original = exp.teacher_factory
    def teachers(name):
        teacher = original(name)
        # Avoid keeping teacher alive via its own bound method/closure.
        implementation = type(teacher).score
        import weakref
        teacher_ref = weakref.proxy(teacher)
        def wrapper(records, answers, **kwargs):
            if name == "30b" and any(p.prompt_id.endswith("-1") for p in records):
                raise SampleError("30B missing grade")
            return implementation(teacher_ref, records, answers, **kwargs)
        teacher.score = wrapper
        return teacher
    exp.teacher_factory = teachers
    assert exp.run(until="preparation").state == "paused"
    c4, m4, _ = exp.preparations[7]["4b"]
    c30, m30, _ = exp.preparations[7]["30b"]
    assert c4.proxy == c30.proxy
    assert torch.equal(m4.vectors, m30.vectors)
    assert [r["example_id"] for r in m4.rows] == [r["example_id"] for r in m30.rows]
    assert all(not r["question_id"].endswith("-1") for r in m4.rows)
    assert len(m4.rows) == 2


def test_coordinator_continues_after_one_ungradable_ppo_batch(setup):
    _, factory, _ = setup
    exp = factory()
    original = exp.scorer_factory
    failed = False
    def scorers(role):
        scorer = original(role)
        score = scorer.score
        def sometimes(records, answers, **kwargs):
            nonlocal failed
            if role == "proxy" and scorer.phase.endswith("training/proxy") and not failed:
                failed = True
                raise SampleError("One missing training grade")
            return score(records, answers, **kwargs)
        scorer.score = sometimes
        return scorer
    exp.scorer_factory = scorers
    result = exp.run()
    assert result.state == "completed" and failed
    train_result = read(result.summary_path)["training"]["seed-7/train/proxy"]
    assert train_result["optimized_batches"] == 1 and train_result["skipped_batches"] == 1
    history = read(train_result["metrics"])
    assert history[0]["skipped"] and history[0]["mean_reward"] is None
    assert not history[1]["skipped"] and history[1]["mean_reward"] is not None
    audit = read(result.run_dir / "seed-7/proxy/rollouts/update-000001.json")
    assert len(audit) == 2 and all(r["reward"] is None for r in audit)
    assert "batch_skipped" in (result.run_dir / "sample_failures.jsonl").read_text()


def test_sample_error_after_native_optimizer_still_requires_checkpoint(loaded, monkeypatch):
    from reward_gap.ppo import PPOError
    run = trainer(loaded)
    run.reward.recover_sample_failures = True
    run.update(prompts(), rollout_seed=7)
    train = run._backend.train
    def bad_callback():
        train()
        raise SampleError("Unexpected failure after optimization")
    monkeypatch.setattr(run._backend, "train", bad_callback)
    with pytest.raises(SampleError, match="after optimization"):
        run.update(prompts(), rollout_seed=8)
    with pytest.raises(PPOError, match="Previous update failed"):
        run.update(prompts(), rollout_seed=8)
    run.release()
