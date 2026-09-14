"""Scientific boundary checks without dataset/model downloads."""

from dataclasses import asdict, replace
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration, ScoreScale
from reward_gap.gsm8k.answers import boxed, evaluate_answer, extract, numeric, parse_grade
from reward_gap.gsm8k.config import COHORTS, GSMConfig, load_gsm_config
from reward_gap.gsm8k.data import Question, PROMPT_VERSION, group, load_prepared, partition, schedule
from reward_gap.gsm8k.memory import QuestionMemory
from reward_gap.gsm8k.metrics import gap_metrics
from reward_gap.gsm8k.rewards import MathReward
from reward_gap.memory import MemoryContext
from reward_gap.scorers import ScoreBatch


@pytest.mark.parametrize("text,expected", [("1,200", 1200), (".5", Fraction(1, 2)),
    (r"\frac{3}{6}", Fraction(1, 2)), ("-2e2", -200), ("2/0", None), ("1,20", None),
    ("5 apples", None), ("NaN", None), ("1+2", None)])
def test_numeric_parser(text, expected):
    assert numeric(text) == expected


def test_numeric_correctness_format_and_completion_are_separate():
    unboxed = evaluate_answer("Work: 2 + 3 = 5\nThe answer is 5.", "5", "length")
    assert unboxed["numeric_match"] and not unboxed["strict_match"]
    assert not unboxed["format_compliant"] and unboxed["length_capped"]
    assert not unboxed["unresolved"]
    wrong = evaluate_answer(r"\boxed{6}", "5", "eos")
    assert wrong["format_compliant"] and wrong["numeric_mismatch"]
    assert not wrong["strict_match"] and not wrong["length_capped"]
    unresolved = evaluate_answer("I do not know.", "5", "eos")
    assert unresolved["unresolved"] and not unresolved["numeric_mismatch"]


@pytest.mark.parametrize("answer", [r"\boxed{5} or \boxed{6}", r"\boxed{5", r"\boxed{x}",
                                    "Answer: 5\nAnswer: 6", "Perhaps 5 or 6."])
def test_ambiguous_answers_never_use_gold_to_pick_a_value(answer):
    a, b = evaluate_answer(answer, "5", "eos"), evaluate_answer(answer, "6", "eos")
    assert a["extracted"] == b["extracted"] is None
    assert a["unresolved"] and b["unresolved"]


def test_balanced_fraction_box():
    assert boxed(r"Thus $\boxed{\frac{1}{2}}$.") == (Fraction(1, 2), "boxed")
    assert extract("#### 1,000") == (Fraction(1000), "explicit_final")


def test_gap_metrics_include_confusion_and_tie_aware_correlations():
    result = gap_metrics([0., 0., 2., 4.], [0., 0., 3., 1.], theta=1.5)
    assert result["mse"] == 2.5
    assert (result["true_positive"], result["false_positive"], result["true_negative"], result["false_negative"]) == (1, 0, 2, 1)
    assert result["spearman"] == pytest.approx(7 / 9)
    constant = gap_metrics([0., 0.], [0., 0.], theta=1.)
    assert constant["pearson"] is constant["spearman"] is constant["auroc"] is constant["average_precision"] is None
    assert constant["precision"] is constant["recall"] is None


@pytest.mark.parametrize("text,expected", [("Good work.\nSCORE: 5", 5), ("score: 1\n", 1),
    ("SCORE: 0", None), ("SCORE: 5\nMore text", None), ("SCORE: 3\nSCORE: 5", None),
    ("Score is 4", None), ("SCORE: 4.5", None)])
def test_grade_requires_one_terminal_score(text, expected):
    assert parse_grade(text) == expected


def test_partition_reserves_even_unused_test_questions_and_hides_current_gold():
    train = [{"question": f"Question {i}?", "answer": f"Secret reasoning {i}. #### {i + 700}"} for i in range(30)]
    test = [train[0], train[1], {"question": "TEST ONLY?", "answer": "#### 99"}]
    counts = dict.fromkeys(COHORTS, 3)
    first = partition(train, test, counts, 42, test_limit=1)
    second = partition(train, test, counts, 42, test_limit=1)
    assert first == second
    assert first != partition(train, test, counts, 43, test_limit=1)
    all_questions = [q for name in COHORTS for q in first[name]]
    assert len({group(q.question) for q in all_questions}) == 18
    assert not {group(q.question) for q in all_questions} & {group(r["question"]) for r in test}
    question = first["calibration"][0]
    messages = question.prompt().messages
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1].content == question.question
    assert all(question.gold not in m.content and "Secret reasoning" not in m.content for m in messages)


def test_schedule_repeats_each_sampled_question_without_repartitioning():
    questions = [Question(str(i), f"Q{i}", f"#### {i}", str(i)) for i in range(10)]
    settings = {"questions_per_update": 3, "responses_per_question": 2, "updates": 4}
    batches = schedule(questions, settings, 42)
    assert batches == schedule(questions, settings, 42)
    for batch in batches:
        ids = [p.prompt_id for p in batch]
        assert len(ids) == 6 and len(set(ids)) == 3
        assert all(ids.count(id_) == 2 for id_ in ids)


def memory_fixture(k=2):
    context = MemoryContext("proxy", "p1", "test-pooling", "cal")
    rows = [{"example_id": f"a/{i}", "question_id": "a", "gap": 100.} for i in range(2)]
    rows += [{"example_id": "b/0", "question_id": "b", "gap": 2.},
             {"example_id": "c/0", "question_id": "c", "gap": 4.}]
    vectors = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    return QuestionMemory(rows, vectors, context, k=k, temperature=.05)


def test_memory_excludes_every_response_to_query_question_and_roundtrips():
    memory = memory_fixture()
    for mem in (memory, QuestionMemory.from_dict(memory.to_dict())):
        values, neighbors = mem.predict(torch.tensor([[1., 0.]]), ["a"], context=mem.context)
        assert values == [3.]
        assert neighbors[0]["example_ids"] == ["b/0", "c/0"]
        assert neighbors[0]["question_ids"] == ["b", "c"]
    with pytest.raises(ValueError, match="Too few"):
        memory_fixture(k=3).predict(torch.tensor([[1., 0.]]), ["a"], context=memory.context)
    with pytest.raises(ValueError, match="context"):
        memory.predict(torch.tensor([[1., 0.]]), ["a"], context=replace(memory.context, calibration_id="other"))


@pytest.mark.parametrize("arm", ["proxy", "judge", "knn"])
def test_all_arms_apply_same_penalties_without_extra_judge_calls(arm):
    calls = []
    class Grader:
        def __init__(self, role):
            self.role = role
        def score(self, prompts, answers, *, return_embeddings=False):
            calls.append(self.role)
            return ScoreBatch(tuple(p.prompt_id for p in prompts), (3., 3.), (10, 10), self.role,
                              self.role, "p1" if self.role == "proxy" else "j1",
                              torch.tensor([[1., 0.]] * 2) if return_embeddings else None,
                              "test-pooling" if return_embeddings else None)
    calibration = FrozenCalibration("cal", ScoreScale("proxy", "p1", 2., 1.), ScoreScale("judge", "j1", 2., 1.))
    reward = MathReward(arm, Grader("proxy"), Grader("judge"), calibration, memory_fixture(),
                        {"format_penalty": .5, "length_penalty": .75})
    prompt = Question("a", "Question", "#### 5", "5").prompt()
    result = reward.score_rollouts([prompt] * 2, [r"\boxed{999}", "not formatted"],
                                   response_lengths=[5, 8], finish_reasons=["eos", "length"])
    assert calls == ["judge" if arm == "judge" else "proxy"]
    assert result.rewards[0] - result.rewards[1] == pytest.approx(1.25)
    assert reward.rows[0]["format_penalty"] == 0  # Wrong gold answer still has valid format.
    assert reward.rows[1]["length_penalty"] == .75
    assert reward.rows[0]["raw_grade"] == 3.
    with pytest.raises(ValueError, match="completion metadata"):
        reward.score([prompt], ["answer"])


@pytest.mark.parametrize("preset,updates,seeds", [("smoke", 2, (42,)), ("pilot", 100, (42,)), ("full", 400, (42, 43, 44))])
def test_presets(preset, updates, seeds):
    config = load_gsm_config(Path(__file__).parents[1] / f"configs/gsm8k_{preset}.json")
    assert config.base.training.total_updates == updates and config.base.seeds == seeds
    assert config.base.models.proxy.id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.settings["evaluate_test"] == (preset == "full")
    assert config.base.generation.max_new_tokens == 768


def write_prepared(config):
    """Fixture with exact source metadata and seven disjoint synthetic cohorts."""
    settings = config.settings
    counts = {**settings["cohorts"], "final": settings["test_limit"] or 3}
    folder = Path(settings["prepared_dir"])
    for name, count in counts.items():
        atomic_write_json(folder / f"{name}.json", [asdict(Question(f"{name}-{i}", f"{name} problem {i}?",
                          f"Private reference. #### {i}", str(i))) for i in range(count)])
    atomic_write_json(folder / "manifest.json", {"schema_version": 1, "dataset": "openai/gsm8k", "subset": "main",
        "revision": "a" * 40, "data_seed": settings["data_seed"], "requested_counts": settings["cohorts"],
        "test_limit": settings["test_limit"], "prompt_version": PROMPT_VERSION, "counts": counts})


def test_prepared_data_rejects_changed_revision_counts_and_cross_cohort_ids(tmp_path):
    original = load_gsm_config(Path(__file__).parents[1] / "configs/gsm8k_smoke.json")
    config = replace(original, settings={**original.settings, "prepared_dir": str(tmp_path),
                                         "cohorts": dict.fromkeys(COHORTS, 3), "test_limit": 3})
    write_prepared(config)
    assert len(load_prepared(config)[0]) == 7
    with pytest.raises(ValueError, match="revision differs"):
        load_prepared(replace(config, settings={**config.settings, "dataset_revision": "b" * 40}))
    path = tmp_path / "memory.json"
    rows = json.loads(path.read_text())
    rows[0]["id"] = "calibration-0"
    atomic_write_json(path, rows)
    with pytest.raises(ValueError, match="cohort"):
        load_prepared(config)
