import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from reward_gap.calibration import FrozenCalibration, ScoreScale
from reward_gap.config import COHORTS, GenerationConfig
from reward_gap.data import Message, PromptRecord
from reward_gap.evaluation import evaluate, EvaluationError
from reward_gap.memory import GapMemory, MemoryContext, MemoryError
from reward_gap.scorers import ScoreBatch


CAL = FrozenCalibration("cal1", ScoreScale("proxy", "p1", 10., 2.), ScoreScale("judge", "j1", 0., 1.))
CONTEXT = MemoryContext("proxy", "p1", "pool", "cal1")


class Actor:
    source = "policy"
    revision = "policy-v1"
    generation = GenerationConfig(max_new_tokens=4)

    def __init__(self):
        self.calls = []

    def generate(self, records, *, seed):
        self.calls.append((tuple(r.prompt_id for r in records), seed))
        return SimpleNamespace(prompt_ids=tuple(r.prompt_id for r in records),
                               answers=tuple("" if r.prompt_id.endswith("0") else "answer" for r in records),
                               response_lengths=tuple(1 if r.prompt_id.endswith("0") else 4 for r in records),
                               finish_reasons=tuple("eos" if r.prompt_id.endswith("0") else "length" for r in records),
                               prompt_token_counts=(3,) * len(records), sampled=True, seed=seed)


class Scorer:
    def __init__(self, role):
        self.role = role
        self.calls = []
        self.pooling = "pool"

    def score(self, records, answers, *, return_embeddings=False):
        self.calls.append((tuple(r.prompt_id for r in records), tuple(answers), return_embeddings))
        scores = tuple((14. if r.prompt_id.endswith("0") else 8.) if self.role == "proxy"
                       else (1. if r.prompt_id.endswith("0") else 2.) for r in records)
        embeddings = torch.tensor([[1., 0.] if r.prompt_id.endswith("0") else [0., 1.]
                                   for r in records]) if return_embeddings else None
        return ScoreBatch(tuple(r.prompt_id for r in records), scores, (6,) * len(records),
                          self.role, self.role, "p1" if self.role == "proxy" else "j1",
                          embeddings, self.pooling if return_embeddings else None)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    for cohort in COHORTS:
        rows = [PromptRecord(f"{cohort}-{i}", f"question {cohort} {i}", (Message("user", f"question {cohort} {i}"),))
                for i in range(3)]
        (root / f"{cohort}.json").write_text(json.dumps([asdict(r) for r in rows]), encoding="utf-8")
    return root


def memory():
    return GapMemory(["old-a", "old-b"], torch.eye(2), [0.5, -2.], context=CONTEXT, k=1)


def test_evaluation_saves_gaps_evidence_and_batch_seeds(prepared, tmp_path):
    actor, proxy, judge = Actor(), Scorer("proxy"), Scorer("judge")
    result = evaluate(actor, proxy, judge, CAL, prepared_dir=prepared, cohort="validation",
                      output_dir=tmp_path / "eval", policy_id="policy-checkpoint-1", seed=7, batch_size=2)
    rows = json.loads(result.rows_path.read_text())
    assert result.count == 3
    assert [r["gap"] for r in rows] == [1., -3., -3.]
    assert [r["generation_seed"] for r in rows] == [7, 7, 8]
    assert rows[0]["answer"] == "" and rows[0]["finish_reason"] == "eos"
    assert rows[1]["finish_reason"] == "length"
    assert "messages" in rows[0] and "predicted_gap" not in rows[0]
    assert all(not c[2] for c in proxy.calls + judge.calls)
    assert [c[:2] for c in proxy.calls] == [c[:2] for c in judge.calls]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["calibration"] == asdict(CAL)
    assert manifest["policy_id"] == "policy-checkpoint-1"
    assert manifest["metrics"]["mean_gap"] == pytest.approx(-5 / 3)
    assert json.loads((result.output_dir / "status.json").read_text())["state"] == "completed"
    assert not (result.output_dir / "embeddings.json").exists()


def test_memory_diagnostics_and_vectors_are_optional_and_do_not_append(prepared, tmp_path):
    mem = memory()
    result = evaluate(Actor(), Scorer("proxy"), Scorer("judge"), CAL,
                      prepared_dir=prepared, cohort="final_evaluation", output_dir=tmp_path / "eval",
                      policy_id="p1", seed=7, memory=mem, memory_id="M0", save_embeddings=True)
    rows = json.loads(result.rows_path.read_text())
    assert [r["predicted_gap"] for r in rows] == [0.5, -2., -2.]
    assert rows[0]["neighbor_ids"] == ["old-a"]
    assert rows[0]["gap_prediction_error"] == -0.5
    vectors = json.loads((result.output_dir / "embeddings.json").read_text())
    assert vectors["example_ids"] == [r["example_id"] for r in rows]
    assert vectors["vectors"] == [[1., 0.], [0., 1.], [0., 1.]]
    assert mem.example_ids == ("old-a", "old-b") and mem.size == 2


def test_output_is_not_overwritten_or_regenerated(prepared, tmp_path):
    actor = Actor()
    arguments = dict(prepared_dir=prepared, cohort="validation", output_dir=tmp_path / "eval", policy_id="p1", seed=7)
    result = evaluate(actor, Scorer("proxy"), Scorer("judge"), CAL, **arguments)
    before = result.rows_path.read_bytes()
    count = len(actor.calls)
    with pytest.raises(FileExistsError):
        evaluate(actor, Scorer("proxy"), Scorer("judge"), CAL, **arguments)
    assert result.rows_path.read_bytes() == before and len(actor.calls) == count


def test_failure_is_marked_and_never_completed(prepared, tmp_path):
    proxy = Scorer("proxy")
    proxy.pooling = "wrong"
    with pytest.raises(MemoryError):
        evaluate(Actor(), proxy, Scorer("judge"), CAL, prepared_dir=prepared, cohort="validation",
                 output_dir=tmp_path / "failed", policy_id="p1", seed=7, memory=memory(), memory_id="M0")
    assert json.loads((tmp_path / "failed/status.json").read_text())["state"] == "failed"
    assert not (tmp_path / "failed/manifest.json").exists()


@pytest.mark.parametrize("changes", [{"cohort": "training"}, {"seed": -1}, {"batch_size": 0},
                                     {"policy_id": ""}, {"output_dir": "relative/output"},
                                     {"memory_id": "M0"}, {"save_embeddings": "yes"}])
def test_invalid_options_fail_before_generation(prepared, tmp_path, changes):
    actor = Actor()
    options = dict(prepared_dir=prepared, cohort="validation", output_dir=tmp_path / "eval", policy_id="p1", seed=7)
    options.update(changes)
    with pytest.raises(EvaluationError):
        evaluate(actor, Scorer("proxy"), Scorer("judge"), CAL, **options)
    assert not actor.calls


def test_changed_scorer_order_is_rejected(prepared, tmp_path):
    from dataclasses import replace
    class ReversedScorer(Scorer):
        def score(self, records, answers, **kwargs):
            result = super().score(records, answers, **kwargs)
            return replace(result, prompt_ids=tuple(reversed(result.prompt_ids)))
    with pytest.raises(EvaluationError, match="order"):
        evaluate(Actor(), ReversedScorer("proxy"), Scorer("judge"), CAL,
                 prepared_dir=prepared, cohort="validation", output_dir=tmp_path / "eval", policy_id="p1", seed=7)
