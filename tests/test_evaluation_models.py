from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from test_policy import loaded as policy_loaded, actor_for
from test_scorers import loaded as scorer_loaded
from test_evaluation import prepared
from reward_gap.calibration import FrozenCalibration, ScoreScale
from reward_gap.data import load_prompts
from reward_gap.evaluation import evaluate
from reward_gap.memory import GapMemory, MemoryContext
from reward_gap.refresh import refresh_memory
from reward_gap.scorers import RewardScorer


def test_real_generation_scoring_and_refresh_leave_weights_unchanged(policy_loaded, scorer_loaded, prepared, tmp_path):
    actor = actor_for(policy_loaded)
    scorer_loaded = replace(scorer_loaded, revision="test-revision")
    proxy = RewardScorer(scorer_loaded, role="proxy", batch_size=2)
    judge = RewardScorer(scorer_loaded, role="judge", batch_size=2)
    scale = ScoreScale(scorer_loaded.source, scorer_loaded.revision, 0., 1.)
    calibration = FrozenCalibration("tiny-cal", scale, scale)
    before = {n: p.detach().clone() for n, p in actor.named_parameters()}
    initial = load_prompts(prepared / "initial_memory.json")[:1]
    scored = proxy.score(initial, ["initial answer"], return_embeddings=True)
    context = MemoryContext(scored.source, scored.revision, scored.embedding_pooling, calibration.calibration_id)
    memory = GapMemory(["initial-example"], scored.embeddings, [1.], context=context, k=1)
    evaluation = evaluate(actor, proxy, judge, calibration, prepared_dir=prepared, cohort="validation",
                          output_dir=tmp_path / "evaluation", policy_id="tiny-policy", seed=7,
                          batch_size=2, memory=memory, memory_id="M0")
    refresh = refresh_memory(actor, proxy, judge, calibration, memory, prepared_dir=prepared,
                             output_dir=tmp_path / "refresh", policy_id="tiny-policy",
                             parent_memory_id="M0", memory_id="M1", seed=7, batch_size=2)
    assert evaluation.count == refresh.added_count == 3
    assert memory.size == 1 and refresh.memory.size == 4
    assert all(torch.equal(p, before[n]) for n, p in actor.named_parameters())
    assert all(p.grad is None for p in actor.parameters())
    assert all(not p.requires_grad for p in scorer_loaded.model.parameters())
