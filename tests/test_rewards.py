from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from reward_gap.calibration import CalibrationError, FrozenCalibration, ScoreScale
from reward_gap.data import Message, PromptRecord
from reward_gap.memory import GapMemory, MemoryContext, MemoryError
from reward_gap.rewards import KNNReward, ProxyReward, RewardError
from reward_gap.scorers import ScoreBatch


PROMPTS = [PromptRecord("p1", "g1", (Message("user", "hello"),)),
           PromptRecord("p2", "g2", (Message("user", "goodbye"),))]
CALIBRATION = FrozenCalibration("cal-1", ScoreScale("proxy", "v1", 10., 2.),
                                ScoreScale("judge", "j1", 1., 3.))
CONTEXT = MemoryContext("proxy", "v1", "pool-v1", "cal-1")


class FakeProxy:
    role = "proxy"

    def __init__(self):
        self.calls = []
        self.batch = ScoreBatch(("p1", "p2"), (14., 8.), (10, 20), "proxy", "proxy", "v1",
                                torch.eye(2), "pool-v1")

    def score(self, prompts, answers, *, return_embeddings=False):
        self.calls.append((tuple(answers), return_embeddings))
        return self.batch if return_embeddings else replace(self.batch, embeddings=None, embedding_pooling=None)


def memory():
    return GapMemory(["a", "b"], torch.eye(2), [0.5, -2.], context=CONTEXT, k=1)


def test_proxy_uses_frozen_scale_without_embeddings_or_correction():
    proxy = FakeProxy()
    result = ProxyReward(proxy, CALIBRATION).score(PROMPTS, ["answer", ""])
    assert result.rewards == (2., -1.)
    assert result.raw_proxy_scores == (14., 8.)
    assert result.predicted_gaps == (0., 0.)
    assert result.neighbors is None
    assert result.prompt_ids == ("p1", "p2")
    assert result.token_counts == (10, 20)
    assert proxy.calls == [(("answer", ""), False)]
    assert result.strategy == "proxy"


def test_knn_signed_correction_and_diagnostics_without_mutation():
    proxy, mem = FakeProxy(), memory()
    result = KNNReward(proxy, CALIBRATION, mem).score(PROMPTS, ["a", "b"])
    assert result.normalized_proxy_scores == (2., -1.)
    assert result.predicted_gaps == (0.5, -2.)
    assert result.rewards == (1.5, 1.)  # Negative gap INCREASES the second reward.
    assert result.neighbors.neighbor_ids == (("a",), ("b",))
    assert result.neighbors.weights.tolist() == [[1.], [1.]]
    assert not result.neighbors.gaps.requires_grad
    assert proxy.calls == [(("a", "b"), True)]
    assert mem.size == 2
    assert mem.example_ids == ("a", "b")


def test_query_context_comes_from_scorer_not_memory():
    proxy = FakeProxy()
    proxy.batch = replace(proxy.batch, embedding_pooling="different-pooling")
    with pytest.raises(MemoryError, match="does not match"):
        KNNReward(proxy, CALIBRATION, memory()).score(PROMPTS, ["a", "b"])


@pytest.mark.parametrize("changes", [{"source": "other"}, {"revision": "v2"}, {"role": "judge"},
                                      {"scores": (float("nan"), 1.)}, {"scores": (1.,)}])
def test_bad_scores_rejected(changes):
    proxy = FakeProxy()
    proxy.batch = replace(proxy.batch, **changes)
    with pytest.raises(CalibrationError):
        ProxyReward(proxy, CALIBRATION).score(PROMPTS, ["a", "b"])


@pytest.mark.parametrize("changes", [{"embeddings": None}, {"embeddings": torch.ones(1, 2)},
                                      {"embedding_pooling": None}])
def test_knn_missing_embedding_metadata(changes):
    proxy = FakeProxy()
    proxy.batch = replace(proxy.batch, **changes)
    with pytest.raises(RewardError):
        KNNReward(proxy, CALIBRATION, memory()).score(PROMPTS, ["a", "b"])


def test_wrong_memory_rejected_before_scoring():
    proxy = FakeProxy()
    with pytest.raises(RewardError, match="does not match"):
        KNNReward(proxy, replace(CALIBRATION, calibration_id="cal-2"), memory())
    assert not proxy.calls


@pytest.mark.parametrize("answers", [[], ["a"], "ab", ["a", None]])
def test_invalid_input_fails_before_scoring(answers):
    proxy = FakeProxy()
    with pytest.raises(RewardError):
        ProxyReward(proxy, CALIBRATION).score(PROMPTS, answers)
    assert not proxy.calls


def test_alignment_and_duplicate_prompt_ids():
    proxy = FakeProxy()
    proxy.batch = replace(proxy.batch, prompt_ids=("p2", "p1"))
    with pytest.raises(RewardError, match="order"):
        ProxyReward(proxy, CALIBRATION).score(PROMPTS, ["a", "b"])
    proxy.batch = replace(proxy.batch, prompt_ids=("p1", "p1"))
    result = ProxyReward(proxy, CALIBRATION).score([PROMPTS[0], PROMPTS[0]], ["a", "b"])
    assert result.prompt_ids == ("p1", "p1")


def test_judge_cannot_be_reward_scorer():
    proxy = FakeProxy()
    proxy.role = "judge"
    with pytest.raises(RewardError, match="proxy scorer"):
        ProxyReward(proxy, CALIBRATION)


def test_reward_independent_of_batch_composition():
    proxy = FakeProxy()
    strategy = KNNReward(proxy, CALIBRATION, memory())
    batch = strategy.score(PROMPTS, ["a", "b"])
    proxy.batch = replace(proxy.batch, prompt_ids=("p1",), scores=(14.,), token_counts=(10,),
                          embeddings=torch.tensor([[1., 0.]]))
    single = strategy.score(PROMPTS[:1], ["a"])
    assert single.rewards == batch.rewards[:1]
