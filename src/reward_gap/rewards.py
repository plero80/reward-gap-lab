"""Answer-level reward strategies. PPO's token KL penalty belongs to the trainer."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from reward_gap.calibration import FrozenCalibration
from reward_gap.data import PromptRecord
from reward_gap.memory import GapMemory, GapPrediction, MemoryContext
from reward_gap.scorers import ScoreBatch


class RewardError(ValueError):
    """Invalid reward inputs or unusable scorer results."""


class ProxyScorer(Protocol):
    """Implemented by RewardScorer; also permits small test doubles."""

    @property
    def role(self) -> Literal["proxy", "judge"]: ...

    def score(self, prompts: Sequence[PromptRecord], answers: Sequence[str], *,
              return_embeddings: bool = False) -> ScoreBatch: ...


@dataclass(frozen=True)
class RewardBatch:
    """One scalar per complete answer, aligned to prompt_ids (duplicates allowed)."""

    prompt_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    raw_proxy_scores: tuple[float, ...]
    normalized_proxy_scores: tuple[float, ...]
    predicted_gaps: tuple[float, ...]
    token_counts: tuple[int, ...]  # Scorer input lengths, NOT generated answer lengths
    strategy: Literal["proxy", "knn"]
    calibration_id: str
    source: str
    revision: str | None
    neighbors: GapPrediction | None = None


class ProxyReward:
    """Normalize frozen proxy scores with no correction or judge calls."""

    def __init__(self, proxy: ProxyScorer, calibration: FrozenCalibration):
        if proxy.role != "proxy":
            raise RewardError("Reward strategies require a proxy scorer")
        if not isinstance(calibration, FrozenCalibration):
            raise RewardError("Provide frozen calibration before calculating rewards")
        self.proxy = proxy
        self.calibration = calibration

    def _score(self, prompts: Sequence[PromptRecord], answers: Sequence[str], *,
               embeddings: bool) -> tuple[ScoreBatch, tuple[float, ...]]:
        if (not prompts or isinstance(answers, str) or len(prompts) != len(answers)
                or any(not isinstance(answer, str) for answer in answers)):
            raise RewardError("Provide a nonempty prompt batch and one answer string per prompt")
        scored = self.proxy.score(prompts, answers, return_embeddings=embeddings)
        if scored.prompt_ids != tuple(prompt.prompt_id for prompt in prompts):
            raise RewardError("Scorer changed prompt order or batch size")
        normalized = self.calibration.normalize_proxy(scored)
        return scored, normalized

    def score(self, prompts: Sequence[PromptRecord], answers: Sequence[str]) -> RewardBatch:
        scored, normalized = self._score(prompts, answers, embeddings=False)
        return RewardBatch(scored.prompt_ids, normalized, scored.scores, normalized,
                           (0.0,) * len(normalized), scored.token_counts, "proxy",
                           self.calibration.calibration_id, scored.source, scored.revision)


class KNNReward(ProxyReward):
    """Signed correction: normalized proxy score minus predicted proxy-judge gap.

    No correction clipping, batch normalization, memory mutation or judge calls.
    Construct another strategy with the new snapshot when memory is refreshed.
    """

    def __init__(self, proxy: ProxyScorer, calibration: FrozenCalibration, memory: GapMemory):
        super().__init__(proxy, calibration)
        if not isinstance(memory, GapMemory):
            raise RewardError("Provide a GapMemory snapshot")
        expected = calibration.proxy
        if (memory.context.encoder_id != expected.source
                or memory.context.encoder_revision != expected.revision
                or memory.context.calibration_id != calibration.calibration_id):
            raise RewardError("Memory does not match the supplied proxy calibration")
        self.memory = memory

    def score(self, prompts: Sequence[PromptRecord], answers: Sequence[str]) -> RewardBatch:
        scored, normalized = self._score(prompts, answers, embeddings=True)
        if scored.embeddings is None or not scored.embedding_pooling or not scored.revision:
            raise RewardError("kNN requires proxy embeddings, pooling metadata and an explicit model revision")
        if scored.embeddings.ndim != 2 or scored.embeddings.shape[0] != len(normalized):
            raise RewardError("Provide one embedding per scored answer")
        # Use actual scorer metadata, not memory.context echoed back to itself.
        context = MemoryContext(scored.source, scored.revision, scored.embedding_pooling,
                                self.calibration.calibration_id)
        prediction = self.memory.predict(scored.embeddings, context=context)
        gaps = tuple(prediction.gaps.tolist())
        rewards = tuple(proxy - gap for proxy, gap in zip(normalized, gaps, strict=True))
        if not all(math.isfinite(value) for value in rewards):
            raise RewardError("Correction produced nonfinite rewards")
        return RewardBatch(scored.prompt_ids, rewards, scored.scores, normalized, gaps,
                           scored.token_counts, "knn", self.calibration.calibration_id,
                           scored.source, scored.revision, prediction)
