"""Signed cosine-kNN gap memory; no model loading or calibration fitting."""

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from reward_gap.artifacts import atomic_write_json


class MemoryError(ValueError):
    """Invalid memory data, settings, or incompatible representations."""


@dataclass(frozen=True)
class MemoryContext:
    """Caller supplies the actual encoder revision and saved calibration identity."""

    encoder_id: str
    encoder_revision: str
    pooling: str
    calibration_id: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise MemoryError(f"{name} must be a nonempty string without surrounding whitespace")


@dataclass(frozen=True)
class GapPrediction:
    gaps: torch.Tensor                  # [queries], CPU float64
    neighbor_ids: tuple[tuple[str, ...], ...]
    similarities: torch.Tensor          # [queries, k], cosine similarity
    weights: torch.Tensor               # [queries, k], sum to one per query


def _vectors(value: torch.Tensor, *, dimension: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) <= 0:
        raise MemoryError("Embeddings must be a nonempty [rows, dimension] tensor")
    if not value.is_floating_point():
        raise MemoryError("Embeddings must be floating point")
    if dimension is not None and value.shape[1] != dimension:
        raise MemoryError("Embedding dimension does not match memory")
    rows = value.detach().to(device="cpu", dtype=torch.float64).clone()
    norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    if not torch.isfinite(rows).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
        raise MemoryError("Embeddings must be finite and have nonzero norms")
    return rows / norms


def _scores(values: Sequence[float], count: int, name: str) -> torch.Tensor:
    if len(values) != count or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise MemoryError(f"{name} must contain one numeric score per example")
    result = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise MemoryError(f"{name} must contain finite scores")
    return result


class GapMemory:
    """Fixed memory snapshot. append() returns a new snapshot, leaving this intact.

    IDs identify prompt-answer examples, not just prompts: multiple answers to
    one prompt need distinct IDs. Exact similarity ties use ascending example ID.
    Scores passed to build/append must already use the same frozen calibration.
    """

    def __init__(self, example_ids: Sequence[str], embeddings: torch.Tensor,
                 gaps: Sequence[float], *, context: MemoryContext, k: int = 8,
                 temperature: float = 0.1):
        if not isinstance(context, MemoryContext):
            raise MemoryError("Provide a MemoryContext")
        if type(k) is not int or k < 1:
            raise MemoryError("k must be a positive integer")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature <= 0):
            raise MemoryError("temperature must be finite and positive")
        vectors = _vectors(embeddings)
        ids = tuple(example_ids)
        if (isinstance(example_ids, str) or len(ids) != len(vectors)
                or any(not isinstance(i, str) or not i.strip() or i != i.strip() for i in ids)):
            raise MemoryError("Provide one nonempty example ID per embedding")
        if len(set(ids)) != len(ids):
            raise MemoryError("Example IDs must be unique")
        if len(ids) < k:
            raise MemoryError("Memory must contain at least k examples")
        labels = _scores(gaps, len(ids), "gaps")
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        self._ids = tuple(ids[i] for i in order)
        self._vectors = vectors[order]
        self._gaps = labels[order]
        self._context = context
        self._k = k
        self._temperature = float(temperature)

    @property
    def context(self) -> MemoryContext:
        return self._context

    @property
    def example_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def size(self) -> int:
        return len(self._ids)

    @classmethod
    def build(cls, example_ids: Sequence[str], embeddings: torch.Tensor, *,
              proxy_scores: Sequence[float], judge_scores: Sequence[float],
              context: MemoryContext, k: int = 8, temperature: float = 0.1) -> "GapMemory":
        """Build from NORMALIZED scores: gap = proxy - judge, retaining its sign."""
        n = len(example_ids)
        gaps = _scores(proxy_scores, n, "proxy_scores") - _scores(judge_scores, n, "judge_scores")
        return cls(example_ids, embeddings, gaps.tolist(), context=context, k=k, temperature=temperature)

    def _compatible(self, context: MemoryContext) -> None:
        if context != self.context:
            raise MemoryError("Encoder, revision, pooling or calibration does not match memory")

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor, *, context: MemoryContext) -> GapPrediction:
        """Cosine top-k, then softmax(similarity / temperature) weighted gaps.

        Process one query at a time to avoid a queries-by-memory allocation.
        No judge calls, gradients, thresholding, or positive-only clipping.
        """
        self._compatible(context)
        queries = _vectors(embeddings, dimension=self._vectors.shape[1])
        predictions, neighbors, similarities, weights = [], [], [], []
        for query in queries:
            similarity = (self._vectors @ query).clamp(-1.0, 1.0)
            indices = torch.argsort(similarity, descending=True, stable=True)[:self._k]
            selected = similarity[indices]
            # Subtract before dividing to avoid overflow for small temperatures.
            weight = torch.softmax((selected - selected.max()) / self._temperature, dim=0)
            predictions.append((weight * self._gaps[indices]).sum())
            neighbors.append(tuple(self._ids[i] for i in indices.tolist()))
            similarities.append(selected)
            weights.append(weight)
        return GapPrediction(torch.stack(predictions), tuple(neighbors),
                             torch.stack(similarities), torch.stack(weights))

    def append(self, example_ids: Sequence[str], embeddings: torch.Tensor, *,
               proxy_scores: Sequence[float], judge_scores: Sequence[float],
               context: MemoryContext) -> "GapMemory":
        """Return a refreshed memory; duplicate IDs and incompatible inputs fail."""
        self._compatible(context)
        addition = self.build(example_ids, embeddings, proxy_scores=proxy_scores,
                              judge_scores=judge_scores, context=context, k=1,
                              temperature=self._temperature)
        if addition._vectors.shape[1] != self._vectors.shape[1]:
            raise MemoryError("Embedding dimension does not match memory")
        return GapMemory(self._ids + addition._ids, torch.cat((self._vectors, addition._vectors)),
                         torch.cat((self._gaps, addition._gaps)).tolist(), context=context,
                         k=self._k, temperature=self._temperature)

    def save(self, path: str | Path) -> Path:
        """Atomically publish one JSON snapshot, refusing to replace any file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "context": asdict(self.context), "k": self._k,
                   "temperature": self._temperature, "example_ids": self._ids,
                   "embeddings": self._vectors.tolist(), "gaps": self._gaps.tolist()}
        handle, name = tempfile.mkstemp(dir=destination.parent, prefix=".memory-", suffix=".json")
        os.close(handle)
        temporary = Path(name)
        try:
            atomic_write_json(temporary, payload)
            # A hard link publishes the completed file without a check/replace race.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path, *, context: MemoryContext) -> "GapMemory":
        """Read JSON only; validate contents and expected representation context."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            fields = {"schema_version", "context", "k", "temperature", "example_ids", "embeddings", "gaps"}
            if not isinstance(payload, dict) or set(payload) != fields:
                raise MemoryError("Invalid memory snapshot fields")
            if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
                raise MemoryError("Unsupported memory schema version")
            result = cls(payload["example_ids"], torch.tensor(payload["embeddings"], dtype=torch.float64),
                         payload["gaps"], context=MemoryContext(**payload["context"]),
                         k=payload["k"], temperature=payload["temperature"])
            result._compatible(context)
            return result
        except (TypeError, ValueError, RuntimeError, KeyError) as exc:
            raise MemoryError(f"Invalid memory snapshot: {exc}") from exc
