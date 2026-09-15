"""Generate and score held-out answers; save inspectable, immutable run artifacts."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import torch

from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration
from reward_gap.config import GenerationConfig
from reward_gap.data import PromptRecord, load_prompts
from reward_gap.memory import GapMemory, MemoryContext
from reward_gap.scorers import ScoreBatch

if TYPE_CHECKING:
    from reward_gap.policy import RolloutBatch


class EvaluationError(ValueError):
    """Invalid evaluation inputs or misaligned generated/scored examples."""


class Actor(Protocol):
    source: str
    revision: str | None
    generation: GenerationConfig

    def generate(self, prompts: Sequence[PromptRecord], *, seed: int) -> "RolloutBatch": ...


class Scorer(Protocol):
    @property
    def role(self) -> Literal["proxy", "judge"]: ...

    def score(self, prompts: Sequence[PromptRecord], answers: Sequence[str], *,
              return_embeddings: bool = False) -> ScoreBatch: ...


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    rows_path: Path
    manifest_path: Path
    count: int


@dataclass(frozen=True)
class _ScoredAnswers:
    rows: list[dict]
    embeddings: torch.Tensor | None
    context: MemoryContext | None


def _resolved_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise EvaluationError("Use absolute paths from load_config; notebook working directories must not change their meaning")
    return value.resolve()


def _validate(prompts: Sequence[PromptRecord], proxy: Scorer, judge: Scorer, *,
              policy_id: str, seed: int, batch_size: int) -> None:
    if not prompts:
        raise EvaluationError("The selected cohort contains no prompts")
    if len({p.prompt_id for p in prompts}) != len(prompts):
        raise EvaluationError("Cohort prompt IDs must be unique")
    if proxy.role != "proxy" or judge.role != "judge":
        raise EvaluationError("Provide separate proxy and judge scoring roles")
    if not isinstance(policy_id, str) or not policy_id.strip() or policy_id != policy_id.strip():
        raise EvaluationError("Provide a nonempty policy checkpoint identity")
    if type(batch_size) is not int or batch_size <= 0:
        raise EvaluationError("batch_size must be a positive integer")
    batches = (len(prompts) + batch_size - 1) // batch_size
    if type(seed) is not int or not 0 <= seed < 2**63 or seed + batches - 1 >= 2**63:
        raise EvaluationError("All generation batch seeds must be in [0, 2**63)")


def _collect(actor: Actor, proxy: Scorer, judge: Scorer, calibration: FrozenCalibration,
             prompts: Sequence[PromptRecord], *, seed: int, batch_size: int,
             need_embeddings: bool, memory: GapMemory | None = None) -> _ScoredAnswers:
    """Shared generation/labeling path. Neither fits calibration nor updates weights."""
    rows: list[dict] = []
    vectors: list[torch.Tensor] = []
    context = None
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        batch_seed = seed + start // batch_size
        rollout = actor.generate(chunk, seed=batch_seed)
        ids = tuple(record.prompt_id for record in chunk)
        if rollout.prompt_ids != ids or any(len(v) != len(chunk) for v in (
                rollout.answers, rollout.response_lengths, rollout.finish_reasons, rollout.prompt_token_counts)):
            raise EvaluationError("Generated answers do not match the requested prompts")
        if rollout.seed != batch_seed or rollout.sampled != actor.generation.do_sample:
            raise EvaluationError("Generation metadata differs from the requested settings")
        if (any(not isinstance(a, str) for a in rollout.answers)
                or any(type(n) is not int or n <= 0 for n in rollout.response_lengths)
                or any(reason not in ("eos", "length") for reason in rollout.finish_reasons)):
            raise EvaluationError("Invalid answer text, lengths or stopping reasons")
        proxy_batch = proxy.score(chunk, rollout.answers, return_embeddings=need_embeddings)
        judge_batch = judge.score(chunk, rollout.answers)
        if proxy_batch.prompt_ids != ids or judge_batch.prompt_ids != ids:
            raise EvaluationError("Scorer changed prompt order or batch size")
        zp = calibration.normalize_proxy(proxy_batch)
        zj = calibration.normalize_judge(judge_batch)
        prediction = None
        if need_embeddings:
            embedding = proxy_batch.embeddings
            if (embedding is None or embedding.ndim != 2 or embedding.shape[0] != len(chunk)
                    or embedding.shape[1] == 0 or not embedding.is_floating_point()
                    or not proxy_batch.revision or not proxy_batch.embedding_pooling):
                raise EvaluationError("Proxy must return embeddings, a revision and pooling metadata")
            current = MemoryContext(proxy_batch.source, proxy_batch.revision,
                                    proxy_batch.embedding_pooling, calibration.calibration_id)
            if context is not None and (current != context or embedding.shape[1] != vectors[0].shape[1]):
                raise EvaluationError("Proxy embedding representation changed between batches")
            context = current
            vector = embedding.detach().to(device="cpu", dtype=torch.float64).clone()
            norms = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
            if not torch.isfinite(vector).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
                raise EvaluationError("Proxy returned nonfinite or zero embeddings")
            vector = vector / norms
            vectors.append(vector)
            if memory is not None:
                prediction = memory.predict(vector, context=current)
        for i, record in enumerate(chunk):
            row = {"example_id": f"example-{start + i:08d}", "prompt_id": record.prompt_id,
                   "conversation_group": record.conversation_group,
                   "messages": [asdict(m) for m in record.messages], "answer": rollout.answers[i],
                   "raw_proxy_score": proxy_batch.scores[i], "raw_judge_score": judge_batch.scores[i],
                   "normalized_proxy_score": zp[i], "normalized_judge_score": zj[i], "gap": zp[i] - zj[i],
                   "prompt_tokens": rollout.prompt_token_counts[i], "response_tokens": rollout.response_lengths[i],
                   "finish_reason": rollout.finish_reasons[i], "sampled": rollout.sampled,
                   "generation_seed": batch_seed, "proxy_tokens": proxy_batch.token_counts[i],
                   "judge_tokens": judge_batch.token_counts[i]}
            if prediction is not None:
                gap = float(prediction.gaps[i])
                row.update(predicted_gap=gap, corrected_reward=zp[i] - gap,
                           gap_prediction_error=gap - row["gap"],
                           neighbor_ids=list(prediction.neighbor_ids[i]),
                           neighbor_similarities=prediction.similarities[i].tolist(),
                           neighbor_weights=prediction.weights[i].tolist())
            rows.append(row)
    return _ScoredAnswers(rows, torch.cat(vectors) if vectors else None, context)


def _manifest(actor: Actor, calibration: FrozenCalibration, *, policy_id: str, prepared_dir: Path,
              cohort: str, seed: int, batch_size: int, rows: list[dict]) -> dict:
    count = len(rows)
    return {"schema_version": 1, "policy_id": policy_id, "policy_source": actor.source,
            "policy_revision": actor.revision, "generation": asdict(actor.generation),
            "effective_eos_ids": list(getattr(actor, "eos_ids", ())),
            "calibration": asdict(calibration), "prepared_dir": str(prepared_dir), "cohort": cohort,
            "seed": seed, "batch_size": batch_size, "count": count,
            "metrics": {name: sum(row[field] for row in rows) / count for name, field in (
                ("mean_proxy_score", "normalized_proxy_score"), ("mean_judge_score", "normalized_judge_score"),
                ("mean_gap", "gap"), ("mean_response_tokens", "response_tokens"))},
            "eos_fraction": sum(row["finish_reason"] == "eos" for row in rows) / count}


def evaluate(actor: Actor, proxy: Scorer, judge: Scorer, calibration: FrozenCalibration, *,
             prepared_dir: str | Path, cohort: Literal["validation", "final_evaluation"],
             output_dir: str | Path, policy_id: str, seed: int, batch_size: int = 4,
             memory: GapMemory | None = None, memory_id: str | None = None,
             save_embeddings: bool = False) -> EvaluationResult:
    """Evaluate one answer per prepared prompt and save rows plus a completion marker.

    Optional memory is queried only; evaluation examples never enter memory.
    Use resolved config paths. An existing output directory, even failed, is refused.
    """
    if cohort not in ("validation", "final_evaluation"):
        raise EvaluationError("Evaluation requires validation or final_evaluation cohort")
    if type(save_embeddings) is not bool:
        raise EvaluationError("save_embeddings must be boolean")
    if (memory is None) != (memory_id is None) or (memory_id is not None and (
            not isinstance(memory_id, str) or not memory_id.strip() or memory_id != memory_id.strip())):
        raise EvaluationError("Provide a memory snapshot and its memory_id together")
    prepared = _resolved_path(prepared_dir)
    prompts = load_prompts(prepared / f"{cohort}.json")
    _validate(prompts, proxy, judge, policy_id=policy_id, seed=seed, batch_size=batch_size)
    destination = _resolved_path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_write_json(destination / "status.json", {"state": "running"})
    try:
        result = _collect(actor, proxy, judge, calibration, prompts, seed=seed, batch_size=batch_size,
                          need_embeddings=save_embeddings or memory is not None, memory=memory)
        rows_path = atomic_write_json(destination / "rows.json", result.rows)
        manifest = _manifest(actor, calibration, policy_id=policy_id, prepared_dir=prepared, cohort=cohort,
                             seed=seed, batch_size=batch_size, rows=result.rows)
        manifest.update(operation="evaluation", memory_id=memory_id, rows_file="rows.json", embeddings_file=None)
        manifest["embedding_context"] = asdict(result.context) if result.context is not None else None
        if save_embeddings and result.embeddings is not None and result.context is not None:
            atomic_write_json(destination / "embeddings.json", {
                "context": asdict(result.context), "example_ids": [r["example_id"] for r in result.rows],
                "vectors": result.embeddings.tolist()})
            manifest["embeddings_file"] = "embeddings.json"
        if memory is not None:
            manifest["metrics"]["gap_prediction_mae"] = sum(abs(r["gap_prediction_error"]) for r in result.rows) / len(result.rows)
        manifest_path = atomic_write_json(destination / "manifest.json", manifest)
        atomic_write_json(destination / "status.json", {"state": "completed"})
        return EvaluationResult(destination, rows_path, manifest_path, len(result.rows))
    except BaseException as exc:
        atomic_write_json(destination / "status.json", {"state": "failed", "error": str(exc)})
        raise
