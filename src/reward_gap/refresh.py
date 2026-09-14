"""Label the prepared refresh cohort and save a new memory; never train policy weights."""

from dataclasses import dataclass
from pathlib import Path

from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration
from reward_gap.config import COHORTS
from reward_gap.data import load_prompts
from reward_gap.evaluation import Actor, Scorer, EvaluationError, _collect, _manifest, _validate, _resolved_path
from reward_gap.memory import GapMemory


class RefreshError(ValueError):
    """Refresh cohort or memory version is invalid."""


@dataclass(frozen=True)
class RefreshResult:
    memory: GapMemory
    output_dir: Path
    memory_path: Path
    rows_path: Path
    manifest_path: Path
    added_count: int


def refresh_memory(actor: Actor, proxy: Scorer, judge: Scorer, calibration: FrozenCalibration,
                   memory: GapMemory, *, prepared_dir: str | Path, output_dir: str | Path,
                   policy_id: str, parent_memory_id: str, memory_id: str,
                   seed: int, batch_size: int = 4) -> RefreshResult:
    """Generate one answer per refresh prompt, append calibrated labels and save M1.

    All six prepared cohorts must exist and refresh groups must be disjoint.
    A fresh output directory holds rows.json, memory.json, manifest.json and status.json.
    Only a completed status marks an artifact ready for the coordinator to use.
    """
    for value in (parent_memory_id, memory_id):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise RefreshError("Memory version IDs must be nonempty strings without surrounding whitespace")
    if memory_id == parent_memory_id:
        raise RefreshError("Refreshed memory needs a new version ID")
    if (memory.context.calibration_id != calibration.calibration_id
            or memory.context.encoder_id != calibration.proxy.source
            or memory.context.encoder_revision != calibration.proxy.revision):
        raise RefreshError("Memory and calibration are incompatible")
    prepared = _resolved_path(prepared_dir)
    prompts = load_prompts(prepared / "refresh.json")
    _validate(prompts, proxy, judge, policy_id=policy_id, seed=seed, batch_size=batch_size)
    groups = {p.conversation_group for p in prompts}
    ids = {p.prompt_id for p in prompts}
    for cohort in COHORTS:
        if cohort == "refresh":
            continue
        other = load_prompts(prepared / f"{cohort}.json")
        if groups.intersection(p.conversation_group for p in other) or ids.intersection(p.prompt_id for p in other):
            raise RefreshError(f"Refresh prompts overlap the {cohort} cohort")
    new_ids = [f"{memory_id}/example-{i:08d}" for i in range(len(prompts))]
    if set(new_ids).intersection(memory.example_ids):
        raise RefreshError("These refresh example IDs already exist in memory")
    destination = _resolved_path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_write_json(destination / "status.json", {"state": "running"})
    try:
        labeled = _collect(actor, proxy, judge, calibration, prompts, seed=seed,
                           batch_size=batch_size, need_embeddings=True)
        if labeled.embeddings is None or labeled.context is None:
            raise EvaluationError("Refresh requires proxy embeddings")
        for row, example_id in zip(labeled.rows, new_ids, strict=True):
            row["example_id"] = example_id
        updated = memory.append(new_ids, labeled.embeddings,
                                proxy_scores=[r["normalized_proxy_score"] for r in labeled.rows],
                                judge_scores=[r["normalized_judge_score"] for r in labeled.rows],
                                context=labeled.context)
        rows_path = atomic_write_json(destination / "rows.json", labeled.rows)
        memory_path = updated.save(destination / "memory.json")
        manifest = _manifest(actor, calibration, policy_id=policy_id, prepared_dir=prepared,
                             cohort="refresh", seed=seed, batch_size=batch_size, rows=labeled.rows)
        manifest.update(operation="refresh", parent_memory_id=parent_memory_id, memory_id=memory_id,
                        previous_count=memory.size, added_count=len(new_ids), total_count=updated.size,
                        rows_file="rows.json", memory_file="memory.json")
        manifest_path = atomic_write_json(destination / "manifest.json", manifest)
        atomic_write_json(destination / "status.json", {"state": "completed"})
        return RefreshResult(updated, destination, memory_path, rows_path, manifest_path, len(new_ids))
    except BaseException as exc:
        atomic_write_json(destination / "status.json", {"state": "failed", "error": str(exc)})
        raise
