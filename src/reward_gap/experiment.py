"""Sequential two-round comparison: proxy-only, static M0 and refreshed M1."""

import gc
import argparse
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from filelock import FileLock

from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration
from reward_gap.config import COHORTS, ExperimentConfig
from reward_gap.data import load_prompts
from reward_gap.evaluation import _collect, evaluate
from reward_gap.memory import GapMemory, MemoryContext
from reward_gap.policy import PPOActor
from reward_gap.ppo import PPOTrainer
from reward_gap.refresh import refresh_memory
from reward_gap.rewards import KNNReward, ProxyReward
from reward_gap.scorers import RewardScorer


class ExperimentError(ValueError):
    """Incompatible artifacts or invalid experiment stage state."""


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    status_path: Path
    summary_path: Path | None
    state: str


def _read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class FollowupExperiment:
    """Coordinate existing modules; no PPO equations and no concurrent branches.

    Factories load fresh actors/scorers and allow tiny offline integration tests.
    run_dir is a new or compatible existing child of runtime.output_root.
    Failed labeling stages retry in a new attempt directory. Training resumes
    its one rolling checkpoint, retaining only final and corrected-round1 states.
    """

    retain_analysis_checkpoints = False

    def __init__(self, config: ExperimentConfig, run_dir: str | Path, *,
                 actor_factory: Callable[[int], PPOActor] | None = None,
                 scorer_factory: Callable[[Literal["proxy", "judge"]], RewardScorer] | None = None):
        if config.data is None:
            raise ExperimentError("Prepare the six data cohorts before running the experiment")
        if not config.generation.do_sample:
            raise ExperimentError("The training experiment requires sampled generation")
        if any(type(s) is not int or not 0 <= s < 2**63 for s in config.seeds) or not config.seeds:
            raise ExperimentError("Provide experiment seeds in [0, 2**63)")
        if len(set(config.seeds)) != len(config.seeds):
            raise ExperimentError("Experiment seeds must be unique")
        root, target = config.runtime.output_root, Path(run_dir)
        if not root.is_absolute() or not target.is_absolute() or not config.data.prepared_dir.is_absolute():
            raise ExperimentError("Use absolute paths from load_config")
        self.run_dir = target.resolve()
        if self.run_dir == root.resolve() or not self.run_dir.is_relative_to(root.resolve()):
            raise ExperimentError("run_dir must be a child of runtime.output_root")
        self.config, self.prepared = config, config.data.prepared_dir
        self.actor_factory = actor_factory or (lambda seed: PPOActor.load(config, seed=seed))
        self.scorer_factory = scorer_factory or (lambda role: RewardScorer.load(config, role))
        self.status: dict = {}
        self.cohorts: dict = {}
        self.schedules: dict = {}

    def validate_inputs(self) -> dict:
        """Read and validate prepared cohorts/schedules without creating a run."""
        manifest = _read(self.prepared / "input_manifest.json")
        if manifest.get("schema_version") != 1:
            raise ExperimentError("Unsupported prepared-data manifest")
        data = self.config.data
        assert data is not None
        for key, expected in (("split_seed", data.split_seed), ("minimum_prompts", data.minimum_prompts),
                              ("schedule_updates", self.config.training.total_updates),
                              ("schedule_batch_size", self.config.training.rollout_batch_size)):
            if key in manifest and manifest[key] != expected:
                raise ExperimentError(f"Prepared data {key} does not match configuration; prepare a new directory")
        if "subsets" in manifest and set(manifest["subsets"]) != set(data.subsets):
            raise ExperimentError("Prepared data subsets do not match configuration")
        self.cohorts = {name: load_prompts(self.prepared / f"{name}.json") for name in COHORTS}
        groups, ids = set(), set()
        for name, records in self.cohorts.items():
            if not records and name != "validation":
                raise ExperimentError(f"Empty required cohort: {name}")
            current_groups, current_ids = {p.conversation_group for p in records}, {p.prompt_id for p in records}
            if groups & current_groups or ids & current_ids:
                raise ExperimentError(f"Cohort overlap detected at {name}")
            groups.update(current_groups)
            ids.update(current_ids)
        if len(self.cohorts["initial_memory"]) < self.config.memory.k:
            raise ExperimentError("Initial memory cohort must contain at least memory.k examples")
        training = {p.prompt_id: p for p in self.cohorts["training"]}
        for seed in self.config.seeds:
            schedule = _read(self.prepared / f"training_schedule_seed{seed}.json")
            if (not isinstance(schedule, list) or len(schedule) != self.config.training.total_updates
                    or any(not isinstance(batch, list) or len(batch) != self.config.training.rollout_batch_size
                           or any(not isinstance(pid, str) or pid not in training for pid in batch) for batch in schedule)):
                raise ExperimentError(f"Invalid prepared training schedule for seed {seed}")
            self.schedules[seed] = [[training[pid] for pid in batch] for batch in schedule]
        inputs = {"manifest": manifest, "cohorts": {n: [asdict(p) for p in rows] for n, rows in self.cohorts.items()},
                  "schedules": {str(s): [[p.prompt_id for p in b] for b in batches] for s, batches in self.schedules.items()}}
        # Canonical JSON also converts tuple-valued message sequences for comparisons.
        return json.loads(json.dumps(inputs))

    def _open(self) -> None:
        inputs = self.validate_inputs()
        status_path = self.run_dir / "status.json"
        if status_path.exists():
            if (_read(self.run_dir / "resolved_config.json") != self.config.to_dict()
                    or _read(self.run_dir / "inputs.json") != inputs):
                raise ExperimentError("Configuration or prepared inputs changed; use a new run directory")
            self.status = _read(status_path)
            if self.status.get("schema_version") != 1:
                raise ExperimentError("Unsupported experiment status schema")
            if self.status.get("response_contract") != "primary-eos-contiguous-nonpad-v1":
                raise ExperimentError("Historical response contract differs; use a new run or hh-evaluate for retained HH checkpoints")
        else:
            # A lock file is harmless; other files indicate an interrupted initial setup.
            if any(p.name != ".run.lock" for p in self.run_dir.iterdir()):
                raise ExperimentError("Run directory has files but no experiment status; choose a new directory")
            atomic_write_json(self.run_dir / "resolved_config.json", self.config.to_dict())
            atomic_write_json(self.run_dir / "inputs.json", inputs)
            self.status = {"schema_version": 1, "state": "ready", "stages": {}, "models": {},
                           "response_contract": "primary-eos-contiguous-nonpad-v1"}
            self._status()

    def _status(self) -> None:
        atomic_write_json(self.run_dir / "status.json", self.status)

    def _stage(self, name: str, work: Callable[[Path], dict]) -> dict:
        entry = self.status["stages"].setdefault(name, {"state": "pending", "attempts": []})
        if entry["state"] == "completed":
            return entry["result"]
        attempt = len(entry["attempts"]) + 1
        folder = self.run_dir / "stages" / name / f"attempt-{attempt:03d}"
        # A killed process may have created a folder before recording its attempt.
        while folder.exists():
            attempt += 1
            folder = self.run_dir / "stages" / name / f"attempt-{attempt:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        entry["attempts"].append(str(folder.relative_to(self.run_dir)))
        entry["state"] = "running"
        self.status["current_stage"] = name
        self._status()
        try:
            result = work(folder)
            atomic_write_json(folder / "result.json", result)
            entry.update(state="completed", result=result)
            entry.pop("error", None)
            self._status()
            return result
        except BaseException as exc:
            entry.update(state="failed", error=str(exc))
            self._status()
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _model(self, role: str, source: str, revision: str | None) -> None:
        actual = {"source": source, "revision": revision}
        old = self.status["models"].get(role)
        if old is not None and old != actual:
            raise ExperimentError(f"Resolved {role} model changed; use the original revision")
        self.status["models"][role] = actual
        self._status()

    def _actor(self, seed: int) -> PPOActor:
        gc.collect()
        actor = self.actor_factory(seed)
        self._model("policy", actor.source, actor.revision)
        return actor

    def _scorer(self, role: Literal["proxy", "judge"]) -> RewardScorer:
        scorer = self.scorer_factory(role)
        self._model(role, scorer.loaded.source, scorer.loaded.revision)
        return scorer

    @staticmethod
    def _seed(seed: int, stage: str) -> int:
        return random.Random(f"{seed}:{stage}").randrange(2**62)

    def _calibration(self, folder: Path, actor: PPOActor, proxy: RewardScorer) -> dict:
        judge = self._scorer("judge")
        records = self.cohorts["calibration"]
        proxy_batches, judge_batches, rows = [], [], []
        size = self.config.training.rollout_batch_size
        seed = self._seed(self.config.seeds[0], "calibration")
        for start in range(0, len(records), size):
            chunk = records[start:start + size]
            rollout = actor.generate(chunk, seed=seed + start // size)
            expected = tuple(p.prompt_id for p in chunk)
            if rollout.prompt_ids != expected or len(rollout.answers) != len(chunk):
                raise ExperimentError("Calibration generation changed prompt alignment")
            pb, jb = proxy.score(chunk, rollout.answers), judge.score(chunk, rollout.answers)
            if pb.prompt_ids != expected or jb.prompt_ids != expected:
                raise ExperimentError("Calibration scorers changed prompt alignment")
            # Validate each batch's source and numeric values before concatenating.
            for batch, role in ((pb, "proxy"), (jb, "judge")):
                if batch.role != role or len(batch.scores) != len(chunk):
                    raise ExperimentError("Invalid calibration score batch")
                self._model(role, batch.source, batch.revision)
            proxy_batches.append(pb)
            judge_batches.append(jb)
            rows.extend({"prompt": asdict(p), "answer": answer, "proxy_score": ps, "judge_score": js,
                         "seed": rollout.seed} for p, answer, ps, js in zip(chunk, rollout.answers, pb.scores, jb.scores, strict=True))
        def combined(batches):
            return replace(batches[0], prompt_ids=tuple(x for b in batches for x in b.prompt_ids),
                           scores=tuple(x for b in batches for x in b.scores),
                           token_counts=tuple(x for b in batches for x in b.token_counts))
        calibration = FrozenCalibration.fit(combined(proxy_batches), combined(judge_batches),
                                            calibration_id=f"{self.run_dir.name}/calibration")
        calibration.save(folder / "calibration.json")
        atomic_write_json(folder / "rows.json", rows)
        return {"calibration": str(folder / "calibration.json"), "rows": str(folder / "rows.json")}

    def _initial_memory(self, folder: Path, actor: PPOActor, proxy: RewardScorer, calibration: FrozenCalibration) -> dict:
        judged = _collect(actor, proxy, self._scorer("judge"), calibration, self.cohorts["initial_memory"],
                          seed=self._seed(self.config.seeds[0], "initial-memory"),
                          batch_size=self.config.training.rollout_batch_size, need_embeddings=True)
        if judged.embeddings is None or judged.context is None:
            raise ExperimentError("Initial memory requires proxy embeddings")
        for row in judged.rows:
            row["example_id"] = "M0/" + row["example_id"]
        memory = GapMemory.build([r["example_id"] for r in judged.rows], judged.embeddings,
                                 proxy_scores=[r["normalized_proxy_score"] for r in judged.rows],
                                 judge_scores=[r["normalized_judge_score"] for r in judged.rows],
                                 context=judged.context, k=self.config.memory.k, temperature=self.config.memory.temperature)
        memory.save(folder / "memory.json")
        atomic_write_json(folder / "rows.json", judged.rows)
        return {"memory": str(folder / "memory.json"), "context": asdict(memory.context), "count": memory.size}

    def _trainer(self, actor, reward, seed: int, reward_id: str) -> PPOTrainer:
        return PPOTrainer(actor, reward, self.config.training, experiment_id=f"{self.run_dir}/seed-{seed}",
                          reward_id=reward_id, seed=seed)

    def _train(self, folder: Path, actor: PPOActor, reward, seed: int, reward_id: str,
               source: Path, source_reward_id: str, checkpoint: Path, stop: int, branch: str) -> dict:
        trainer = self._trainer(actor, reward, seed, reward_id)
        latest = self.run_dir / f"seed-{seed}" / branch / "latest.pt"
        metrics_path = latest.parent / "metrics.json"
        try:
            if checkpoint.exists():
                trainer.load_checkpoint(checkpoint)
                if trainer.update_count != stop:
                    raise ExperimentError("Boundary checkpoint has incorrect progress")
                latest.unlink(missing_ok=True)
                return {"checkpoint": str(checkpoint), "update": stop}
            if latest.exists():
                trainer.load_checkpoint(latest)
            elif reward_id == source_reward_id:
                trainer.load_checkpoint(source)
            else:
                trainer.fork_checkpoint(source, expected_reward_id=source_reward_id)
            if trainer.update_count > stop:
                raise ExperimentError("Recovery checkpoint is beyond this stage")
            history = _read(metrics_path) if metrics_path.exists() else []
            history = [m for m in history if m["update"] <= trainer.update_count]
            batches = self.schedules[seed]
            seeds = [self._seed(seed, f"training-{i}") for i in range(len(batches))]
            while trainer.update_count < stop:
                history.extend(asdict(m) for m in trainer.train(batches, seeds, until_update=trainer.update_count + 1))
                atomic_write_json(metrics_path, history)
                if self.retain_analysis_checkpoints and trainer.update_count == 1 and branch in ("raw", "static"):
                    first = latest.parent / "update-1.pt"
                    if not first.exists():
                        trainer.save_checkpoint(first)
                if trainer.update_count < stop and trainer.update_count % self.config.training.checkpoint_every == 0:
                    trainer.save_checkpoint(latest, replace_existing=True)
            trainer.save_checkpoint(checkpoint)
            latest.unlink(missing_ok=True)  # Boundary state is safely published first.
            return {"checkpoint": str(checkpoint), "update": stop, "metrics": str(metrics_path)}
        finally:
            trainer.release()

    def _round1(self, seed: int, proxy: RewardScorer, calibration: FrozenCalibration, memory: GapMemory) -> None:
        prefix = f"seed-{seed}"
        raw_id, static_id = f"proxy/{calibration.calibration_id}", f"knn/{calibration.calibration_id}/M0"
        raw_done = self.status["stages"].get(f"{prefix}/raw-round1", {}).get("state") == "completed"
        static_done = self.status["stages"].get(f"{prefix}/corrected-round1", {}).get("state") == "completed"
        initial = self.run_dir / prefix / "initial.pt"
        if raw_done and static_done:
            if not self.retain_analysis_checkpoints:
                initial.unlink(missing_ok=True)
            return
        actor = self._actor(seed)
        if not initial.exists():
            trainer = self._trainer(actor, ProxyReward(proxy, calibration), seed, raw_id)
            try:
                trainer.save_checkpoint(initial)
            finally:
                trainer.release()
        for branch, name, reward, rid, checkpoint in (
            ("raw", "raw-round1", ProxyReward(proxy, calibration), raw_id, self.run_dir / prefix / "raw" / "round1.pt"),
            ("static", "corrected-round1", KNNReward(proxy, calibration, memory), static_id, self.run_dir / prefix / "corrected_round1.pt"),
        ):
            self._stage(f"{prefix}/{name}", lambda folder, b=branch, r=reward, identity=rid, path=checkpoint:
                        self._train(folder, actor, r, seed, identity, initial, raw_id, path,
                                    self.config.training.round1_updates, b))
        if not self.retain_analysis_checkpoints:
            initial.unlink(missing_ok=True)

    def _refresh(self, folder: Path, actor, proxy, calibration, memory, seed, checkpoint, reward_id) -> dict:
        trainer = self._trainer(actor, KNNReward(proxy, calibration, memory), seed, reward_id)
        try:
            trainer.load_checkpoint(checkpoint)
        finally:
            trainer.release()
        result = refresh_memory(actor, proxy, self._scorer("judge"), calibration, memory,
                                prepared_dir=self.prepared, output_dir=folder / "artifacts",
                                policy_id=f"seed-{seed}/corrected_round1", parent_memory_id="M0",
                                memory_id=f"M1-seed-{seed}", seed=self._seed(seed, "refresh"),
                                batch_size=self.config.training.rollout_batch_size)
        return {"memory": str(result.memory_path), "manifest": str(result.manifest_path)}

    def _round2(self, seed, proxy, calibration, memory):
        prefix = f"seed-{seed}"
        actor = self._actor(seed)
        raw_id, static_id = f"proxy/{calibration.calibration_id}", f"knn/{calibration.calibration_id}/M0"
        fork = self.run_dir / prefix / "corrected_round1.pt"
        refreshed = self._stage(f"{prefix}/refresh", lambda folder:
                                self._refresh(folder, actor, proxy, calibration, memory, seed, fork, static_id))
        m1 = GapMemory.load(refreshed["memory"], context=memory.context)
        for branch, reward, rid, source, source_id in (
            ("raw", ProxyReward(proxy, calibration), raw_id, self.run_dir / prefix / "raw/round1.pt", raw_id),
            ("static", KNNReward(proxy, calibration, memory), static_id, fork, static_id),
            ("iterative", KNNReward(proxy, calibration, m1), f"knn/{calibration.calibration_id}/M1-seed-{seed}", fork, static_id),
        ):
            final = self.run_dir / prefix / branch / "final.pt"
            self._stage(f"{prefix}/{branch}-final", lambda folder, b=branch, r=reward, identity=rid, src=source, src_id=source_id, dest=final:
                        self._train(folder, actor, r, seed, identity, src, src_id, dest, self.config.training.total_updates, b))
            if branch == "raw" and not self.retain_analysis_checkpoints:
                source.unlink(missing_ok=True)  # Raw final replaces its round-one recovery state.

    def _evaluate(self, folder, actor, proxy, calibration, memory, seed, branch, reward, reward_id):
        trainer = self._trainer(actor, reward, seed, reward_id)
        try:
            trainer.load_checkpoint(self.run_dir / f"seed-{seed}" / branch / "final.pt")
            if trainer.update_count != self.config.training.total_updates:
                raise ExperimentError("Final evaluation requires completed training")
        finally:
            trainer.release()
        result = evaluate(actor, proxy, self._scorer("judge"), calibration,
                          prepared_dir=self.prepared, cohort="final_evaluation", output_dir=folder / "artifacts",
                          policy_id=f"seed-{seed}/{branch}/final", seed=self._seed(seed, "final-evaluation"),
                          batch_size=self.config.training.rollout_batch_size,
                          memory=memory if branch != "raw" else None,
                          memory_id=("M0" if branch == "static" else f"M1-seed-{seed}") if branch != "raw" else None)
        return {"manifest": str(result.manifest_path), "rows": str(result.rows_path)}

    def run(self, *, until: Literal["round1", "training", "complete"] = "complete") -> ExperimentResult:
        if until not in ("round1", "training", "complete"):
            raise ExperimentError("until must be round1, training or complete")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.run_dir / ".run.lock"), timeout=0):
            self._open()
            if self.status["state"] == "completed":
                self._verify_completed()
                return ExperimentResult(self.run_dir, self.run_dir / "status.json", self.run_dir / "summary.json", "completed")
            self.status["state"] = "running"
            self.status.pop("error", None)
            self._status()
            try:
                proxy = self._scorer("proxy")
                actor = self._actor(self.config.seeds[0])
                fitted = self._stage("calibration", lambda folder: self._calibration(folder, actor, proxy))
                calibration = FrozenCalibration.load(fitted["calibration"])
                initial = self._stage("initial-memory", lambda folder: self._initial_memory(folder, actor, proxy, calibration))
                memory = GapMemory.load(initial["memory"], context=MemoryContext(**initial["context"]))
                del actor
                for seed in self.config.seeds:
                    self._round1(seed, proxy, calibration, memory)
                if until != "round1":
                    for seed in self.config.seeds:
                        self._round2(seed, proxy, calibration, memory)
                if until == "complete":
                    # No final labels are generated until ALL seeds/branches are trained.
                    for seed in self.config.seeds:
                        for branch in ("raw", "static", "iterative"):
                            if not (self.run_dir / f"seed-{seed}" / branch / "final.pt").is_file():
                                raise ExperimentError("A final policy checkpoint is missing")
                    results = {}
                    for seed in self.config.seeds:
                        actor = self._actor(seed)
                        refreshed = self.status["stages"][f"seed-{seed}/refresh"]["result"]
                        m1 = GapMemory.load(refreshed["memory"], context=memory.context)
                        for branch, mem, reward, rid in (
                            ("raw", memory, ProxyReward(proxy, calibration), f"proxy/{calibration.calibration_id}"),
                            ("static", memory, KNNReward(proxy, calibration, memory), f"knn/{calibration.calibration_id}/M0"),
                            ("iterative", m1, KNNReward(proxy, calibration, m1), f"knn/{calibration.calibration_id}/M1-seed-{seed}"),
                        ):
                            result = self._stage(f"seed-{seed}/evaluate-{branch}", lambda folder, b=branch, m=mem, r=reward, identity=rid:
                                                 self._evaluate(folder, actor, proxy, calibration, m, seed, b, r, identity))
                            results[f"seed-{seed}/{branch}"] = {**result, "metrics": _read(Path(result["manifest"]))["metrics"]}
                        del actor
                    atomic_write_json(self.run_dir / "summary.json", {"schema_version": 1, "results": results})
                self.status["state"] = "completed" if until == "complete" else "paused"
                self.status["current_stage"] = None
                self._status()
            except BaseException as exc:
                self.status.update(state="failed", error=str(exc))
                self._status()
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return ExperimentResult(self.run_dir, self.run_dir / "status.json",
                                self.run_dir / "summary.json" if until == "complete" else None, self.status["state"])

    def _verify_completed(self) -> None:
        """Check required artifact presence without loading models or checkpoint tensors."""
        required = [self.run_dir / "summary.json",
                    Path(self.status["stages"]["calibration"]["result"]["calibration"]),
                    Path(self.status["stages"]["initial-memory"]["result"]["memory"])]
        for seed in self.config.seeds:
            required.append(self.run_dir / f"seed-{seed}" / "corrected_round1.pt")
            required.append(Path(self.status["stages"][f"seed-{seed}/refresh"]["result"]["memory"]))
            for branch in ("raw", "static", "iterative"):
                required.append(self.run_dir / f"seed-{seed}" / branch / "final.pt")
                result = self.status["stages"][f"seed-{seed}/evaluate-{branch}"]["result"]
                required.extend((Path(result["rows"]), Path(result["manifest"])))
                status = Path(result["manifest"]).parent / "status.json"
                if not status.is_file() or _read(status).get("state") != "completed":
                    raise ExperimentError("Final evaluation is not marked completed")
        if any(not path.is_file() for path in required):
            raise ExperimentError("Completed experiment is missing required artifacts")


def main() -> None:
    from reward_gap.config import load_config
    parser = argparse.ArgumentParser(description="Run or resume the two-round reward-gap experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True, help="Directory name within configured output_root")
    parser.add_argument("--until", choices=("round1", "training", "complete"), default="complete")
    args = parser.parse_args()
    config = load_config(args.config)
    result = FollowupExperiment(config, config.runtime.output_root / args.run_name).run(until=args.until)
    print(f"Experiment {result.state}: {result.status_path}")


if __name__ == "__main__":
    main()
