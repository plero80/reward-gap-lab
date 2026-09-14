"""Three-arm GSM8K follow-up using TRL PPO and independently checked answers."""

import gc
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.version import cuda as torch_cuda
from filelock import FileLock

from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration
from reward_gap.experiment import ExperimentResult, FollowupExperiment
from reward_gap.memory import MemoryContext
from reward_gap.policy import PPOActor
from reward_gap.ppo import PPOTrainer, load_policy_checkpoint
from reward_gap.scorers import ScoreBatch
from reward_gap.gsm8k.answers import VERSION, evaluate_answer
from reward_gap.gsm8k.config import GSMConfig
from reward_gap.gsm8k.data import load_prepared, schedule
from reward_gap.gsm8k.graders import LanguageGrader, RUBRIC_VERSION
from reward_gap.gsm8k.memory import QuestionMemory
from reward_gap.gsm8k.metrics import gap_metrics
from reward_gap.gsm8k.rewards import MathReward

PROTOCOL = "gsm8k_common_penalties_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class GSMExperiment(FollowupExperiment):
    """Reuse generic atomic stage bookkeeping, with GSM8K-specific data and roles."""

    def __init__(self, config: GSMConfig, run_dir, *, actor_factory=None, scorer_factory=None):
        self.gsm_config, self.config, self.settings = config, config.base, config.settings
        self.run_dir = Path(run_dir).resolve()
        root = config.base.runtime.output_root.resolve()
        if self.run_dir == root or not self.run_dir.is_relative_to(root):
            raise ValueError("GSM8K run directory must be inside output_root")
        self.cohorts, self.manifest = load_prepared(config)
        self.questions = {q.id: q for rows in self.cohorts.values() for q in rows}
        self.actor_factory = actor_factory or (lambda seed: PPOActor.load(self.config, seed=seed))
        self.scorer_factory = scorer_factory or (lambda role: LanguageGrader.load(config, role, self.questions, self.run_dir))
        self.status = {}

    def _scorer(self, role) -> Any:
        # The stage coordinator also accepts injected graders for CPU tests.
        scorer = self.scorer_factory(role)
        self._model(role, scorer.loaded.source, scorer.loaded.revision)
        return scorer

    def _actor(self, seed):
        actor = super()._actor(seed)
        # Native TRL PPO supports one stop token. Use that same stop condition
        # for preparation and evaluation so length penalties have one meaning.
        eos = actor.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise ValueError("GSM8K policy needs a primary EOS token")
        actor.eos_ids = (eos,)
        return actor

    def _open_gsm(self):
        snapshot = {"protocol": getattr(self, "protocol", PROTOCOL), "answer_parser": VERSION, "rubric": RUBRIC_VERSION,
                    "policy_stop_rule": "tokenizer_primary_eos",
                    "config": self.gsm_config.to_dict(), "manifest": self.manifest,
                    "cohorts": {name: [asdict(q) for q in rows] for name, rows in self.cohorts.items()}}
        path = self.run_dir / "resolved_protocol.json"
        if path.exists():
            if read(path) != snapshot:
                raise ValueError("GSM8K protocol or data changed; use a new run name")
            self.status = read(self.run_dir / "status.json")
        else:
            if any(p.name != ".run.lock" for p in self.run_dir.iterdir()):
                raise ValueError("Run directory is not empty")
            atomic_write_json(path, snapshot)
            atomic_write_json(self.run_dir / "environment.json", {
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in
                             ("torch", "transformers", "peft", "trl", "accelerate", "numpy", "datasets", "matplotlib", "pyarrow")},
                "torch_cuda": torch_cuda, "device": self.config.runtime.device,
            })
            self.status = {"schema_version": 1, "kind": "gsm8k_rq2", "state": "ready", "stages": {}, "models": {}}
            self._status()

    def _phase(self, name):
        self.proxy.phase = self.judge.phase = name

    def _labels(self, actor, cohort, seed, repeats, *, greedy=False):
        original = actor.generation
        actor.generation = replace(original, do_sample=not greedy)
        rows, vectors, metadata = [], [], None
        try:
            for sample in range(repeats):
                for start in range(0, len(self.cohorts[cohort]), self.config.scoring.batch_size):
                    questions = self.cohorts[cohort][start:start + self.config.scoring.batch_size]
                    prompts = [q.prompt() for q in questions]
                    rollout = actor.generate(prompts, seed=self._seed(seed, f"{cohort}/{sample}/{start}"),
                                             temperature=1. if greedy else self.settings["policy_temperature"])
                    pb = self.proxy.score(prompts, rollout.answers, return_embeddings=True)
                    jb = self.judge.score(prompts, rollout.answers)
                    expected = tuple(q.id for q in questions)
                    if rollout.prompt_ids != expected or pb.prompt_ids != expected or jb.prompt_ids != expected:
                        raise ValueError("GSM8K generation/grading alignment differs")
                    current = {"proxy": {"source": pb.source, "revision": pb.revision},
                               "judge": {"source": jb.source, "revision": jb.revision}, "pooling": pb.embedding_pooling}
                    if metadata is not None and metadata != current:
                        raise ValueError("Grader identity changed within labeling stage")
                    metadata = current
                    if pb.embeddings is None or len(pb.embeddings) != len(questions):
                        raise ValueError("Missing proxy representations")
                    vectors.extend(pb.embeddings.tolist())
                    for i, question in enumerate(questions):
                        rows.append({"example_id": f"{question.id}/sample-{sample}", "question_id": question.id,
                                     "question": question.question, "answer": rollout.answers[i],
                                     "raw_proxy": pb.scores[i], "raw_judge": jb.scores[i],
                                     "proxy_tokens": pb.token_counts[i], "judge_tokens": jb.token_counts[i],
                                     "response_tokens": rollout.response_lengths[i], "finish_reason": rollout.finish_reasons[i],
                                     **evaluate_answer(rollout.answers[i], question.gold, rollout.finish_reasons[i])})
        finally:
            actor.generation = original
        if metadata is None:
            raise ValueError("Cannot label an empty cohort")
        return {"rows": rows, "embeddings": vectors, **metadata}

    @staticmethod
    def _batch(data, role):
        return ScoreBatch(tuple(r["example_id"] for r in data["rows"]),
                          tuple(r[f"raw_{role}"] for r in data["rows"]),
                          tuple(r[f"{role}_tokens"] for r in data["rows"]), role,
                          data[role]["source"], data[role]["revision"])

    def _normalized(self, data, calibration):
        zp = calibration.normalize_proxy(self._batch(data, "proxy"))
        zj = calibration.normalize_judge(self._batch(data, "judge"))
        for row, p, j in zip(data["rows"], zp, zj, strict=True):
            row.update(proxy=p, judge=j, gap=p - j)
        return data

    def _prepare_seed(self, folder, actor, seed):
        self._phase(f"seed-{seed}/calibration")
        data = self._labels(actor, "calibration", seed, self.settings["preparation_responses"])
        calibration = FrozenCalibration.fit(self._batch(data, "proxy"), self._batch(data, "judge"),
                                            calibration_id=f"{self.run_dir.name}/seed-{seed}/{PROTOCOL}")
        self._normalized(data, calibration)
        theta = float(np.quantile([r["gap"] for r in data["rows"]], self.settings["gap_quantile"]))
        calibration.save(folder / "calibration.json")
        atomic_write_json(folder / "calibration_rows.json", data)
        return {"calibration": str(folder / "calibration.json"), "theta": theta}

    def _build_memory(self, folder, actor, seed, calibration):
        self._phase(f"seed-{seed}/memory")
        training = self._normalized(self._labels(actor, "memory", seed, self.settings["preparation_responses"]), calibration)
        self._phase(f"seed-{seed}/selection")
        selection = self._normalized(self._labels(actor, "selection", seed, self.settings["preparation_responses"]), calibration)
        context = MemoryContext(training["proxy"]["source"], training["proxy"]["revision"], training["pooling"], calibration.calibration_id)
        if selection["pooling"] != context.pooling:
            raise ValueError("Selection and memory representations differ")
        grid, best, best_error = [], None, float("inf")
        for k in sorted(self.settings["k_values"]):
            for temperature in sorted(self.settings["similarity_temperatures"]):
                memory = QuestionMemory(training["rows"], torch.tensor(training["embeddings"]), context, k=k, temperature=temperature)
                predicted, _ = memory.predict(torch.tensor(selection["embeddings"]), [r["question_id"] for r in selection["rows"]], context=context)
                error = float(np.mean((np.array(predicted) - [r["gap"] for r in selection["rows"]]) ** 2))
                grid.append({"k": k, "temperature": temperature, "selection_mse": error})
                if error < best_error:
                    best, best_error = memory, error
        if best is None:
            raise ValueError("No finite memory selection result")
        atomic_write_json(folder / "memory.json", best.to_dict())
        atomic_write_json(folder / "selection.json", {"data": selection, "grid": grid})
        return {"memory": str(folder / "memory.json"), "selection": str(folder / "selection.json")}

    def _evaluate_math(self, folder, actor, seed, arm, update, cohort, calibration, memory, theta, kl):
        self._phase(f"seed-{seed}/evaluation/{cohort}/{arm}/{update}")
        data = self._normalized(self._labels(actor, cohort, seed, 1, greedy=True), calibration)
        context = MemoryContext(data["proxy"]["source"], data["proxy"]["revision"], data["pooling"], calibration.calibration_id)
        predicted, neighbors = memory.predict(torch.tensor(data["embeddings"]), [r["question_id"] for r in data["rows"]], context=context)
        for row, gap, nearby in zip(data["rows"], predicted, neighbors, strict=True):
            row.update(predicted_gap=gap, neighbors=nearby, high_gap=row["gap"] > theta,
                       format_penalty=self.settings["format_penalty"] * (not row["format_compliant"]),
                       length_penalty=self.settings["length_penalty"] * row["length_capped"])
        n = len(data["rows"])
        fields = ("numeric_match", "strict_match", "format_compliant", "unresolved", "numeric_mismatch",
                  "length_capped", "response_tokens", "proxy", "judge", "gap", "high_gap")
        scores: dict[str, Any] = {key: sum(row[key] for row in data["rows"]) / n for key in fields}
        scores.update(count=n, tail_severity=sum(max(0., row["gap"] - theta) for row in data["rows"]) / n,
                      training_rollout_kl=kl)
        actual = [row["gap"] for row in data["rows"]]
        scores["gap_prediction"] = gap_metrics(actual, predicted, theta=theta)
        scores["zero_gap"] = gap_metrics(actual, [0.] * n, theta=theta)
        atomic_write_json(folder / "rows.json", data["rows"])
        return {"seed": seed, "arm": arm, "update": update, "cohort": cohort,
                "theta": theta, "metrics": scores, "rows": str(folder / "rows.json")}

    def _reward(self, arm, calibration, memory):
        return MathReward(arm, self.proxy, self.judge, calibration, memory, self.settings)

    def _math_trainer(self, actor, reward, seed, arm, calibration):
        return PPOTrainer(actor, reward, self.config.training, experiment_id=f"{self.run_dir}/seed-{seed}",
                          reward_id=f"{arm}/{calibration.calibration_id}", seed=seed,
                          temperature=self.settings["policy_temperature"])

    def _train_math(self, folder, actor, seed, arm, calibration, memory, theta, initial):
        reward = self._reward(arm, calibration, memory)
        trainer = self._math_trainer(actor, reward, seed, arm, calibration)
        directory = self.run_dir / f"seed-{seed}" / arm
        latest, final = directory / "latest.pt", directory / "final.pt"
        batches = schedule(self.cohorts["ppo"], self.settings, seed)
        seeds = [self._seed(seed, f"ppo/{i}") for i in range(self.settings["updates"])]
        metrics_path = directory / "metrics.json"
        try:
            if final.is_file():
                trainer.load_checkpoint(final)
            elif latest.is_file():
                trainer.load_checkpoint(latest)
            else:
                trainer.fork_checkpoint(initial, expected_reward_id=f"proxy/{calibration.calibration_id}")
            history = read(metrics_path) if metrics_path.exists() else []
            history = [row for row in history if row["update"] <= trainer.update_count]
            if [row["update"] for row in history] != list(range(1, trainer.update_count + 1)):
                raise ValueError("Saved training metrics do not cover the restored checkpoint")
            if final.is_file() and trainer.update_count != self.settings["updates"]:
                raise ValueError("Final checkpoint does not contain the declared number of updates")
            def monitor():
                update = trainer.update_count
                if update and (update % self.settings["monitor_every"] == 0 or update == self.settings["updates"]):
                    kl = history[-1]["library_metrics"].get("objective/kl") if history else None
                    self._stage(f"seed-{seed}/monitor/{arm}/{update:06d}", lambda output:
                                self._evaluate_math(output, actor, seed, arm, update, "monitor", calibration, memory, theta, kl))
            monitor()
            while trainer.update_count < self.settings["updates"]:
                self._phase(f"seed-{seed}/training/{arm}")
                reward.rows.clear()
                result = trainer.train(batches, seeds, until_update=trainer.update_count + 1)
                history.extend(asdict(row) for row in result)
                atomic_write_json(metrics_path, history)
                atomic_write_json(directory / "rollouts" / f"update-{trainer.update_count:06d}.json", reward.rows)
                boundary = trainer.update_count % self.settings["monitor_every"] == 0
                if trainer.update_count == self.settings["updates"]:
                    trainer.save_checkpoint(final)
                elif boundary or trainer.update_count % self.settings["checkpoint_every"] == 0:
                    trainer.save_checkpoint(latest, replace_existing=True)
                monitor()
            latest.unlink(missing_ok=True)
            return {"checkpoint": str(final), "metrics": str(metrics_path)}
        finally:
            trainer.release()

    def run(self, *, until: Literal["round1", "training", "complete"] = "complete") -> ExperimentResult:
        if until not in ("training", "complete"):
            raise ValueError("GSM8K has no round-one fork; use --until training")
        from reward_gap.gsm8k.reporting import write_report
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.run_dir / ".run.lock"), timeout=0):
            self._open_gsm()
            if self.status["state"] == "completed":
                if not (self.run_dir / "summary.json").is_file():
                    raise ValueError("Completed run has no summary")
                return ExperimentResult(self.run_dir, self.run_dir / "status.json", self.run_dir / "summary.json", "completed")
            self.status.update(state="running")
            self._status()
            try:
                self.proxy, self.judge = self._scorer("proxy"), self._scorer("judge")
                preparations = {}
                for seed in self.config.seeds:
                    actor = self._actor(seed)
                    prepared = self._stage(f"seed-{seed}/calibration", lambda folder: self._prepare_seed(folder, actor, seed))
                    calibration = FrozenCalibration.load(prepared["calibration"])
                    built = self._stage(f"seed-{seed}/memory", lambda folder: self._build_memory(folder, actor, seed, calibration))
                    memory = QuestionMemory.from_dict(read(built["memory"]))
                    preparations[seed] = (calibration, memory, prepared["theta"])
                    self._stage(f"seed-{seed}/monitor/base/000000", lambda folder:
                                self._evaluate_math(folder, actor, seed, "base", 0, "monitor", calibration, memory, prepared["theta"], 0.))
                    initial = self.run_dir / f"seed-{seed}" / "initial.pt"
                    pending = any(self.status["stages"].get(f"seed-{seed}/train/{arm}", {}).get("state") != "completed"
                                  for arm in ("proxy", "judge", "knn"))
                    if pending and not initial.exists():
                        trainer = self._math_trainer(actor, self._reward("proxy", calibration, memory), seed, "proxy", calibration)
                        try:
                            trainer.save_checkpoint(initial)
                        finally:
                            trainer.release()
                    for arm in ("proxy", "judge", "knn"):
                        self._stage(f"seed-{seed}/train/{arm}", lambda folder, a=arm:
                                    self._train_math(folder, actor, seed, a, calibration, memory, prepared["theta"], initial))
                    initial.unlink(missing_ok=True)
                    del actor
                    gc.collect()
                if until == "complete" and self.settings["evaluate_test"]:
                    # Official test labels are produced only after every arm/seed finishes.
                    for seed in self.config.seeds:
                        calibration, memory, theta = preparations[seed]
                        actor = self._actor(seed)
                        for arm in ("base", "proxy", "judge", "knn"):
                            update, kl = 0, 0.
                            if arm != "base":
                                source = self.status["stages"][f"seed-{seed}/train/{arm}"]["result"]
                                info = load_policy_checkpoint(actor, source["checkpoint"])
                                update = info["update"]
                                kl = read(source["metrics"])[-1]["library_metrics"].get("objective/kl")
                            self._stage(f"seed-{seed}/final/{arm}", lambda folder, a=arm, u=update, k=kl:
                                        self._evaluate_math(folder, actor, seed, a, u, "final", calibration, memory, theta, k))
                        del actor
                results = [entry["result"] for name, entry in self.status["stages"].items()
                           if entry["state"] == "completed" and ("/monitor/" in name or "/final/" in name)]
                summary = {"schema_version": 1, "protocol": PROTOCOL, "parser": VERSION,
                           "data_seed": self.settings["data_seed"], "run_seeds": list(self.config.seeds),
                           "test_evaluated": until == "complete" and self.settings["evaluate_test"],
                           "test_limit": self.settings["test_limit"], "results": results}
                write_report(self.run_dir, summary)
                atomic_write_json(self.run_dir / "summary.json", summary)
                self.status.update(state="paused" if until == "training" and self.settings["evaluate_test"] else "completed", current_stage=None)
                self._status()
            except BaseException as exc:
                self.status.update(state="failed", error=str(exc))
                self._status()
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return ExperimentResult(self.run_dir, self.run_dir / "status.json", self.run_dir / "summary.json", self.status["state"])
