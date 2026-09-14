"""RQ1: frozen gap prediction and high-gap detection on unseen answers."""

import gc
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from filelock import FileLock

from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration
from reward_gap.evaluation import _collect
from reward_gap.experiment import ExperimentError, ExperimentResult, FollowupExperiment
from reward_gap.gap_prediction import RidgePredictor, metrics, select_cutoff, select_ridge
from reward_gap.memory import GapMemory, MemoryContext
from reward_gap.ppo import load_policy_checkpoint


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_plan(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = _read(path)
    allowed = {"theta", "calibration_quantile", "ridge_alphas", "answers_per_prompt", "checkpoints"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ExperimentError("Unknown RQ1 plan fields")
    plan = {"theta": None, "calibration_quantile": .95, "ridge_alphas": [.001, .01, .1, 1.],
            "answers_per_prompt": 1, "checkpoints": [], **raw}
    theta, quantile = plan["theta"], plan["calibration_quantile"]
    if theta is not None and (type(theta) not in (int, float) or not math.isfinite(theta) or theta < 0):
        raise ExperimentError("theta must be null or a finite nonnegative number")
    if type(quantile) not in (int, float) or not 0 < quantile < 1:
        raise ExperimentError("calibration_quantile must be between zero and one")
    if type(plan["answers_per_prompt"]) is not int or plan["answers_per_prompt"] < 1:
        raise ExperimentError("answers_per_prompt must be a positive integer")
    alphas = plan["ridge_alphas"]
    if (not isinstance(alphas, list) or not alphas or any(type(a) not in (int, float)
            or not math.isfinite(a) or a <= 0 for a in alphas)):
        raise ExperimentError("Provide positive finite ridge_alphas")
    root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), None)
    if root is None:
        raise ExperimentError("RQ1 plan must be within a project containing pyproject.toml")
    if not isinstance(plan["checkpoints"], list):
        raise ExperimentError("checkpoints must be a list")
    labels = {"initial"}
    for checkpoint in plan["checkpoints"]:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"label", "path"}:
            raise ExperimentError("Each checkpoint needs label and path")
        label = checkpoint["label"]
        if not isinstance(label, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", label) or label in labels:
            raise ExperimentError("Checkpoint labels must be unique simple directory names")
        labels.add(label)
        if not isinstance(checkpoint["path"], str) or not checkpoint["path"].strip():
            raise ExperimentError("Checkpoint path must be nonempty")
        checkpoint["path"] = str((root / checkpoint["path"]).resolve())
    return plan


class RQ1Experiment(FollowupExperiment):
    """Reuse stage persistence/model loading; train only two small ridge models.

    Calibration uses its dedicated cohort. All predictors fit on initial_memory,
    select settings/cutoffs on validation, and freeze before any final labels.
    Later PPO checkpoints are inference sources only; they never update memory.
    """

    def __init__(self, config, run_dir, *, plan, **factories):
        super().__init__(config, run_dir, **factories)
        if len(config.seeds) != 1:
            raise ExperimentError("RQ1 uses one generation seed per run; use separate runs for additional seeds")
        self.plan = plan

    def _check_checkpoint_splits(self):
        final_groups = {r.conversation_group for r in self.cohorts["final_evaluation"]}
        for checkpoint in self.plan["checkpoints"]:
            path = Path(checkpoint["path"])
            if not path.is_file():
                raise ExperimentError(f"Missing PPO checkpoint: {path}")
            source = next((p / "inputs.json" for p in path.parents if (p / "inputs.json").is_file()), None)
            if source is None:
                raise ExperimentError("Checkpoint needs its coordinator inputs.json to verify held-out groups")
            cohorts = _read(source)["cohorts"]
            if not {"calibration", "initial_memory", "training", "refresh", "validation", "final_evaluation"} <= set(cohorts):
                raise ExperimentError("Incomplete checkpoint cohort evidence")
            development_groups = {r["conversation_group"] for name, rows in cohorts.items()
                                  if name != "final_evaluation" for r in rows}
            if final_groups & development_groups:
                raise ExperimentError("RQ1 test groups overlap checkpoint development/training groups")

    def _answers(self, folder, actor, proxy, calibration, cohort):
        rows, vectors, context = [], [], None
        judge = self._scorer("judge")
        for sample in range(self.plan["answers_per_prompt"]):
            collected = _collect(actor, proxy, judge, calibration, self.cohorts[cohort],
                                 seed=self._seed(self.config.seeds[0], f"rq1/{cohort}/{sample}"),
                                 batch_size=self.config.scoring.batch_size, need_embeddings=True)
            if collected.embeddings is None or collected.context is None:
                raise ExperimentError("RQ1 needs proxy embeddings")
            if context is not None and context != collected.context:
                raise ExperimentError("Representation changed across samples")
            context = collected.context
            for row in collected.rows:
                row["example_id"] = f"{cohort}/sample-{sample}/{row['example_id']}"
                row["sample"] = sample
            rows.extend(collected.rows)
            vectors.extend(collected.embeddings.tolist())
        if context is None:
            raise ExperimentError("No answers were collected")
        atomic_write_json(folder / "answers.json", {"rows": rows, "vectors": vectors, "context": asdict(context)})
        return {"answers": str(folder / "answers.json"), "count": len(rows)}

    def _fit(self, folder, training, validation, calibration, calibration_rows):
        train, val = _read(training["answers"]), _read(validation["answers"])
        if train["context"] != val["context"]:
            raise ExperimentError("Training and validation representations differ")
        x, vx = np.array(train["vectors"]), np.array(val["vectors"])
        gaps = np.array([r["gap"] for r in train["rows"]])
        vgaps = np.array([r["gap"] for r in val["rows"]])
        judge = np.array([r["normalized_judge_score"] for r in train["rows"]])
        vjudge = np.array([r["normalized_judge_score"] for r in val["rows"]])
        ridge = select_ridge(x, gaps, vx, vgaps, self.plan["ridge_alphas"])
        student = select_ridge(x, judge, vx, vjudge, self.plan["ridge_alphas"])
        memory = GapMemory([r["example_id"] for r in train["rows"]], torch.tensor(x), gaps.tolist(),
                           context=MemoryContext(**train["context"]), k=self.config.memory.k,
                           temperature=self.config.memory.temperature)
        memory.save(folder / "memory.json")
        cgaps = [(r["proxy_score"] - calibration.proxy.mean) / calibration.proxy.std
                 - (r["judge_score"] - calibration.judge.mean) / calibration.judge.std for r in calibration_rows]
        theta = self.plan["theta"]
        if theta is None:
            theta = max(0., float(np.quantile(cgaps, self.plan["calibration_quantile"])))
        fitted = {"memory": str(folder / "memory.json"), "ridge_gap": ridge.to_dict(),
                  "judge_student": student.to_dict(), "mean_gap": float(gaps.mean()), "theta": theta,
                  "threshold_rule": "explicit" if self.plan["theta"] is not None else "max(0, calibration quantile)",
                  "context": train["context"], "cutoffs": {}}
        predictions = self._predictions(val, fitted, memory)
        fitted["cutoffs"] = {name: select_cutoff(vgaps, values, theta) for name, values in predictions.items()}
        fitted["validation_metrics"] = {name: metrics(vgaps, values, theta=theta, cutoff=fitted["cutoffs"][name])
                                        for name, values in predictions.items()}
        atomic_write_json(folder / "predictors.json", fitted)
        return {"predictors": str(folder / "predictors.json")}

    @staticmethod
    def _predictions(data, fitted, memory):
        if data["context"] != fitted["context"]:
            raise ExperimentError("Evaluation representation differs from frozen predictors")
        x = np.array(data["vectors"])
        return {"knn": memory.predict(torch.tensor(x), context=MemoryContext(**data["context"])).gaps.numpy(),
                "zero_gap": np.zeros(len(x)), "mean_gap": np.full(len(x), fitted["mean_gap"]),
                "ridge_gap": RidgePredictor(**fitted["ridge_gap"]).predict(x),
                "judge_student": np.array([r["normalized_proxy_score"] for r in data["rows"]])
                                 - RidgePredictor(**fitted["judge_student"]).predict(x)}

    def _test(self, folder, data, fitted, policy):
        data = _read(data["answers"])
        memory = GapMemory.load(fitted["memory"], context=MemoryContext(**fitted["context"]))
        predictions = self._predictions(data, fitted, memory)
        actual = [r["gap"] for r in data["rows"]]
        scores = {name: metrics(actual, values, theta=fitted["theta"], cutoff=fitted["cutoffs"][name])
                  for name, values in predictions.items()}
        for index, row in enumerate(data["rows"]):
            row["high_gap"] = row["gap"] > fitted["theta"]
            row["predictions"] = {name: float(values[index]) for name, values in predictions.items()}
        atomic_write_json(folder / "predictions.json", data["rows"])
        return {"policy": policy, "metrics": scores, "predictions": str(folder / "predictions.json")}

    def run(self, *, until: Literal["round1", "training", "complete"] = "complete") -> ExperimentResult:
        if until != "complete":
            raise ExperimentError("RQ1 has no PPO round boundaries; rerunning resumes its completed stages")
        # Fail on missing plotting dependencies before paying for generation.
        from reward_gap.rq1_reporting import write_report
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.run_dir / ".run.lock"), timeout=0):
            self._open()
            plan_path = self.run_dir / "rq1_plan.json"
            if plan_path.exists():
                if _read(plan_path) != self.plan:
                    raise ExperimentError("RQ1 plan changed; use a new run directory")
            else:
                if self.status["stages"]:
                    raise ExperimentError("This directory already contains another experiment")
                atomic_write_json(plan_path, self.plan)
            if not self.cohorts["validation"]:
                raise ExperimentError("RQ1 needs a nonempty validation cohort")
            self._check_checkpoint_splits()
            if self.status["state"] == "completed":
                for name in ("summary.json", "metrics.csv", "report.md", "prediction_vs_actual.png", "checkpoint_metrics.png"):
                    if not (self.run_dir / name).is_file():
                        raise ExperimentError(f"Completed RQ1 run is missing {name}")
                return ExperimentResult(self.run_dir, self.run_dir / "status.json", self.run_dir / "summary.json", "completed")
            self.status.update(state="running", kind="rq1")
            self.status.pop("error", None)
            self._status()
            try:
                proxy = self._scorer("proxy")
                actor = self._actor(self.config.seeds[0])
                cal = self._stage("calibration", lambda folder: self._calibration(folder, actor, proxy))
                calibration = FrozenCalibration.load(cal["calibration"])
                training = self._stage("predictor-training", lambda folder:
                                       self._answers(folder, actor, proxy, calibration, "initial_memory"))
                validation = self._stage("predictor-validation", lambda folder:
                                         self._answers(folder, actor, proxy, calibration, "validation"))
                fit = self._stage("fit-predictors", lambda folder:
                                  self._fit(folder, training, validation, calibration, _read(cal["rows"])))
                fitted = _read(fit["predictors"])
                policies = [{"label": "initial", "path": None}, *self.plan["checkpoints"]]
                results = {}
                for checkpoint in policies:
                    label = checkpoint["label"]
                    stage = f"evaluate-{label}"
                    if self.status["stages"].get(stage, {}).get("state") == "completed":
                        results[label] = self.status["stages"][stage]["result"]
                        continue
                    policy = {"label": label, "checkpoint": None, "update": 0}
                    if checkpoint["path"]:
                        policy = {"label": label, **load_policy_checkpoint(actor, checkpoint["path"])}
                    answers = self._stage(f"answers-{label}", lambda folder:
                                          self._answers(folder, actor, proxy, calibration, "final_evaluation"))
                    results[label] = self._stage(stage, lambda folder:
                                                 self._test(folder, answers, fitted, policy))
                summary = {"schema_version": 1, "question": "Can memory predict proxy-judge disagreement?",
                           "seed": self.config.seeds[0], "theta": fitted["theta"], "cutoffs": fitted["cutoffs"],
                           "distribution_shift_evaluated": bool(self.plan["checkpoints"]), "results": results,
                           "judge_labels": {"calibration": len(_read(cal["rows"])), "predictor_training": training["count"],
                                            "validation": validation["count"],
                                            "final_evaluation": sum(r["metrics"]["knn"]["count"] for r in results.values())},
                           "limitations": ["Single generation seed; exploratory, without between-run uncertainty.",
                                            "Gap is disagreement with this judge, not independently verified reward hacking.",
                                            "Ridge student is a linear frozen-feature model, not a fine-tuned reward LLM.",
                                            "Undefined ranking metrics are null when a required class is absent."]}
                write_report(self.run_dir, summary)
                atomic_write_json(self.run_dir / "summary.json", summary)
                self.status.update(state="completed", current_stage=None)
                self._status()
            except BaseException as exc:
                self.status.update(state="failed", error=str(exc))
                self._status()
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return ExperimentResult(self.run_dir, self.run_dir / "status.json", self.run_dir / "summary.json", "completed")
