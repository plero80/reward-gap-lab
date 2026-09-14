"""Matched 4B/30B teacher labels, shared proxy vectors, and static-memory PPO."""

import gc
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from filelock import FileLock

from reward_gap.artifacts import atomic_write_json
from reward_gap.failures import record_failure
from reward_gap.calibration import FrozenCalibration
from reward_gap.config import ModelReference
from reward_gap.experiment import ExperimentResult
from reward_gap.ppo import load_policy_checkpoint
from reward_gap.gsm8k.answers import GRADE_PARSER_VERSION, VERSION, evaluate_answer
from reward_gap.gsm8k.experiment import GSMExperiment, read
from reward_gap.gsm8k.graders import LanguageGrader
from reward_gap.gsm8k.memory import QuestionMemory
from reward_gap.gsm8k.metrics import gap_metrics
from reward_gap.gsm8k.rewards import MathReward
from reward_gap.gsm8k.recovery import FAILURE_POLICY, mean_present, score_partial
from reward_gap.memory import MemoryContext

TEACHERS = ("4b", "30b")
ARMS = ("proxy", "judge4", "knn4", "knn30")
PREPARATION = ("calibration", "memory", "selection", "monitor")


def release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TeacherReward(MathReward):
    def __init__(self, arm, proxy, calibration, memory, settings, *, judge=None):
        if arm not in ARMS:
            raise ValueError("Unknown teacher-comparison training arm")
        if arm == "judge4" and judge is None:
            raise ValueError("Judge-4B PPO requires the frozen 4B judge")
        strategy = "judge" if arm == "judge4" else "proxy" if arm == "proxy" else "knn"
        super().__init__(strategy, proxy, judge, calibration, memory, settings)
        self.condition = arm

    def score_rollouts(self, *args, **kwargs):
        start = len(self.rows)
        result = super().score_rollouts(*args, **kwargs)
        for row in self.rows[start:]:
            row.update(condition=self.condition, teacher=None if self.condition == "proxy" else {
                "source": self.calibration.judge.source, "revision": self.calibration.judge.revision})
        return result


class TeacherExperiment(GSMExperiment):
    protocol = "gsm8k_matched_teacher_labels_judge4_v2"

    def __init__(self, config, run_dir, *, teacher_factory=None, **factories):
        super().__init__(config, run_dir, **factories)
        if "teacher_comparison" not in self.settings:
            raise ValueError("This run requires teacher_comparison settings")
        self.teacher_factory = teacher_factory or self._load_teacher
        self.preparations: dict = {}

    def _load_teacher(self, name):
        config = self.gsm_config
        if name == "30b":
            base_models = self.config.models
            if base_models is None:
                raise ValueError("Teacher comparison requires model references")
            models = replace(base_models, judge=ModelReference(**self.settings["teacher_comparison"]["teacher30"]))
            config = replace(config, base=replace(self.config, models=models))
        return LanguageGrader.load(config, "judge", self.questions, self.run_dir / "graders" / name)

    def _teacher(self, name):
        teacher = self.teacher_factory(name)
        self._model(f"teacher-{name}", teacher.loaded.source, teacher.loaded.revision)
        return teacher

    def _phase(self, name):
        self.proxy.phase = name
        if hasattr(self, "training_judge"):
            self.training_judge.phase = name

    def _shared(self, actor, cohort, seed, repeats, *, greedy=False):
        """Generate once; never use a teacher to generate replacement answers."""
        return self._collect_labels(actor, cohort, seed, repeats, greedy=greedy, include_judge=False)

    def _save_shared(self, folder, actor, cohort, seed):
        self._phase(f"seed-{seed}/shared/{cohort}")
        data = self._shared(actor, cohort, seed, 1 if cohort == "monitor" else self.settings["preparation_responses"],
                            greedy=cohort == "monitor")
        path = atomic_write_json(folder / "shared.json", data)
        return {"shared": str(path)}

    def _grade_shared(self, folder, teacher, name, shared_path):
        shared = read(shared_path)
        rows = []
        identity = {"source": teacher.loaded.source, "revision": teacher.loaded.revision}
        for start in range(0, len(shared["rows"]), self.config.scoring.batch_size):
            chunk = shared["rows"][start:start + self.config.scoring.batch_size]
            prompts = [self.questions[r["question_id"]].prompt() for r in chunk]
            batches, errors = score_partial(teacher, prompts, [r["answer"] for r in chunk])
            for row, batch, error in zip(chunk, batches, errors, strict=True):
                if error:
                    rows.append({**row, "raw_judge": None, "judge_tokens": None,
                                 "teacher": name, "teacher_identity": identity, "grading_error": error})
                    record_failure(self.run_dir, phase=teacher.phase, teacher=name, error=error,
                                   **{**row, "status": "failed"})
                    continue
                if (batch.prompt_ids != (row["question_id"],) or batch.role != "judge"
                        or {"source": batch.source, "revision": batch.revision} != identity):
                    raise ValueError("Teacher changed input alignment or model identity")
                rows.append({**row, "raw_judge": batch.scores[0], "judge_tokens": batch.token_counts[0],
                             "teacher": name, "teacher_identity": identity})
        path = atomic_write_json(folder / "grades.json", {"shared": str(shared_path), "judge": identity, "rows": rows})
        return {"grades": str(path)}

    def _joined(self, seed, name, cohort):
        source = self.status["stages"][f"seed-{seed}/labels/{name}/{cohort}"]["result"]["grades"]
        grades = read(source)
        shared = read(grades["shared"])
        if len(shared["rows"]) != len(grades["rows"]):
            raise ValueError("Teacher labels do not cover the shared responses")
        for original, labeled in zip(shared["rows"], grades["rows"], strict=True):
            if any(labeled.get(key) != value for key, value in original.items()):
                raise ValueError("Teacher labels changed a shared response or proxy score")
        # Filter both teachers to the same examples, preserving vector order.
        usable = {r["example_id"] for r in grades["rows"] if r["raw_judge"] is not None}
        for other in TEACHERS:
            path = self.status["stages"][f"seed-{seed}/labels/{other}/{cohort}"]["result"]["grades"]
            other_grades = read(path)
            if other_grades["shared"] != grades["shared"]:
                raise ValueError("Teachers labeled different shared artifacts")
            usable &= {r["example_id"] for r in other_grades["rows"] if r["raw_judge"] is not None}
        selected = [i for i, r in enumerate(grades["rows"]) if r["example_id"] in usable]
        return {**shared, "rows": [grades["rows"][i] for i in selected],
                "embeddings": [shared["embeddings"][i] for i in selected], "judge": grades["judge"],
                "excluded_ids": [r["example_id"] for r in grades["rows"] if r["example_id"] not in usable]}

    def _fit_teacher(self, folder, seed, name):
        calibration_data = self._joined(seed, name, "calibration")
        calibration = FrozenCalibration.fit(self._batch(calibration_data, "proxy"), self._batch(calibration_data, "judge"),
                                            calibration_id=f"{self.run_dir.name}/seed-{seed}/{self.protocol}/{name}")
        self._normalized(calibration_data, calibration)
        theta = float(np.quantile([r["gap"] for r in calibration_data["rows"]], self.settings["gap_quantile"]))
        data = self._normalized(self._joined(seed, name, "memory"), calibration)
        context = MemoryContext(data["proxy"]["source"], data["proxy"]["revision"], data["pooling"], calibration.calibration_id)
        comparison = self.settings["teacher_comparison"]
        memory = QuestionMemory(data["rows"], torch.tensor(data["embeddings"]), context,
                                k=comparison["k"], temperature=comparison["temperature"])
        calibration.save(folder / "calibration.json")
        coverage = {}
        for cohort in PREPARATION:
            joined = self._joined(seed, name, cohort)
            coverage[cohort] = {"included_count": len(joined["rows"]), "excluded_ids": joined["excluded_ids"],
                                "failed_shared_ids": [r["example_id"] for r in joined.get("failed_rows", [])]}
        atomic_write_json(folder / "coverage.json", coverage)
        atomic_write_json(folder / "memory.json", memory.to_dict())
        diagnostics = {}
        for cohort in ("selection", "monitor"):
            held = self._normalized(self._joined(seed, name, cohort), calibration)
            if held["rows"] and held["pooling"] != context.pooling:
                raise ValueError("Held-out representation differs from memory")
            predictions, neighbors = (memory.predict(torch.tensor(held["embeddings"]),
                [r["question_id"] for r in held["rows"]], context=context) if held["rows"] else ([], []))
            for row, gap, nearby in zip(held["rows"], predictions, neighbors, strict=True):
                row.update(predicted_gap=gap, corrected_reward=row["proxy"] - gap, neighbors=nearby)
            diagnostics[cohort] = {"gap_prediction": gap_metrics([r["gap"] for r in held["rows"]], predictions, theta=theta),
                                   "zero_gap": gap_metrics([r["gap"] for r in held["rows"]], [0.] * len(predictions), theta=theta)}
            atomic_write_json(folder / f"{cohort}.json", held["rows"])
        atomic_write_json(folder / "diagnostics.json", diagnostics)
        return {"calibration": str(folder / "calibration.json"), "memory": str(folder / "memory.json"),
                "theta": theta, "diagnostics": str(folder / "diagnostics.json"), "monitor": str(folder / "monitor.json")}

    def _preparation(self):
        shared_pending = any(self.status["stages"].get(f"seed-{seed}/shared/{cohort}", {}).get("state") != "completed"
                             for seed in self.config.seeds for cohort in PREPARATION)
        if shared_pending:
            self.proxy = self._scorer("proxy")
            try:
                for seed in self.config.seeds:
                    actor = self._actor(seed)
                    for cohort in PREPARATION:
                        self._stage(f"seed-{seed}/shared/{cohort}", lambda folder, c=cohort:
                                    self._save_shared(folder, actor, c, seed))
                    del actor
            finally:
                del self.proxy
                release_models()
        # No policy/proxy reference is retained while the large teacher is loaded.
        for name in TEACHERS:
            pending = any(self.status["stages"].get(f"seed-{seed}/labels/{name}/{cohort}", {}).get("state") != "completed"
                          for seed in self.config.seeds for cohort in PREPARATION)
            if pending:
                teacher = self._teacher(name)
                try:
                    for seed in self.config.seeds:
                        for cohort in PREPARATION:
                            teacher.phase = f"seed-{seed}/preparation/{cohort}"
                            shared = self.status["stages"][f"seed-{seed}/shared/{cohort}"]["result"]["shared"]
                            self._stage(f"seed-{seed}/labels/{name}/{cohort}", lambda folder, src=shared:
                                        self._grade_shared(folder, teacher, name, src))
                finally:
                    del teacher
                    release_models()
        for seed in self.config.seeds:
            self.preparations[seed] = {}
            for name in TEACHERS:
                result = self._stage(f"seed-{seed}/fit/{name}", lambda folder, n=name: self._fit_teacher(folder, seed, n))
                self.preparations[seed][name] = (FrozenCalibration.load(result["calibration"]),
                    QuestionMemory.from_dict(read(result["memory"])), result["theta"])
            c4, m4, _ = self.preparations[seed]["4b"]
            c30, m30, _ = self.preparations[seed]["30b"]
            if (c4.proxy != c30.proxy or not torch.equal(m4.vectors, m30.vectors)
                    or [(r["example_id"], r["question_id"]) for r in m4.rows] != [(r["example_id"], r["question_id"]) for r in m30.rows]
                    or (m4.k, m4.temperature) != (m30.k, m30.temperature)):
                raise ValueError("Teacher comparison requires identical proxy scales, examples, vectors and retrieval settings")
        from reward_gap.gsm8k.teacher_reporting import preparation_report
        preparation_report(self)

    def _reward(self, arm, calibration, memory):
        return TeacherReward(arm, self.proxy, calibration, memory, self.settings,
                             judge=getattr(self, "training_judge", None) if arm == "judge4" else None)

    def _train_arm(self, folder, actor, seed, arm, calibration, memory, theta, initial):
        if arm != "judge4":
            return self._train_math(folder, actor, seed, arm, calibration, memory, theta, initial)
        self.training_judge = self._teacher("4b")
        try:
            return self._train_math(folder, actor, seed, arm, calibration, memory, theta, initial)
        finally:
            del self.training_judge
            release_models()

    def _evaluate_math(self, folder, actor, seed, arm, update, cohort, calibration, memory, theta, kl):
        self._phase(f"seed-{seed}/evaluation/{cohort}/{arm}/{update}")
        data = self._shared(actor, cohort, seed, 1, greedy=True)
        proxy = calibration.normalize_proxy(self._batch(data, "proxy")) if data["rows"] else []
        for row, score in zip(data["rows"], proxy, strict=True):
            row.update(proxy=score, predicted_gaps={}, corrected_rewards={})
        for teacher, (cal, mem, _) in self.preparations[seed].items():
            context = (MemoryContext(data["proxy"]["source"], data["proxy"]["revision"], data["pooling"], cal.calibration_id)
                       if data["rows"] else None)
            gaps, neighbors = (mem.predict(torch.tensor(data["embeddings"]), [r["question_id"] for r in data["rows"]], context=context)
                               if data["rows"] else ([], []))
            for row, gap, nearby in zip(data["rows"], gaps, neighbors, strict=True):
                row["predicted_gaps"][teacher] = gap
                row["corrected_rewards"][teacher] = row["proxy"] - gap
                row.setdefault("neighbors", {})[teacher] = nearby
        for row in data["rows"]:
            row.update(format_penalty=self.settings["format_penalty"] * (not row["format_compliant"]),
                       length_penalty=self.settings["length_penalty"] * row["length_capped"])
        valid_count = len(data["rows"])
        data["rows"] += data.get("failed_rows", [])
        fields = ("numeric_match", "strict_match", "format_compliant", "unresolved", "numeric_mismatch",
                  "length_capped", "response_tokens", "proxy")
        scores: dict[str, Any] = {key: mean_present(data["rows"], key) for key in fields}
        scores.update(count=len(data["rows"]), graded_count=valid_count, failed_count=len(data["rows"]) - valid_count,
                      training_rollout_kl=kl)
        atomic_write_json(folder / "rows.json", data["rows"])
        return {"seed": seed, "arm": arm, "update": update, "cohort": cohort, "metrics": scores,
                "rows": str(folder / "rows.json"), "teacher_grades": "not_computed"}

    def run(self, *, until="complete"):
        if until not in ("preparation", "training", "complete"):
            raise ValueError("Use preparation, training, or complete for the teacher comparison")
        from reward_gap.gsm8k.teacher_reporting import write_report
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.run_dir / ".run.lock"), timeout=0):
            self._open_gsm()
            if self.status["state"] == "completed":
                required = [self.run_dir / "summary.json", self.run_dir / "report.md"]
                required += [self.run_dir / f"seed-{s}" / a / "final.pt" for s in self.config.seeds for a in ARMS]
                if any(not p.is_file() for p in required):
                    raise ValueError("Completed teacher comparison is missing artifacts")
                return ExperimentResult(self.run_dir, self.run_dir / "status.json", self.run_dir / "summary.json", "completed")
            self.status.update(state="running", kind="gsm8k_teacher_comparison")
            self.status.pop("error", None)
            self._status()
            try:
                self._preparation()
                if until != "preparation":
                    self.proxy = self._scorer("proxy")
                    for seed in self.config.seeds:
                        actor = self._actor(seed)
                        c4, m4, t4 = self.preparations[seed]["4b"]
                        self._stage(f"seed-{seed}/monitor/base/000000", lambda folder:
                            self._evaluate_math(folder, actor, seed, "base", 0, "monitor", c4, m4, t4, 0.))
                        initials = {}
                        # Distinct teacher identities, identical initial trainable weights.
                        for teacher, (cal, mem, _) in self.preparations[seed].items():
                            initial = self.run_dir / f"seed-{seed}" / f"initial-{teacher}.pt"
                            initials[teacher] = initial
                            if not initial.exists():
                                trainer = self._math_trainer(actor, self._reward("proxy", cal, mem), seed, "proxy", cal)
                                try:
                                    trainer.save_checkpoint(initial)
                                finally:
                                    trainer.release()
                        for arm in ARMS:
                            name = "30b" if arm == "knn30" else "4b"
                            cal, mem, theta = self.preparations[seed][name]
                            self._stage(f"seed-{seed}/train/{arm}", lambda folder, a=arm, c=cal, m=mem, t=theta, n=name:
                                        self._train_arm(folder, actor, seed, a, c, m, t, initials[n]))
                        for path in initials.values():
                            path.unlink(missing_ok=True)
                        del actor
                        release_models()
                    if until == "complete" and self.settings["evaluate_test"]:
                        for seed in self.config.seeds:
                            actor = self._actor(seed)
                            cal, mem, theta = self.preparations[seed]["4b"]
                            for arm in ("base", *ARMS):
                                update, kl = 0, 0.
                                if arm != "base":
                                    source = self.status["stages"][f"seed-{seed}/train/{arm}"]["result"]
                                    update = load_policy_checkpoint(actor, source["checkpoint"])["update"]
                                    kl = read(source["metrics"])[-1]["library_metrics"].get("objective/kl")
                                self._stage(f"seed-{seed}/final/{arm}", lambda folder, a=arm, u=update, k=kl:
                                            self._evaluate_math(folder, actor, seed, a, u, "final", cal, mem, theta, k))
                            del actor
                results = [r["result"] for name, r in self.status["stages"].items()
                           if r["state"] == "completed" and ("/monitor/" in name or "/final/" in name)]
                summary = {"protocol": self.protocol, "parser": VERSION, "grade_parser": GRADE_PARSER_VERSION,
                           "failure_policy": FAILURE_POLICY,
                           "training": {name: entry["result"] for name, entry in self.status["stages"].items()
                                        if "/train/" in name and entry["state"] == "completed"},
                           "run_seeds": list(self.config.seeds),
                           "arms": ["base", *ARMS],
                           "data_seed": self.settings["data_seed"], "results": results,
                           "teachers": {n: self.status["models"][f"teacher-{n}"] for n in TEACHERS},
                           "retrieval": self.settings["teacher_comparison"], "test_limit": self.settings["test_limit"],
                           "test_evaluated": until == "complete" and self.settings["evaluate_test"]}
                write_report(self.run_dir, summary)
                atomic_write_json(self.run_dir / "summary.json", summary)
                state = "paused" if until == "preparation" or (until == "training" and self.settings["evaluate_test"]) else "completed"
                self.status.update(state=state, current_stage=None)
                self._status()
            except BaseException as exc:
                self.status.update(state="failed", error=str(exc))
                self._status()
                raise
            finally:
                if hasattr(self, "proxy"):
                    del self.proxy
                release_models()
        return ExperimentResult(self.run_dir, self.run_dir / "status.json", self.run_dir / "summary.json", self.status["state"])
