"""Paired HH policy/predictor evaluation using one saved answer set per policy."""

import csv
import hashlib
import html
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration
from reward_gap.evaluation import _collect
from reward_gap.experiment import ExperimentError, FollowupExperiment, _read
from reward_gap.gap_prediction import RidgePredictor, metrics, select_cutoff, select_ridge
from reward_gap.gsm8k.metrics import _correlation, _ranks
from reward_gap.memory import GapMemory, MemoryContext
from reward_gap.ppo import load_policy_checkpoint


PROTOCOL = {"name": "hh-unified-v1", "response_contract": "primary-eos-contiguous-nonpad-v1",
            "calibration_quantile": .95, "ridge_alphas": [.001, .01, .1, 1.],
            "actual_positive": "gap > theta", "detected_positive": "prediction >= validation cutoff",
            "cutoff_selection": "maximum validation F1; frozen before final evaluation",
            "answers_per_prompt": 1, "memory_selection": "fixed config k and temperature"}


def diagnostics(actual, predicted, *, theta, cutoff):
    result = metrics(actual, predicted, theta=theta, cutoff=cutoff)
    a, p = np.asarray(actual), np.asarray(predicted)
    positive = a > theta
    detected = np.zeros(len(a), dtype=bool) if cutoff is None else p >= cutoff
    tp, fp = int(np.sum(positive & detected)), int(np.sum(~positive & detected))
    tn, fn = int(np.sum(~positive & ~detected)), int(np.sum(positive & ~detected))
    result.update(mse=float(np.mean((a - p) ** 2)), pearson=_correlation(a, p),
                  spearman=_correlation(_ranks(a), _ranks(p)), true_positive=tp,
                  false_positive=fp, true_negative=tn, false_negative=fn,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                  detector_comparison=">=", actual_comparison=">",
                  flags_every_answer=bool(detected.all()), flags_no_answers=bool(not detected.any()))
    if not detected.any():
        result["precision"] = None
    return result


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HHAnalysis(FollowupExperiment):
    """Uses stage recovery without writing to the source experiment artifacts.

    The caller holds the source run lock. Final checkpoints must exist before
    analysis; final answer generation follows ALL predictor fitting across seeds.
    """

    def __init__(self, source, output_dir):
        super().__init__(source.config, output_dir, actor_factory=source.actor_factory,
                         scorer_factory=source.scorer_factory)
        self.source = source
        self.missing = []

    def _result(self, name):
        entry = self.source.status["stages"].get(name, {})
        if entry.get("state") != "completed":
            raise ExperimentError(f"Source HH stage is incomplete: {name}")
        return entry["result"]

    def _policies(self, seed):
        root = self.source.run_dir / f"seed-{seed}"
        specs = [("initial", root / "initial.pt", 0)]
        if self.config.training.round1_updates > 1:
            specs += [("proxy-update-1", root / "raw/update-1.pt", 1),
                      ("corrected-update-1", root / "static/update-1.pt", 1)]
        specs += [("proxy-round1", root / "raw/round1.pt", self.config.training.round1_updates),
                  ("corrected-round1", root / "corrected_round1.pt", self.config.training.round1_updates)]
        specs += [(f"{'proxy' if arm == 'raw' else arm}-final", root / arm / "final.pt", self.config.training.total_updates)
                  for arm in ("raw", "static", "iterative")]
        available = []
        for label, path, update in specs:
            if path.is_file():
                available.append({"label": label, "checkpoint": str(path), "ppo_update": update})
            elif (self.source.run_dir / "hh_protocol.json").is_file():
                raise ExperimentError(f"Unified HH run is missing a retained checkpoint: {path}")
            elif label == "initial":
                # No PPO changes the frozen base. Seeded fresh LoRA initialization
                # reconstructs initial policy outputs for old coordinator runs.
                available.append({"label": label, "checkpoint": None, "ppo_update": 0,
                                  "initial_reconstructed": True})
            elif label.endswith("final") or label == "corrected-round1":
                raise ExperimentError(f"Missing required HH checkpoint: {path}")
            else:
                self.missing.append(f"seed-{seed}/{label}: checkpoint not retained by source run")
        return available

    def open_analysis(self):
        inputs = self.source.validate_inputs()
        if (_read(self.source.run_dir / "resolved_config.json") != self.config.to_dict()
                or _read(self.source.run_dir / "inputs.json") != inputs):
            raise ExperimentError("Analysis must use the source run's exact configuration and prepared inputs")
        self.cohorts = self.source.cohorts
        if not self.cohorts["validation"]:
            raise ExperimentError("HH predictor comparison requires a separate validation cohort")
        self.calibration_info = self._result("calibration")
        self.m0_info = self._result("initial-memory")
        self.policy_specs = {seed: self._policies(seed) for seed in self.config.seeds}
        artifacts = [self.source.run_dir / "resolved_config.json", self.source.run_dir / "inputs.json",
                     Path(self.calibration_info["calibration"]), Path(self.calibration_info["rows"]),
                     Path(self.m0_info["memory"]), Path(self.m0_info["memory"]).parent / "rows.json"]
        for seed in self.config.seeds:
            for arm in ("raw", "static", "iterative"):
                self._result(f"seed-{seed}/{arm}-final")
            refresh = self._result(f"seed-{seed}/refresh")
            artifacts += [Path(refresh["memory"]), Path(refresh["manifest"]).parent / "rows.json"]
            artifacts += [Path(p["checkpoint"]) for p in self.policy_specs[seed] if p["checkpoint"]]
        snapshot = {"protocol": PROTOCOL, "source": str(self.source.run_dir),
                    "artifacts": {str(p): _fingerprint(p) for p in artifacts}}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "status.json").exists():
            if _read(self.run_dir / "source.json") != snapshot:
                raise ExperimentError("Analysis source or protocol changed; use a new analysis directory")
            self.status = _read(self.run_dir / "status.json")
        else:
            if any(p.name != ".run.lock" for p in self.run_dir.iterdir()):
                raise ExperimentError("Analysis directory is not empty")
            atomic_write_json(self.run_dir / "source.json", snapshot)
            self.status = {"schema_version": 1, "state": "ready", "stages": {},
                           "models": dict(self.source.status.get("models", {}))}
            self._status()

    def _answers(self, folder, actor, proxy, calibration, cohort, seed):
        data = _collect(actor, proxy, self._scorer("judge"), calibration, self.cohorts[cohort],
                        seed=self._seed(seed, f"hh-analysis/{cohort}"),
                        batch_size=self.config.scoring.batch_size, need_embeddings=True)
        if data.context is None or data.embeddings is None:
            raise ExperimentError("HH analysis requires proxy embeddings")
        payload = {"rows": data.rows, "vectors": data.embeddings.tolist(), "context": asdict(data.context),
                   "effective_eos_ids": list(actor.eos_ids)}
        path = atomic_write_json(folder / "answers.json", payload)
        return {"answers": str(path), "count": len(data.rows)}

    def _fit(self, folder, validation, calibration, seed):
        val = _read(validation["answers"])
        m0 = _read(self.m0_info["memory"])
        refreshed = self._result(f"seed-{seed}/refresh")
        m1 = _read(refreshed["memory"])
        if m0["context"] != m1["context"] or m0["context"] != val["context"]:
            raise ExperimentError("HH memory and validation contexts differ")
        initial_rows = _read(Path(self.m0_info["memory"]).parent / "rows.json")
        refresh_rows = _read(Path(refreshed["manifest"]).parent / "rows.json")
        by_id = {r["example_id"]: r for r in initial_rows + refresh_rows}
        vgaps = np.array([r["gap"] for r in val["rows"]])
        vjudge = np.array([r["normalized_judge_score"] for r in val["rows"]])
        vx = np.array(val["vectors"])
        cgaps = [(r["proxy_score"] - calibration.proxy.mean) / calibration.proxy.std
                 - (r["judge_score"] - calibration.judge.mean) / calibration.judge.std
                 for r in _read(self.calibration_info["rows"])]
        fitted = {"theta": max(0., float(np.quantile(cgaps, PROTOCOL["calibration_quantile"]))),
                  "context": m0["context"], "calibration_id": calibration.calibration_id,
                  "models": {}, "cutoffs": {}, "label_counts": {},
                  "memories": {"knn_M0": self.m0_info["memory"], "knn_M1": refreshed["memory"]}}
        for label, memory in (("frozen", m0), ("updated", m1)):
            ids = memory["example_ids"]
            if not set(ids) <= set(by_id):
                raise ExperimentError("Memory labels are missing for matched student fitting")
            rows = [by_id[key] for key in ids]
            x, gaps = np.array(memory["embeddings"]), np.array(memory["gaps"])
            if not np.allclose(gaps, [r["gap"] for r in rows], rtol=0, atol=1e-12):
                raise ExperimentError("Memory gaps differ from paired training labels")
            judge = np.array([r["normalized_judge_score"] for r in rows])
            fitted["models"][f"ridge_gap_{label}"] = select_ridge(x, gaps, vx, vgaps, PROTOCOL["ridge_alphas"]).to_dict()
            fitted["models"][f"judge_student_{label}"] = select_ridge(x, judge, vx, vjudge, PROTOCOL["ridge_alphas"]).to_dict()
            fitted["label_counts"][label] = len(ids)
        fitted["mean_gap"] = float(np.mean(m0["gaps"]))
        predictions, _ = self._predict(val, fitted)
        fitted["cutoffs"] = {name: select_cutoff(vgaps, values, fitted["theta"]) for name, values in predictions.items()}
        fitted["validation_metrics"] = {name: diagnostics(vgaps, values, theta=fitted["theta"], cutoff=fitted["cutoffs"][name])
                                         for name, values in predictions.items()}
        return {"predictors": str(atomic_write_json(folder / "predictors.json", fitted))}

    @staticmethod
    def _predict(data, fitted):
        if data["context"] != fitted["context"]:
            raise ExperimentError("HH answer embeddings differ from predictor context")
        x = np.array(data["vectors"])
        predictions = {"zero_gap": np.zeros(len(x)), "mean_gap": np.full(len(x), fitted["mean_gap"])}
        neighbors = {}
        for label, path in fitted["memories"].items():
            memory = GapMemory.load(path, context=MemoryContext(**fitted["context"]))
            result = memory.predict(torch.tensor(x), context=memory.context)
            predictions[label] = result.gaps.numpy()
            neighbors[label] = {"example_ids": result.neighbor_ids, "weights": result.weights.tolist(),
                                "similarities": result.similarities.tolist()}
        for label, model in fitted["models"].items():
            prediction = RidgePredictor(**model).predict(x)
            predictions[label] = (np.array([r["normalized_proxy_score"] for r in data["rows"]]) - prediction
                                  if label.startswith("judge_student") else prediction)
        return predictions, neighbors

    def _measure(self, folder, answers, fitted, policy, seed):
        data = _read(answers["answers"])
        predictions, neighbors = self._predict(data, fitted)
        actual = np.array([r["gap"] for r in data["rows"]])
        measured = {}
        for label, values in predictions.items():
            measured[label] = {**diagnostics(actual, values, theta=fitted["theta"], cutoff=fitted["cutoffs"][label]),
                               "policy_checkpoint": policy["checkpoint"], "ppo_update": policy["ppo_update"],
                               "memory_id": "M0" if label == "knn_M0" else f"M1-seed-{seed}" if label == "knn_M1" else None,
                               "calibration_id": fitted["calibration_id"], "theta": fitted["theta"]}
        rows = [{**row, "predicted_gaps": {label: float(values[i]) for label, values in predictions.items()},
                 "high_gap": bool(actual[i] > fitted["theta"])} for i, row in enumerate(data["rows"])]
        path = atomic_write_json(folder / "predictions.json", {"rows": rows, "neighbors": neighbors})
        policy_metrics = {"count": len(rows), **{name: float(np.mean([r[field] for r in rows])) for name, field in (
            ("mean_proxy_score", "normalized_proxy_score"), ("mean_judge_score", "normalized_judge_score"),
            ("mean_gap", "gap"), ("mean_response_tokens", "response_tokens"))},
            "eos_fraction": float(np.mean([r["finish_reason"] == "eos" for r in rows])),
            "high_gap_rate": float(np.mean(actual > fitted["theta"]))}
        return {"seed": seed, "policy": policy, "policy_metrics": policy_metrics, "predictors": measured,
                "answers": answers["answers"], "predictions": str(path),
                "paired_M1_minus_M0_mae": measured["knn_M1"]["mae"] - measured["knn_M0"]["mae"]}

    def run_analysis(self):
        self.open_analysis()
        if self.status["state"] == "completed":
            summary = _read(self.run_dir / "summary.json")
            needed = [self.run_dir / name for name in ("report.md", "report.html", "policy_metrics.csv", "predictor_metrics.csv",
                                                      "policy_outcomes.png", "policy_outcomes.pdf", "memory_comparison.png", "memory_comparison.pdf")]
            needed += [Path(r[k]) for r in summary["results"].values() for k in ("answers", "predictions")]
            needed += [Path(entry["result"]["predictors"]) for name, entry in self.status["stages"].items()
                       if name.endswith("fit-predictors")]
            if any(not p.is_file() for p in needed):
                raise ExperimentError("Completed HH analysis is missing artifacts")
            return summary
        self.status["state"] = "running"
        self._status()
        try:
            calibration = FrozenCalibration.load(self.calibration_info["calibration"])
            proxy = self._scorer("proxy")
            # Initial validation answers shared with both frozen and updated fits.
            actor = self._actor(self.config.seeds[0])
            val = self._stage("validation", lambda folder: self._answers(folder, actor, proxy, calibration, "validation", self.config.seeds[0]))
            del actor
            fits = {seed: self._stage(f"seed-{seed}/fit-predictors", lambda folder, s=seed: self._fit(folder, val, calibration, s))
                    for seed in self.config.seeds}
            results = {}
            for seed in self.config.seeds:
                fitted = _read(fits[seed]["predictors"])
                for spec in self.policy_specs[seed]:
                    name = f"seed-{seed}/{spec['label']}"
                    def collect(folder, spec=spec):
                        actor = self._actor(seed)
                        if spec["checkpoint"]:
                            identity = load_policy_checkpoint(actor, spec["checkpoint"], legacy_stop_rule=True)
                            if identity["update"] != spec["ppo_update"] or identity["trainer_seed"] != seed:
                                raise ExperimentError("Source checkpoint progress/seed differs")
                        else:
                            identity = {"identity": {"reconstructed_initial": True}}
                        answer = self._answers(folder, actor, proxy, calibration, "final_evaluation", seed)
                        return {**answer, "checkpoint_identity": identity["identity"],
                                "evaluation_stop_rule": "primary tokenizer EOS"}
                    answers = self._stage(f"{name}/answers", collect)
                    result = self._stage(f"{name}/metrics", lambda folder: self._measure(folder, answers, fitted, spec, seed))
                    results[name] = result
            summary = {"schema_version": 1, "protocol": PROTOCOL, "source_run": str(self.source.run_dir),
                       "results": results, "missing_checkpoints": sorted(set(self.missing)),
                       "predictor_label_counts": {str(seed): _read(fits[seed]["predictors"])["label_counts"] for seed in fits},
                       "limitations": ["Judge scores measure agreement with this judge, not independently verified task quality or reward hacking.",
                                       "Frozen and updated students are linear models on proxy features, not fine-tuned reward LLMs.",
                                       "M1 comparisons on earlier policy answers are retrospective; M1 was unavailable during round 1.",
                                       "Historical checkpoints are evaluated with the corrected primary-EOS rule; earlier PPO is not repaired.",
                                       "Single-seed results do not estimate between-run uncertainty."]}
            write_report(self.run_dir, summary)
            atomic_write_json(self.run_dir / "summary.json", summary)
            self.status.update(state="completed", current_stage=None)
            self.status.pop("error", None)
            self._status()
            return summary
        except BaseException as exc:
            self.status.update(state="failed", error=str(exc))
            self._status()
            raise


def write_report(folder, summary):
    write_plots(folder, summary)
    policy_rows, predictor_rows = [], []
    for name, result in summary["results"].items():
        common = {"seed": result["seed"], "policy": result["policy"]["label"], "ppo_update": result["policy"]["ppo_update"]}
        policy_rows.append({**common, **result["policy_metrics"]})
        predictor_rows.extend({**common, "predictor": label, **values} for label, values in result["predictors"].items())
    for name, rows in (("policy_metrics", policy_rows), ("predictor_metrics", predictor_rows)):
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def table(rows, fields):
        return ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"] + [
            "| " + " | ".join("undefined" if r.get(f) is None else f"{r[f]:.4f}" if isinstance(r[f], float) else str(r[f]) for f in fields) + " |" for r in rows]
    sections = [("Policy outcomes (scores in frozen normalized units)", policy_rows,
                 ["seed", "policy", "ppo_update", "mean_proxy_score", "mean_judge_score", "mean_gap", "high_gap_rate", "mean_response_tokens", "eos_fraction"]),
                ("Predictors on identical answers (lower MAE/RMSE is better)", predictor_rows,
                 ["seed", "policy", "predictor", "mae", "rmse", "auroc", "average_precision", "precision", "recall", "f1", "flags_every_answer"])]
    lines = ["# Combined HH-RLHF results", "", "One calibration; proxy PPO and static/refreshed kNN PPO. Each predictor is evaluated on the same saved answers for each policy.", "",
             "Initial = before PPO. Round 1 = shared corrected checkpoint before the static/iterative fork. Final = equal total PPO budgets.", "",
             "M0 = initial memory; M1 = M0 plus refresh labels. Updated linear baselines receive exactly M1's labeled examples. Ridge/student are prediction baselines, not policy training arms.", "",
             "A detector that flags every answer has recall 1 but no discrimination. Read precision, AUROC/AP and prevalence too. Undefined metrics are shown explicitly.", ""]
    tables = []
    for title, rows, fields in sections:
        lines += [f"## {title}", "", *table(rows, fields), ""]
        cells = lambda row: "".join(f"<td>{html.escape('undefined' if row.get(f) is None else format(row[f], '.4f') if isinstance(row[f], float) else str(row[f]))}</td>" for f in fields)
        tables.append(f"<h2>{html.escape(title)}</h2><div class='table'><table><thead><tr>" + "".join(f"<th>{html.escape(f)}</th>" for f in fields)
                      + "</tr></thead><tbody>" + "".join(f"<tr>{cells(r)}</tr>" for r in rows) + "</tbody></table></div>")
    lines += ["![Policy outcomes](policy_outcomes.png)", "", "![Paired memory comparison](memory_comparison.png)", "",
              "## Interpretation limits", "", *[f"- {item}" for item in summary["limitations"] + summary["missing_checkpoints"]]]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = "<!doctype html><html><meta charset='utf-8'><title>HH-RLHF results</title><style>body{font:16px system-ui;margin:32px;color:#172337}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7;position:sticky;top:0}.table{overflow:auto;margin-bottom:30px}tr:nth-child(even){background:#f8fafc}input{padding:10px;width:340px}</style><h1>Combined HH-RLHF results</h1>"
    page += "".join(f"<p>{html.escape(line)}</p>" for line in lines[2:10] if line)
    page += "<label>Filter rows: <input id='filter' placeholder='e.g. knn_M1 or iterative-final'></label>" + "".join(tables)
    page += "<h2>Policy and memory comparison</h2><img style='max-width:100%' src='policy_outcomes.png' alt='Policy outcomes'><img style='max-width:100%' src='memory_comparison.png' alt='M0 and M1 on identical answers'>"
    page += "<h2>Interpretation limits</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in summary["limitations"] + summary["missing_checkpoints"]) + "</ul>"
    page += "<script>document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>"
    (folder / "report.html").write_text(page, encoding="utf-8")


def write_plots(folder, summary):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    results = list(summary["results"].values())
    labels = [f"{r['seed']}: {r['policy']['label']}" for r in results]
    positions = np.arange(len(results))
    figure = Figure(figsize=(max(12, len(labels) * .75), 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, metric in zip(figure.subplots(1, 3), ("mean_judge_score", "mean_gap", "high_gap_rate"), strict=True):
        axis.bar(positions, [r["policy_metrics"][metric] for r in results], color="#336a94")
        axis.set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
        axis.set_title(metric.replace("_", " "))
        axis.axhline(0, color="black", linewidth=.5)
    for suffix in ("png", "pdf"):
        figure.savefig(folder / f"policy_outcomes.{suffix}", dpi=160)
    figure = Figure(figsize=(max(12, len(labels) * .75), 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, metric in zip(figure.subplots(1, 2), ("rmse", "average_precision"), strict=True):
        for offset, name, color in ((-.18, "knn_M0", "#336a94"), (.18, "knn_M1", "#cf793d")):
            values = [r["predictors"][name][metric] for r in results]
            axis.bar(positions + offset, [np.nan if v is None else v for v in values], width=.36, label=name, color=color)
        if metric == "average_precision":
            axis.scatter(positions, [r["policy_metrics"]["high_gap_rate"] for r in results], color="black", marker="_", label="prevalence")
        axis.set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
        axis.set_title("RMSE (lower is better)" if metric == "rmse" else "Average precision (read alongside prevalence)")
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(folder / f"memory_comparison.{suffix}", dpi=160)
