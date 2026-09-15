"""Export RQ1 tables and figures from saved predictions, without model loading."""

import csv
import json
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


def write_report(folder, summary):
    results = summary["results"]
    rows = [{"policy": label, "update": result["policy"]["update"], "predictor": name, **values}
            for label, result in results.items() for name, values in result["metrics"].items()]
    with (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = ("policy", "predictor", "count", "mae", "rmse", "r2", "auroc", "average_precision",
              "precision", "recall", "positive_prevalence")
    lines = ["# RQ1: Can memory predict disagreement?", "",
             f"Fixed actual-gap threshold: **{summary['theta']:.6g}**; positive means `gap > theta`.",
             "Detector cutoffs use initial-policy validation F1 and remain fixed across checkpoints.", "",
             "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    def value(item):
        return "undefined" if item is None else f"{item:.4f}" if isinstance(item, float) else str(item)
    lines.extend("| " + " | ".join(value(row[field]) for field in fields) + " |" for row in rows)
    lines.extend(["", "![Predicted versus actual gap](prediction_vs_actual.png)", "",
                  "![Checkpoint metrics](checkpoint_metrics.png)", "", "## Interpretation", "",
                  "AUROC/AP measure ranking; MAE/RMSE measure numerical correction accuracy.",
                  "AP should be interpreted alongside positive prevalence. An all-equal predictor has",
                  "AUROC 0.5 when both classes exist. No positive validation labels yields an",
                  "explicit never-positive detector. Precision is reported as 0 for no detections.", ""])
    lines.extend(["A constant baseline whose cutoff equals its score flags every answer: recall 1",
                  "then means no positives were missed, while precision equals positive prevalence.",
                  "This is not useful discrimination; the constant baseline's AUROC is 0.5.", ""])
    if not summary["distribution_shift_evaluated"]:
        lines.append("Only the initial policy was evaluated. PPO distribution shift has not been tested.")
    training = summary.get("ppo_training", {})
    if training.get("performed"):
        lines.extend(["", f"Proxy and corrected PPO each ran **{training['updates_per_arm']} updates**, starting from the same initial weights.",
                      f"Evaluation updates: `{training['evaluation_updates']}`; the final update is always included.",
                      "Both branches use the same prepared training schedule. The original memory, calibration,",
                      "five predictors and detector thresholds stay frozen. Held-out results do not control training.",
                      "Only the two final policy checkpoints are retained; intermediate answers and predictions remain saved."])
        if training["updates_per_arm"] < 25:
            lines.append("This short PPO run checks the pipeline; it cannot establish robustness to sustained optimization.")
    lines.extend(["", *[f"- {note}" for note in summary["limitations"]], "",
                  "Judge labels (shared by all predictors): `" + json.dumps(summary["judge_labels"]) + "`."])
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    methods = list(next(iter(results.values()))["metrics"])
    fig = Figure(figsize=(15, 3.4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(methods), squeeze=False)[0]
    for ax, method in zip(axes, methods, strict=True):
        low, high = 0., 0.
        for label, result in results.items():
            predictions = json.loads(Path(result["predictions"]).read_text(encoding="utf-8"))
            actual = np.array([row["gap"] for row in predictions])
            predicted = np.array([row["predictions"][method] for row in predictions])
            low = min(low, float(actual.min()), float(predicted.min()))
            high = max(high, float(actual.max()), float(predicted.max()))
            ax.scatter(actual, predicted, s=9, alpha=.35, label=label)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set(title=method, xlabel="Actual gap", ylabel="Predicted gap")
    axes[-1].legend(fontsize=7)
    fig.savefig(folder / "prediction_vs_actual.png", dpi=180)
    fig.savefig(folder / "prediction_vs_actual.pdf")

    fig = Figure(figsize=(12, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3, squeeze=False)[0]
    labels = list(results)
    for ax, metric in zip(axes, ("rmse", "auroc", "average_precision"), strict=True):
        for index, method in enumerate(methods):
            values = [results[label]["metrics"][method][metric] for label in labels]
            ax.scatter(np.arange(len(labels)) + (index - 2) * .07,
                       [np.nan if v is None else v for v in values], label=method, s=28)
        if metric == "average_precision":
            ax.scatter(np.arange(len(labels)), [results[label]["metrics"]["knn"]["positive_prevalence"]
                                                for label in labels], marker="_", color="black", label="prevalence")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        ax.set(title=metric, xlabel="Policy checkpoint")
        if metric != "rmse":
            ax.set_ylim(-.03, 1.03)
    axes[-1].legend(fontsize=7)
    fig.savefig(folder / "checkpoint_metrics.png", dpi=180)
    fig.savefig(folder / "checkpoint_metrics.pdf")
