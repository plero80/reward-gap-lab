"""Numeric, format, grader and PPO-KL results for every GSM8K seed."""

import csv
import json
from collections import defaultdict

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from reward_gap.gsm8k.recovery import aggregate_present


def write_report(folder, summary):
    rows = [{"seed": r["seed"], "arm": r["arm"], "cohort": r["cohort"], "update": r["update"],
             **{k: v for k, v in r["metrics"].items() if not isinstance(v, dict)},
             **{f"{method}_{k}": v for method in ("gap_prediction", "zero_gap")
                for k, v in r["metrics"][method].items()}} for r in summary["results"]]
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = ("numeric_match", "strict_match", "format_compliant", "unresolved", "length_capped", "proxy", "judge", "high_gap", "tail_severity")
    finals = [r for r in rows if r["cohort"] == "final"]
    aggregates = {}
    for arm in ("base", "proxy", "judge", "knn"):
        arm_rows = [r for r in finals if r["arm"] == arm]
        if arm_rows:
            aggregates[arm] = {field: aggregate_present(arm_rows, field) for field in fields}
    paired = []
    for seed in summary["run_seeds"]:
        arms = {r["arm"]: r for r in finals if r["seed"] == seed}
        if "knn" in arms and "proxy" in arms:
            paired.append({"seed": seed, "numeric_match_difference": arms["knn"]["numeric_match"] - arms["proxy"]["numeric_match"]})
    summary["aggregate_final"] = aggregates
    summary["paired_knn_minus_proxy"] = paired
    summary["paired_difference_mean"] = float(np.mean([r["numeric_match_difference"] for r in paired])) if paired else None
    summary["paired_difference_sample_std"] = float(np.std([r["numeric_match_difference"] for r in paired], ddof=1)) if len(paired) > 1 else None
    costs = defaultdict(lambda: {"events": 0, "generation_attempts": 0, "embedding_forwards": 0,
                                "cache_hits": 0, "input_tokens": 0, "generated_tokens": 0, "seconds": 0., "invalid_attempts": 0})
    log = folder / "grading_cost.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            bucket = costs[f"{event['phase']}/{event['role']}"]
            bucket["events"] += 1
            bucket["generation_attempts"] += "attempt" in event
            bucket["embedding_forwards"] += bool(event.get("embedding_only"))
            bucket["cache_hits"] += bool(event.get("cache_hit"))
            bucket["invalid_attempts"] += event.get("valid_grade") is False
            for key in ("input_tokens", "generated_tokens", "seconds"):
                bucket[key] += event[key]
    summary["grading_cost"] = dict(costs)
    lines = ["# GSM8K Experiment 2", "", f"Protocol: `{summary['protocol']}`; numeric checker: `{summary['parser']}`.",
             "Numeric-match rate is the primary endpoint. All responses, including unresolved cases, remain in the denominator.",
             "Missing grades are excluded only from grader/gap statistics; graded_count and failed_count give coverage.",
             "Update numbers count scheduled PPO batches. summary.json reports optimized_batches and skipped_batches per arm.",
             "Numeric matching does not certify reasoning. This new checker is not the historical post-hoc checker.", "",
             "| Seed | Arm | Cohort | Update | Numeric | Strict | Format | Unresolved | Truncated | Proxy z | Judge z | High gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        values = [str(r[k]) for k in ("seed", "arm", "cohort", "update")]
        values.extend(f"{r[k]:.4f}" if r[k] is not None else "N/A" for k in ("numeric_match", "strict_match", "format_compliant", "unresolved", "length_capped", "proxy", "judge", "high_gap"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "![Development trajectories](monitor.png)", "",
                  "Per-seed results, final means/sample standard deviations, paired kNN-minus-proxy differences,",
                  "and grading costs are also saved in summary.json. Single-seed standard deviations are undefined.",
                  "Training KL is TRL's sampled rollout log-ratio estimate before that update, not a greedy-test KL.",
                  "Grading cost includes retries and cache hits. It excludes policy generation/training time.", ""])
    if not summary["test_evaluated"]:
        lines.append("Official-test evaluation is disabled or pending; these are development results.")
    elif summary["test_limit"] is not None:
        lines.append("This smoke run uses a subset of the official test questions, not the full benchmark.")
    lines.append("The official test set was previously inspected in exploratory work; this is a new follow-up protocol.")
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for seed in summary["run_seeds"]:
        fig = Figure(figsize=(14, 7), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3, squeeze=False).flatten()
        labels = {"numeric_match": "Numeric-match rate", "format_compliant": "Format compliance",
                  "length_capped": "Length-capped rate", "proxy": "Proxy grade (z)",
                  "judge": "Judge grade (z)", "high_gap": "High-gap rate"}
        for ax, field in zip(axes, labels, strict=True):
            baseline = next(r for r in rows if r["seed"] == seed and r["arm"] == "base" and r["cohort"] == "monitor")
            for arm in ("proxy", "judge", "knn"):
                data = [baseline] + sorted([r for r in rows if r["seed"] == seed and r["arm"] == arm and r["cohort"] == "monitor"], key=lambda r: r["update"])
                ax.plot([r["update"] for r in data], [r[field] for r in data], marker="o", label=arm)
            ax.set(title=labels[field], xlabel="Scheduled PPO batches")
            if field not in ("proxy", "judge"):
                ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        fig.suptitle(f"GSM8K development monitor — seed {seed}")
        fig.savefig(folder / f"monitor-seed-{seed}.png", dpi=160)
        fig.savefig(folder / f"monitor-seed-{seed}.pdf")
        if seed == summary["run_seeds"][0]:
            fig.savefig(folder / "monitor.png", dpi=160)
