"""Independent outcomes, paired teacher effects, development checks and costs."""

import csv
import json
import random
from collections import defaultdict

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from reward_gap.artifacts import atomic_write_json
from reward_gap.gsm8k.metrics import _correlation
from reward_gap.gsm8k.recovery import aggregate_present


def read(path):
    from pathlib import Path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preparation_report(experiment):
    checks, packs, keys = [], [], []
    for seed in experiment.config.seeds:
        rows = {}
        for name in ("4b", "30b"):
            source = experiment.status["stages"][f"seed-{seed}/fit/{name}"]["result"]
            rows[name] = read(source["monitor"])
            resolved = [r for r in rows[name] if not r["unresolved"]]
            wrong = [r for r in resolved if r["numeric_mismatch"]]
            checks.append({"seed": seed, "teacher": name, "teacher_identity": experiment.status["models"][f"teacher-{name}"],
                           "count": len(rows[name]), "resolved": len(resolved), "unresolved": len(rows[name]) - len(resolved),
                           "wrong_numeric": len(wrong), "wrong_numeric_grade_ge_4": sum(r["raw_judge"] >= 4 for r in wrong),
                           "high_grade_rate_among_wrong": sum(r["raw_judge"] >= 4 for r in wrong) / len(wrong) if wrong else None,
                           "grade_numeric_correlation_resolved": _correlation(np.array([r["raw_judge"] for r in resolved]),
                                                                             np.array([int(r["numeric_match"]) for r in resolved])),
                           "corrected_reward_numeric_correlation_resolved": _correlation(np.array([r["corrected_reward"] for r in resolved]),
                                                                                        np.array([int(r["numeric_match"]) for r in resolved])),
                           "prediction_diagnostics": read(source["diagnostics"])})
        rng = random.Random(f"{seed}/teacher-review")
        for left, right in zip(rows["4b"], rows["30b"], strict=True):
            if (left["example_id"], left["answer"]) != (right["example_id"], right["answer"]):
                raise ValueError("Teacher review requires identical examples")
            if left["raw_judge"] == right["raw_judge"]:
                continue
            order = [left, right]
            rng.shuffle(order)
            review_id = f"seed-{seed}/review-{len(packs)}"
            packs.append({"id": review_id, "question": left["question"], "response": left["answer"],
                          "reference_solution": experiment.questions[left["question_id"]].solution,
                          "grade_A": order[0]["raw_judge"], "grade_B": order[1]["raw_judge"],
                          "human_numeric_verdict": None, "human_reasoning_verdict": None, "notes": ""})
            keys.append({"id": review_id, "example_id": left["example_id"],
                         "A": order[0]["teacher"], "B": order[1]["teacher"]})
    atomic_write_json(experiment.run_dir / "teacher_checks.json", checks)
    # Resuming after preparation must preserve any reviewer edits.
    if not (experiment.run_dir / "blinded_review.json").exists():
        atomic_write_json(experiment.run_dir / "blinded_review.json", packs)
    if not (experiment.run_dir / "review_key.json").exists():
        atomic_write_json(experiment.run_dir / "review_key.json", keys)


def write_report(folder, summary):
    rows = [{"seed": r["seed"], "arm": r["arm"], "cohort": r["cohort"], "update": r["update"], **r["metrics"]}
            for r in summary["results"]]
    fields = ["seed", "arm", "cohort", "update", "numeric_match", "strict_match", "format_compliant", "unresolved",
              "numeric_mismatch", "length_capped", "response_tokens", "proxy", "graded_count", "failed_count", "count", "training_rollout_kl"]
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finals = [r for r in rows if r["cohort"] == "final"]
    aggregates, paired = {}, []
    for arm in summary["arms"]:
        selected = [r for r in finals if r["arm"] == arm]
        if selected:
            aggregates[arm] = {field: aggregate_present(selected, field) for field in fields[4:-2]}
    for seed in summary["run_seeds"]:
        arms = {r["arm"]: r for r in finals if r["seed"] == seed}
        if "knn4" in arms and "knn30" in arms:
            paired.append({"seed": seed, "numeric_match_difference": arms["knn30"]["numeric_match"] - arms["knn4"]["numeric_match"]})
    summary.update(aggregate_final=aggregates, paired_knn30_minus_knn4=paired,
                   paired_difference_mean=float(np.mean([r["numeric_match_difference"] for r in paired])) if paired else None,
                   paired_difference_sample_std=float(np.std([r["numeric_match_difference"] for r in paired], ddof=1)) if len(paired) > 1 else None)
    costs = defaultdict(lambda: {"generation_attempts": 0, "invalid_attempts": 0, "cache_hits": 0,
                                "embedding_forwards": 0, "unknown_output_attempts": 0,
                                "input_tokens": 0, "generated_tokens": 0, "seconds": 0.})
    for path in folder.rglob("grading_cost.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            key = f"{path.parent.relative_to(folder).as_posix()}/{event['phase']}/{event['role']}"
            cost = costs[key]
            cost["generation_attempts"] += "attempt" in event
            cost["invalid_attempts"] += event.get("valid_grade") is False
            cost["cache_hits"] += bool(event.get("cache_hit"))
            cost["embedding_forwards"] += bool(event.get("embedding_only"))
            cost["unknown_output_attempts"] += event.get("output_tokens_known") is False
            for field in ("input_tokens", "generated_tokens", "seconds"):
                cost[field] += event[field]
    for cost in costs.values():
        cost["valid_attempt_fraction"] = 1 - cost["invalid_attempts"] / cost["generation_attempts"] if cost["generation_attempts"] else None
    summary["grading_cost"] = dict(costs)
    lines = ["# GSM8K: matched 4B versus 30B memory teachers", "",
             f"Protocol: `{summary['protocol']}`. Numeric checker: `{summary['parser']}`.",
             "Primary comparison: kNN–30B minus kNN–4B numeric-match rate, paired within each seed.",
             "Both memories use identical examples/proxy vectors and fixed retrieval settings. Only teacher grades and teacher normalization differ.",
             "Judge-4B PPO trains the same small policy using the frozen 4B judge's normalized grade plus the common penalties.",
             "All policies use the same numeric evaluator; unresolved answers remain in the denominator.",
             "Missing proxy grades are excluded from proxy statistics only; graded_count and failed_count report coverage.",
             "Teacher fitting uses the intersection of valid labels; each fit/coverage.json lists excluded examples.",
             "Update numbers count scheduled batches. summary.json reports optimized_batches and skipped_batches per arm.",
             "Final teacher grades are not computed. Predicted gaps are not substituted for actual teacher measurements.", "",
             "| Seed | Policy | Cohort | Update | Numeric | Strict | Format | Unresolved | Truncated |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        values = [str(row[k]) for k in ("seed", "arm", "cohort", "update")]
        values += [f"{row[k]:.4f}" for k in ("numeric_match", "strict_match", "format_compliant", "unresolved", "length_capped")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Development checks: teacher_checks.json (including wrong numeric answers graded >=4 and reserved-cohort prediction errors).",
              "Review reasoning disagreements in blinded_review.json; keep review_key.json away from reviewers.",
              "Numeric correctness does not establish reasoning correctness. A smaller gap error against a different teacher is not evidence of better answers.",
              "Per-seed final means/sample standard deviations and paired differences are in summary.json. One-seed standard deviations are undefined.",
              "Cost logs count grading attempts, retries, cache hits, tokens and grading time. They exclude policy training and model loading time.",
              "Generation exceptions count as failed attempts; their unavailable output-token counts are flagged separately, not treated as known zero output.",
              "The official test set was previously inspected in exploratory work; this is a separate follow-up protocol."]
    if not summary["test_evaluated"]:
        lines.append("Official-test evaluation is disabled or pending. Current results are development-only.")
    elif summary["test_limit"] is not None:
        lines.append("This configuration evaluates a test subset, not the full benchmark.")
    if rows:
        for seed in summary["run_seeds"]:
            fig = Figure(figsize=(12, 6), layout="constrained")
            FigureCanvasAgg(fig)
            for ax, field in zip(fig.subplots(2, 2, squeeze=False).flatten(), ("numeric_match", "format_compliant", "unresolved", "length_capped"), strict=True):
                initial = [r for r in rows if r["seed"] == seed and r["arm"] == "base" and r["cohort"] == "monitor"]
                for arm in (a for a in summary["arms"] if a != "base"):
                    points = initial + sorted([r for r in rows if r["seed"] == seed and r["arm"] == arm and r["cohort"] == "monitor"], key=lambda r: r["update"])
                    ax.plot([r["update"] for r in points], [r[field] for r in points], marker="o", label=arm)
                ax.set(title=field.replace("_", " "), xlabel="Scheduled PPO batches", ylim=(0, 1))
                ax.legend(fontsize=8)
            fig.suptitle(f"GSM8K teacher comparison — seed {seed}")
            fig.savefig(folder / f"teacher-monitor-{seed}.png", dpi=160)
            fig.savefig(folder / f"teacher-monitor-{seed}.pdf")
            lines += ["", f"![Seed {seed}](teacher-monitor-{seed}.png)"]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
