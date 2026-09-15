"""Physical call and per-answer cost accounting across old and new run logs."""

import json

import pytest

from reward_gap.gsm8k.costs import summarize_grading_costs


def write_log(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"phase": "memory", "role": "judge", "input_tokens": 0,
                                     "generated_tokens": 0, "seconds": 0, **e}) + "\n"
                            for e in events), encoding="utf-8")


def test_mixed_serial_and_batched_logs_preserve_real_costs(tmp_path):
    path = tmp_path / "grading_cost.jsonl"
    write_log(path, [
        {"attempt": 1, "valid_grade": True, "seconds": 2, "input_tokens": 10, "generated_tokens": 3},
        {"embedding_only": True, "seconds": 1, "input_tokens": 10},
        {"attempt": 1, "valid_grade": True, "batch_id": "generation", "batch_size": 2,
         "seconds": 1.5, "batch_seconds": 3, "input_tokens": 11, "generated_tokens": 4},
        {"attempt": 1, "valid_grade": False, "batch_id": "generation", "batch_size": 2,
         "seconds": 1.5, "batch_seconds": 3, "input_tokens": 12, "generated_tokens": 5},
        {"attempt": 2, "valid_grade": False, "batch_id": "retry", "batch_size": 1,
         "output_tokens_known": False, "seconds": 1, "input_tokens": 12},
        {"embedding_only": True, "batch_id": "embedding", "batch_size": 2, "seconds": .25, "batch_seconds": .5},
        {"embedding_only": True, "batch_id": "embedding", "batch_size": 2, "seconds": .25, "batch_seconds": .5},
        {"cache_hit": True},
        {"deduplicated": True},
    ])
    cost = summarize_grading_costs([path])["memory/judge"]
    assert cost["generation_attempts"] == 4 and cost["generation_calls"] == 3
    assert cost["embedding_samples"] == 3 and cost["embedding_forwards"] == 2
    assert cost["invalid_attempts"] == 2 and cost["unknown_output_attempts"] == 1
    assert cost["valid_attempt_fraction"] == .5 and cost["cache_hits"] == 1
    assert cost["seconds"] == pytest.approx(7.5)
    assert cost["input_tokens"] == 55 and cost["generated_tokens"] == 12


def test_teacher_folders_keep_separate_call_counts(tmp_path):
    for name in ("4b", "30b"):
        write_log(tmp_path / name / "grading_cost.jsonl", [
            {"attempt": 1, "valid_grade": True, "batch_id": "same-id", "batch_size": 2},
            {"attempt": 1, "valid_grade": True, "batch_id": "same-id", "batch_size": 2},
        ])
    costs = summarize_grading_costs(tmp_path.rglob("grading_cost.jsonl"), root=tmp_path)
    assert set(costs) == {"4b/memory/judge", "30b/memory/judge"}
    assert all(c["generation_calls"] == 1 and c["generation_attempts"] == 2 for c in costs.values())
    assert summarize_grading_costs([tmp_path / "missing.jsonl"]) == {}
