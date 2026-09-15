"""Grading cost summaries for both serial and batched inference logs."""

import json
from collections import defaultdict


def summarize_grading_costs(paths, *, root=None):
    costs = defaultdict(lambda: {
        "events": 0, "generation_attempts": 0, "generation_calls": 0,
        "embedding_samples": 0, "embedding_forwards": 0, "cache_hits": 0,
        "invalid_attempts": 0, "unknown_output_attempts": 0,
        "input_tokens": 0, "generated_tokens": 0, "seconds": 0.,
    })
    seen_calls = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            key = f"{event['phase']}/{event['role']}"
            if root is not None:
                key = f"{path.parent.relative_to(root).as_posix()}/{key}"
            cost = costs[key]
            cost["events"] += 1
            cost["generation_attempts"] += "attempt" in event
            cost["embedding_samples"] += bool(event.get("embedding_only"))
            cost["invalid_attempts"] += event.get("valid_grade") is False
            cost["cache_hits"] += bool(event.get("cache_hit"))
            cost["unknown_output_attempts"] += event.get("output_tokens_known") is False
            for field in ("input_tokens", "generated_tokens", "seconds"):
                cost[field] += event[field]
            # One batch emits an event per answer. Older logs have no batch_id
            # because every answer used its own model call.
            for field, active in (("generation_calls", "attempt" in event),
                                  ("embedding_forwards", bool(event.get("embedding_only")))):
                if active:
                    batch_id = event.get("batch_id")
                    identity = (key, field, batch_id)
                    if batch_id is None or identity not in seen_calls:
                        cost[field] += 1
                    if batch_id is not None:
                        seen_calls.add(identity)
    for cost in costs.values():
        attempts = cost["generation_attempts"]
        cost["valid_attempt_fraction"] = 1 - cost["invalid_attempts"] / attempts if attempts else None
    return dict(costs)
