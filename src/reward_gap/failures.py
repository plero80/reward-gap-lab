"""Explicit sample failures; infrastructure and programming errors stay fatal."""

import json
from pathlib import Path


class SampleError(ValueError):
    """An unusable model response, with no replacement score implied."""


def record_failure(folder, **event):
    path = Path(folder) / "sample_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
