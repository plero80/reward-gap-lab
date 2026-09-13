"""Save experiment files. Run identity and lifecycle tracking come later."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from reward_gap.config import ExperimentConfig


def atomic_write_json(path: str | Path, data: Any) -> Path:
    """Replace one JSON file only after its new contents have been written.

    The temporary file uses the destination directory so replacement stays
    on the same filesystem. This is a single-file operation, not a lock or
    a transaction across multiple files. Existing destinations are replaced.
    """
    destination = Path(path)
    contents = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        # Close the temporary file before replacement, including on Windows.
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def save_resolved_config(config: ExperimentConfig, run_dir: str | Path) -> Path:
    """Save all resolved settings in the caller's chosen run directory.

    Callers will enforce run identity and completed-run protection when the
    run lifecycle is implemented. This helper itself replaces existing files.
    """
    return atomic_write_json(Path(run_dir) / "resolved_config.json", config.to_dict())
