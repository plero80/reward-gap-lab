"""Prepare prompt-only HH-RLHF cohorts without loading any models.

Uses Anthropic's original JSONL gzip files via the Hugging Face HTTP API.
Related conversations are conservatively grouped by normalized first user
message because the preference files do not provide conversation IDs.
"""

import argparse
import gzip
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

from reward_gap.artifacts import atomic_write_json
from reward_gap.config import COHORTS, DataConfig, load_config

REPOSITORY = "Anthropic/hh-rlhf"
TURN = re.compile(r"\n\n(Human|Assistant):")


class DataError(ValueError):
    """Missing, malformed, or insufficient experiment data."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    conversation_group: str
    messages: tuple[Message, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["messages"] = [asdict(message) for message in self.messages]
        return result


def extract_prompt(row: dict) -> tuple[Message, ...]:
    """Remove the final answer from both transcripts and require agreement.

    Preserve earlier assistant turns as conversation context. Do not select
    the chosen answer as a training target or include it in the prompt.
    """
    if not isinstance(row, dict):
        raise DataError("row_not_object")
    prompts = []
    for key in ("chosen", "rejected"):
        text = row.get(key)
        if not isinstance(text, str):
            raise DataError("missing_transcript")
        markers = list(TURN.finditer(text))
        if not markers or text[:markers[0].start()].strip():
            raise DataError("invalid_transcript_prefix")
        if len(markers) < 2 or markers[-1].group(1) != "Assistant":
            raise DataError("missing_final_answer")
        messages = []
        for index, marker in enumerate(markers):
            expected = "Human" if index % 2 == 0 else "Assistant"
            if marker.group(1) != expected:
                raise DataError("nonalternating_roles")
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            content = text[marker.end():end].strip()
            if not content:
                raise DataError("empty_message")
            messages.append(Message("user" if expected == "Human" else "assistant", content))
        prompts.append(tuple(messages[:-1]))
    if prompts[0] != prompts[1]:
        raise DataError("different_pair_contexts")
    return prompts[0]


def read_hh_file(path: Path, source_name: str) -> tuple[list[PromptRecord], dict]:
    """Read original HH JSONL(.gz); count rejected records rather than guess.

    Invalid JSON or unreadable files fail the whole operation. Structurally
    ambiguous dialogue rows are omitted with reason counts in the manifest.
    """
    records = []
    skipped = Counter()
    opener = gzip.open if path.suffix == ".gz" else open
    rows = 0
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                rows += 1
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise DataError(f"{path}:{line_number}: invalid JSON") from exc
                try:
                    messages = extract_prompt(row)
                except DataError as exc:
                    skipped[str(exc)] += 1
                    continue
                # No content hash: store a conservative group key directly.
                group = " ".join(messages[0].content.split()).casefold()
                records.append(PromptRecord(f"{source_name}:{line_number}", group, messages))
    except (OSError, EOFError, UnicodeError) as exc:
        raise DataError(f"Cannot read {path}: {exc}") from exc
    return records, {"rows": rows, "accepted": len(records), "skipped": dict(skipped)}


def _deduplicate(records: list[PromptRecord]) -> tuple[list[PromptRecord], int]:
    unique = {}
    ids = set()
    for record in sorted(records, key=lambda r: r.prompt_id):
        if record.prompt_id in ids:
            raise DataError(f"Duplicate prompt ID: {record.prompt_id}")
        ids.add(record.prompt_id)
        unique.setdefault(record.messages, record)
    return list(unique.values()), len(records) - len(unique)


def partition_prompts(train: list[PromptRecord], test: list[PromptRecord],
                      minimum_prompts: dict[str, int], seed: int) -> tuple[dict, dict]:
    """Allocate whole groups; counts are minimum prompts, not exact quotas.

    All test groups are reserved before sampling, even unused test groups.
    This excludes overlapping train conversations from development cohorts.
    Sorting before shuffling makes selection independent of input row order.
    """
    if type(seed) is not int or seed < 0:
        raise DataError("split seed must be a nonnegative integer")
    if set(minimum_prompts) != set(COHORTS):
        raise DataError("Provide minimum prompt counts for all six cohorts")
    for name, count in minimum_prompts.items():
        minimum = 0 if name == "validation" else 1
        if type(count) is not int or count < minimum:
            raise DataError(f"{name}: expected integer >= {minimum}")
    train, train_duplicates = _deduplicate(train)
    test, test_duplicates = _deduplicate(test)
    test_groups = {r.conversation_group for r in test}
    eligible = [r for r in train if r.conversation_group not in test_groups]
    removed = len(train) - len(eligible)

    def groups(records):
        grouped = defaultdict(list)
        for record in sorted(records, key=lambda r: r.prompt_id):
            grouped[record.conversation_group].append(record)
        keys = sorted(grouped)
        random.Random(seed).shuffle(keys)
        return [grouped[key] for key in keys]

    training_groups = groups(eligible)
    evaluation_groups = groups(test)
    cohorts = {}
    for name in COHORTS:
        pool = evaluation_groups if name == "final_evaluation" else training_groups
        selected = []
        while len(selected) < minimum_prompts[name] and pool:
            selected.extend(pool.pop())
        if len(selected) < minimum_prompts[name]:
            raise DataError(f"Insufficient disjoint data for {name}: requested at least "
                            f"{minimum_prompts[name]} prompts, available {len(selected)}")
        cohorts[name] = tuple(selected)
    return cohorts, {
        "train_duplicate_prompts_removed": train_duplicates,
        "test_duplicate_prompts_removed": test_duplicates,
        "train_prompts_excluded_for_test_overlap": removed,
        "unused_train_prompts": sum(map(len, training_groups)),
        "unused_test_prompts": sum(map(len, evaluation_groups)),
    }


def prompt_schedule(records: tuple[PromptRecord, ...], *, seed: int,
                    updates: int, batch_size: int) -> tuple[tuple[str, ...], ...]:
    """Return repeatable shuffled-epoch batches to share across branches.

    Draw without replacement within an epoch, reshuffle at the next epoch.
    A batch crossing an epoch boundary can contain a repeated prompt.
    """
    for name, value in (("seed", seed), ("updates", updates), ("batch_size", batch_size)):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise DataError(f"Invalid schedule {name}")
    ids = sorted(record.prompt_id for record in records)
    if not ids or len(set(ids)) != len(ids):
        raise DataError("Schedule needs nonempty records with unique prompt IDs")
    rng = random.Random(seed)
    epoch = []
    batches = []
    for _ in range(updates):
        batch = []
        for _ in range(batch_size):
            if not epoch:
                epoch = ids.copy()
                rng.shuffle(epoch)
            batch.append(epoch.pop())
        batches.append(tuple(batch))
    return tuple(batches)


def load_prompts(path: str | Path) -> tuple[PromptRecord, ...]:
    """Read a prepared cohort and validate its prompt-only chat records."""
    path = Path(path)
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError(f"Cannot read prepared prompts {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise DataError(f"{path}: expected a JSON list")
    records = []
    ids = set()
    for index, row in enumerate(rows):
        location = f"{path}: record {index + 1}"
        if not isinstance(row, dict) or set(row) != {"prompt_id", "conversation_group", "messages"}:
            raise DataError(f"{location}: invalid record fields")
        for name in ("prompt_id", "conversation_group"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise DataError(f"{location}: missing {name}")
        if row["prompt_id"] in ids:
            raise DataError(f"{location}: duplicate prompt_id")
        ids.add(row["prompt_id"])
        messages = row["messages"]
        if not isinstance(messages, list) or not messages or len(messages) % 2 != 1:
            raise DataError(f"{location}: prompt must end with a user turn")
        parsed = []
        for turn, message in enumerate(messages):
            expected = "user" if turn % 2 == 0 else "assistant"
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] != expected
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise DataError(f"{location}: invalid message {turn + 1}")
            parsed.append(Message(message["role"], message["content"]))
        expected_group = " ".join(parsed[0].content.split()).casefold()
        if row["conversation_group"] != expected_group:
            raise DataError(f"{location}: conversation_group does not match first user turn")
        records.append(PromptRecord(row["prompt_id"], row["conversation_group"], tuple(parsed)))
    return tuple(records)


def _revision(config: DataConfig, allow_downloads: bool) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", config.revision):
        return config.revision
    if not allow_downloads:
        raise DataError("Offline preparation needs the resolved revision from a previous manifest")
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{quote(config.revision, safe='')}"
    try:
        with urlopen(url, timeout=60) as response:
            revision = json.load(response)["sha"]
    except (OSError, URLError, ValueError, KeyError) as exc:
        raise DataError(f"Cannot resolve HH-RLHF revision: {exc}") from exc
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DataError("Dataset API returned an invalid revision")
    return revision


def _source_file(config: DataConfig, revision: str, subset: str,
                 split: str, allow_downloads: bool) -> Path:
    relative = f"{subset}/{split}.jsonl.gz"
    path = config.cache_dir / revision / relative
    if path.is_file():
        return path
    if not allow_downloads:
        raise DataError(f"Missing cached dataset file: {path}. Enable downloads explicitly.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{relative}"
        with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            shutil.copyfileobj(response, handle)
        # Check the gzip stream before publishing a downloaded cache file.
        with gzip.open(temporary, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        os.replace(temporary, path)
    except (OSError, URLError, EOFError) as exc:
        raise DataError(f"Cannot download {relative}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def prepare_data(config_path: str | Path, *, download: bool = False) -> Path:
    """Prepare all six cohorts and schedules; never overwrite a directory."""
    config = load_config(config_path)
    data = config.data
    if data is None:
        raise DataError("Add a data section to the experiment configuration")
    if data.prepared_dir.exists():
        raise DataError(f"Prepared directory already exists: {data.prepared_dir}. Choose a new one.")
    allow = download or config.runtime.allow_downloads
    revision = _revision(data, allow)
    loaded = {"train": [], "test": []}
    source_stats = {}
    for subset in data.subsets:
        for split in ("train", "test"):
            key = f"{subset}/{split}"
            print(f"Reading {key}...", flush=True)
            path = _source_file(data, revision, subset, split, allow)
            records, stats = read_hh_file(path, key)
            loaded[split].extend(records)
            source_stats[key] = stats
    cohorts, stats = partition_prompts(loaded["train"], loaded["test"],
                                       data.minimum_prompts, data.split_seed)
    # Directory creation is exclusive, even if another process arrived here.
    data.prepared_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, records in cohorts.items():
        filename = f"{name}.json"
        atomic_write_json(data.prepared_dir / filename, [r.to_dict() for r in records])
        files[name] = {"file": filename, "prompts": len(records),
                       "groups": len({r.conversation_group for r in records})}
    schedules = {}
    for seed in config.seeds:
        filename = f"training_schedule_seed{seed}.json"
        batches = prompt_schedule(cohorts["training"], seed=seed,
                                  updates=config.training.total_updates,
                                  batch_size=config.training.rollout_batch_size)
        atomic_write_json(data.prepared_dir / filename, batches)
        schedules[str(seed)] = filename
    atomic_write_json(data.prepared_dir / "resolved_config.json", config.to_dict())
    manifest = {
        "schema_version": 1, "dataset": REPOSITORY, "revision": revision,
        "subsets": list(data.subsets), "split_seed": data.split_seed,
        "procedure": "normalized-first-user groups; test overlap excluded from train; whole-group allocation v1",
        "minimum_prompts": data.minimum_prompts, "source_rows": source_stats,
        "selection": stats, "cohorts": files, "training_schedules": schedules,
        "schedule_updates": config.training.total_updates,
        "schedule_batch_size": config.training.rollout_batch_size,
    }
    # Written last: its absence identifies incomplete preparation.
    return atomic_write_json(data.prepared_dir / "input_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF experiment prompts")
    parser.add_argument("--config", required=True, help="Experiment JSON path")
    parser.add_argument("--download", action="store_true", help="Allow dataset downloads")
    args = parser.parse_args()
    try:
        path = prepare_data(args.config, download=args.download)
    except (DataError, ValueError, OSError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    print(f"Prepared data manifest: {path}")


if __name__ == "__main__":
    main()
