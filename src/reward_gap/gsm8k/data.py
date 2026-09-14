"""Question-disjoint GSM8K preparation with reference-free policy prompts."""

import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from reward_gap.artifacts import atomic_write_json
from reward_gap.data import Message, PromptRecord
from reward_gap.gsm8k.answers import numeric
from reward_gap.gsm8k.config import COHORTS

SYSTEM = "Solve the math problem. Show concise calculations and finish with exactly one numeric answer in \\boxed{}."
DEMO_QUESTION = "Mira has 2 apples and buys 3 more. How many apples does she have?"
DEMO_ANSWER = "2 + 3 = 5. The answer is \\boxed{5}."
PROMPT_VERSION = "gsm8k_boxed_fewshot_v1"


def group(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    solution: str
    gold: str

    def prompt(self):
        return PromptRecord(self.id, group(self.question),
                            (Message("system", SYSTEM), Message("user", DEMO_QUESTION),
                             Message("assistant", DEMO_ANSWER), Message("user", self.question)))


def partition(train, test, counts, data_seed, test_limit=None):
    def rows(data, split):
        result = {}
        for index, row in enumerate(data):
            question, answer = row.get("question"), row.get("answer")
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
                raise ValueError("Malformed GSM8K row")
            parts = answer.rsplit("####", 1)
            if len(parts) != 2 or numeric(parts[1]) is None:
                raise ValueError("GSM8K reference requires a numeric #### answer")
            key = group(question)
            if key in result:
                raise ValueError("Duplicate GSM8K question; inspect source rather than silently splitting it")
            result[key] = Question(f"gsm8k/{split}/{index}", question.strip(), answer, parts[1].strip())
        return result
    train_rows, test_rows = rows(train, "train"), rows(test, "test")
    eligible = sorted(k for k in train_rows if k not in test_rows and k != group(DEMO_QUESTION))
    random.Random(data_seed).shuffle(eligible)
    if sum(counts.values()) > len(eligible):
        raise ValueError("Insufficient disjoint GSM8K training questions")
    cohorts, start = {}, 0
    for name in COHORTS:
        cohorts[name] = [train_rows[k] for k in eligible[start:start + counts[name]]]
        start += counts[name]
    ordered_test = sorted(test_rows)
    if test_limit is not None and test_limit > len(ordered_test):
        raise ValueError("test_limit exceeds available test questions")
    cohorts["final"] = [test_rows[k] for k in ordered_test[:test_limit]]
    return cohorts


def prepare(config, *, allow_downloads=False):
    from huggingface_hub import HfApi, snapshot_download
    import pyarrow.parquet as pq
    settings = config.settings
    destination = Path(settings["prepared_dir"])
    if destination.exists():
        raise ValueError("Prepared directory already exists; reuse it or configure a new directory")
    revision = settings["dataset_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        if not allow_downloads:
            raise ValueError("Offline preparation needs the dataset SHA from a previous manifest")
        revision = HfApi().dataset_info("openai/gsm8k", revision=revision).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset API returned an invalid revision")
    snapshot = Path(snapshot_download("openai/gsm8k", repo_type="dataset", revision=revision,
                                     cache_dir=settings["cache_dir"], allow_patterns=["main/*.parquet"],
                                     local_files_only=not allow_downloads))
    def read(split):
        files = sorted((snapshot / "main").glob(f"{split}-*.parquet"))
        if not files:
            raise ValueError(f"Missing GSM8K main/{split} parquet files")
        return [row for file in files for row in pq.read_table(file).to_pylist()]
    cohorts = partition(read("train"), read("test"), settings["cohorts"], settings["data_seed"], settings["test_limit"])
    destination.mkdir(parents=True, exist_ok=False)
    for name, questions in cohorts.items():
        atomic_write_json(destination / f"{name}.json", [asdict(q) for q in questions])
    return atomic_write_json(destination / "manifest.json", {
        "schema_version": 1, "dataset": "openai/gsm8k", "subset": "main", "revision": revision,
        "data_seed": settings["data_seed"], "requested_counts": settings["cohorts"],
        "test_limit": settings["test_limit"], "prompt_version": PROMPT_VERSION,
        "counts": {name: len(rows) for name, rows in cohorts.items()}})


def load_prepared(config):
    root = Path(config.settings["prepared_dir"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for key, expected in (("schema_version", 1), ("dataset", "openai/gsm8k"), ("subset", "main"),
                          ("data_seed", config.settings["data_seed"]),
                          ("requested_counts", config.settings["cohorts"]), ("test_limit", config.settings["test_limit"]),
                          ("prompt_version", PROMPT_VERSION)):
        if manifest.get(key) != expected:
            raise ValueError(f"Prepared GSM8K {key} differs from configuration")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Prepared GSM8K revision is invalid")
    requested = config.settings["dataset_revision"]
    if re.fullmatch(r"[0-9a-f]{40}", requested) and requested != revision:
        raise ValueError("Prepared GSM8K revision differs from configuration")
    result, seen, seen_ids = {}, set(), set()
    for name in (*COHORTS, "final"):
        questions = [Question(**q) for q in json.loads((root / f"{name}.json").read_text(encoding="utf-8"))]
        keys = {group(q.question) for q in questions}
        ids = {q.id for q in questions}
        expected_count = config.settings["cohorts"].get(name, config.settings["test_limit"])
        if (not questions or len(keys) != len(questions) or keys & seen or len(ids) != len(questions)
                or ids & seen_ids or any(not q.id or numeric(q.gold) is None or "####" not in q.solution
                                        or numeric(q.solution.rsplit("####", 1)[1]) != numeric(q.gold) for q in questions)
                or manifest.get("counts", {}).get(name) != len(questions)
                or (expected_count is not None and len(questions) != expected_count)):
            raise ValueError(f"Invalid or overlapping GSM8K cohort: {name}")
        seen.update(keys)
        seen_ids.update(ids)
        result[name] = questions
    return result, manifest


def schedule(questions, settings, seed):
    rng = random.Random(seed)
    return [[q.prompt() for q in rng.sample(questions, settings["questions_per_update"])
             for _ in range(settings["responses_per_question"])] for _ in range(settings["updates"])]
