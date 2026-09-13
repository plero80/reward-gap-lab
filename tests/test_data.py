import gzip
import json

import pytest

from reward_gap.config import COHORTS, ConfigError, load_config
from reward_gap.data import (DataError, Message, PromptRecord, extract_prompt,
                             partition_prompts, prepare_data, prompt_schedule, read_hh_file, load_prompts)


def pair(prompt="Question", answer="Answer"):
    context = f"\n\nHuman: {prompt}\n\nAssistant: "
    return {"chosen": context + answer, "rejected": context + "Other answer"}


def records(prefix, count):
    return [PromptRecord(f"{prefix}{i}", f"{prefix}group{i}",
                         (Message("user", f"{prefix}question{i}"),)) for i in range(count)]


def counts():
    return {name: 2 for name in COHORTS}


def test_extract_multiturn_removes_only_final_answer():
    row = pair("First\n\nAssistant: Earlier answer\n\nHuman: Follow-up")
    assert extract_prompt(row) == (
        Message("user", "First"), Message("assistant", "Earlier answer"),
        Message("user", "Follow-up"),
    )


@pytest.mark.parametrize("row, error", [
    ({}, "missing_transcript"),
    ([], "row_not_object"),
    ({"chosen": "plain", "rejected": "plain"}, "invalid_transcript_prefix"),
    (pair(""), "empty_message"),
    (pair("Q", ""), "empty_message"),
    (pair("Q\n\nHuman: repeated"), "nonalternating_roles"),
    ({"chosen": pair("A")["chosen"], "rejected": pair("B")["rejected"]}, "different_pair_contexts"),
])
def test_reject_ambiguous_pairs(row, error):
    with pytest.raises(DataError, match=error):
        extract_prompt(row)


def test_gzip_reader_counts_skips_and_normalizes_group(tmp_path):
    path = tmp_path / "train.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in (pair(" A  QUESTION "), pair("a question"), {}):
            handle.write(json.dumps(row) + "\n")
    result, stats = read_hh_file(path, "helpful-base/train")
    assert len(result) == 2
    assert result[0].conversation_group == result[1].conversation_group
    assert result[0].prompt_id == "helpful-base/train:1"
    assert stats == {"rows": 3, "accepted": 2, "skipped": {"missing_transcript": 1}}


def test_invalid_json_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(pair()) + "\n{", encoding="utf-8")
    with pytest.raises(DataError, match=r":2: invalid JSON"):
        read_hh_file(path, "bad")


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="Cannot read"):
        read_hh_file(tmp_path / "missing.jsonl", "missing")


def test_partition_has_no_group_leakage_and_excludes_all_test_groups():
    train = records("train", 30)
    test = records("test", 10)
    train.append(PromptRecord("overlap", test[-1].conversation_group, (Message("user", "alternate"),)))
    # Another prompt from an existing conversation must stay in its cohort.
    train.append(PromptRecord("related", train[0].conversation_group, (Message("user", "followup"),)))
    train.append(PromptRecord("duplicate", train[1].conversation_group, train[1].messages))
    cohorts, stats = partition_prompts(train, test, counts(), 42)
    seen = set()
    for name, cohort in cohorts.items():
        groups = {r.conversation_group for r in cohort}
        assert not seen.intersection(groups)
        seen.update(groups)
        assert len(cohort) >= 2
        if name != "final_evaluation":
            assert not groups.intersection(r.conversation_group for r in test)
    assert stats["train_prompts_excluded_for_test_overlap"] == 1
    assert stats["train_duplicate_prompts_removed"] == 1
    assert partition_prompts(list(reversed(train)), list(reversed(test)), counts(), 42) == (cohorts, stats)
    assert partition_prompts(train, test, counts(), 43)[0] != cohorts


def test_insufficient_groups_fail():
    with pytest.raises(DataError, match="Insufficient disjoint data"):
        partition_prompts(records("train", 2), records("test", 2), counts(), 42)


def test_optional_validation_can_be_empty():
    requested = counts()
    requested["validation"] = 0
    cohorts, _ = partition_prompts(records("train", 8), records("test", 2), requested, 42)
    assert cohorts["validation"] == ()


def test_schedule_repeatable_and_covers_epoch():
    prompts = tuple(records("train", 4))
    batches = prompt_schedule(prompts, seed=42, updates=3, batch_size=2)
    assert batches == prompt_schedule(tuple(reversed(prompts)), seed=42, updates=3, batch_size=2)
    assert len(set(batches[0] + batches[1])) == 4
    assert len(batches) == 3
    assert all(len(batch) == 2 for batch in batches)
    assert set(sum(batches, ())) <= {r.prompt_id for r in prompts}


@pytest.fixture
def prepared_config(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    raw = {
        "schema_version": 1, "experiment": "smoke", "seeds": [42, 43],
        "data": {"revision": "a" * 40, "subsets": ["helpful-base"],
                 "cache_dir": "raw", "prepared_dir": "prepared", "minimum_prompts": counts()},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    cache = tmp_path / "raw" / ("a" * 40) / "helpful-base"
    cache.mkdir(parents=True)
    for split, count in (("train", 20), ("test", 6)):
        with gzip.open(cache / f"{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for i in range(count):
                handle.write(json.dumps(pair(f"{split} question {i}")) + "\n")
    return path


def test_preparation_offline_and_refuses_overwrite(prepared_config, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline preparation used network")
    monkeypatch.setattr("reward_gap.data.urlopen", no_network)
    manifest_path = prepare_data(prepared_config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["revision"] == "a" * 40
    assert set(manifest["cohorts"]) == set(COHORTS)
    for cohort in manifest["cohorts"].values():
        rows = json.loads((manifest_path.parent / cohort["file"]).read_text(encoding="utf-8"))
        assert len(rows) == cohort["prompts"] == 2
        assert all(r["messages"][-1]["role"] == "user" for r in rows)
        assert [r.to_dict() for r in load_prompts(manifest_path.parent / cohort["file"])] == rows
    before = {p.name: p.read_bytes() for p in manifest_path.parent.iterdir()}
    with pytest.raises(DataError, match="already exists"):
        prepare_data(prepared_config)
    assert before == {p.name: p.read_bytes() for p in manifest_path.parent.iterdir()}


@pytest.mark.parametrize("patch, error", [
    ({"subset": "helpful-base"}, "unknown fields"),
    ({"subsets": ["red-team-attempts"]}, "unsupported"),
    ({"subsets": ["helpful-base", "helpful-base"]}, "duplicate"),
    ({"subsets": []}, "nonempty"),
    ({"split_seed": True}, "split_seed"),
    ({"minimum_prompts": {"training": 2}}, "six cohort"),
    ({"cache_dir": ""}, "cache_dir"),
])
def test_data_configuration_validation(prepared_config, patch, error):
    raw = json.loads(prepared_config.read_text(encoding="utf-8"))
    raw["data"].update(patch)
    prepared_config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=error):
        load_config(prepared_config)


def test_data_paths_and_serialization(prepared_config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(prepared_config)
    assert config.data.cache_dir == tmp_path / "raw"
    assert config.data.prepared_dir == tmp_path / "prepared"
    prepared_config.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_config(prepared_config) == config


@pytest.mark.parametrize("rows, error", [
    ({}, "JSON list"),
    ([{}], "invalid record fields"),
    ([{"prompt_id": "a", "conversation_group": "q", "messages": []}], "end with a user"),
    ([{"prompt_id": "a", "conversation_group": "q",
       "messages": [{"role": "assistant", "content": "answer"}]}], "invalid message"),
    ([{"prompt_id": "a", "conversation_group": "wrong",
       "messages": [{"role": "user", "content": "q"}]}], "does not match"),
])
def test_prepared_reader_rejects_invalid_records(tmp_path, rows, error):
    path = tmp_path / "training.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(DataError, match=error):
        load_prompts(path)
