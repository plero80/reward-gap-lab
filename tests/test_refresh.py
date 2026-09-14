import json
from dataclasses import replace

import pytest

pytest.importorskip("torch")

from test_evaluation import Actor, Scorer, CAL, CONTEXT, prepared, memory
from reward_gap.refresh import refresh_memory, RefreshError
from reward_gap.memory import GapMemory


def refresh(prepared, destination, mem=None, **kwargs):
    options = dict(prepared_dir=prepared, output_dir=destination, policy_id="trained-policy1",
                   parent_memory_id="M0", memory_id="M1", seed=7, batch_size=2)
    options.update(kwargs)
    return refresh_memory(Actor(), Scorer("proxy"), Scorer("judge"), CAL,
                          memory() if mem is None else mem, **options)


def test_refresh_extends_memory_preserves_parent_and_saves_evidence(prepared, tmp_path):
    parent = memory()
    result = refresh(prepared, tmp_path / "refresh", parent)
    assert parent.size == 2
    assert result.memory.size == 5 and result.added_count == 3
    loaded = GapMemory.load(result.memory_path, context=CONTEXT)
    assert loaded.example_ids == result.memory.example_ids
    rows = json.loads(result.rows_path.read_text())
    assert [r["gap"] for r in rows] == [1., -3., -3.]
    assert all(r["prompt_id"].startswith("refresh-") for r in rows)
    payload = json.loads(result.memory_path.read_text())
    gaps = dict(zip(payload["example_ids"], payload["gaps"]))
    assert all(gaps[r["example_id"]] == r["gap"] for r in rows)
    assert gaps["old-a"] == 0.5 and gaps["old-b"] == -2.
    manifest = json.loads(result.manifest_path.read_text())
    assert (manifest["parent_memory_id"], manifest["memory_id"]) == ("M0", "M1")
    assert manifest["total_count"] == 5
    assert json.loads((result.output_dir / "status.json").read_text())["state"] == "completed"


@pytest.mark.parametrize("cohort", ["calibration", "initial_memory", "training", "validation", "final_evaluation"])
def test_refresh_rejects_overlap_with_every_other_cohort(prepared, tmp_path, cohort):
    path = prepared / f"{cohort}.json"
    rows = json.loads(path.read_text())
    rows[0]["conversation_group"] = "question refresh 0"
    rows[0]["messages"][0]["content"] = "question refresh 0"
    path.write_text(json.dumps(rows))
    with pytest.raises(RefreshError, match="overlap"):
        refresh(prepared, tmp_path / "refresh")
    assert not (tmp_path / "refresh").exists()


def test_duplicate_refresh_version_cannot_be_appended_twice(prepared, tmp_path):
    first = refresh(prepared, tmp_path / "first")
    with pytest.raises(RefreshError, match="already exist"):
        refresh(prepared, tmp_path / "second", first.memory)
    assert first.memory.size == 5


def test_version_and_calibration_must_match(prepared, tmp_path):
    with pytest.raises(RefreshError, match="new version"):
        refresh(prepared, tmp_path / "same", memory_id="M0")
    with pytest.raises(RefreshError, match="incompatible"):
        refresh_memory(Actor(), Scorer("proxy"), Scorer("judge"), replace(CAL, calibration_id="different"),
                       memory(), prepared_dir=prepared, output_dir=tmp_path / "bad", policy_id="p1",
                       parent_memory_id="M0", memory_id="M1", seed=7)


def test_memory_write_failure_does_not_publish_completed_result(prepared, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(GapMemory, "save", fail)
    parent = memory()
    with pytest.raises(OSError, match="disk failure"):
        refresh(prepared, tmp_path / "failed", parent)
    assert parent.size == 2
    assert json.loads((tmp_path / "failed/status.json").read_text())["state"] == "failed"
    assert not (tmp_path / "failed/manifest.json").exists()
