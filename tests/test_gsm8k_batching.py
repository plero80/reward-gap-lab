"""True grader batching, row isolation and padding-safe real model embeddings."""

from collections import defaultdict
from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_gsm8k_integration import make_grader, loaded
from reward_gap.gsm8k.costs import summarize_grading_costs


def scripted(loaded, monkeypatch, scripts):
    calls, attempts = [], defaultdict(int)
    tokenizer = loaded.tokenizer
    def generate(**kwargs):
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        budget = kwargs["generation_config"].max_new_tokens
        rows, labels = [], []
        for row in ids:
            text = tokenizer.decode(row, skip_special_tokens=False)
            key = next(key for key in scripts if f'"candidate": "{key}"' in text)
            script = scripts[key]
            output = script[min(attempts[key], len(script) - 1)]
            attempts[key] += 1
            labels.append(key)
            tokens = tokenizer.encode(output, add_special_tokens=False) + [tokenizer.eos_token_id]
            rows.append(tokens[:budget])
        calls.append({"labels": labels, "ids": ids.clone(), "mask": mask.clone(), "budget": budget})
        suffix = torch.full((len(rows), max(map(len, rows))), tokenizer.pad_token_id, dtype=torch.long)
        for index, tokens in enumerate(rows):
            suffix[index, :len(tokens)] = torch.tensor(tokens)
        return torch.cat((ids, suffix), dim=1)
    monkeypatch.setattr(loaded.model, "generate", generate)
    return calls


def events(folder):
    return [json.loads(line) for line in (folder / "grading_cost.jsonl").read_text().splitlines()]


def test_batches_uncached_rows_and_retries_only_invalid_rows(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    answers = ["a", "longer answer", "c"]
    # A terminal score must be on its own line.
    calls = scripted(loaded, monkeypatch, {"a": ["SCORE: 5"], "longer answer": ["unknown", "SCORE: 2"], "c": ["OK\nSCORE: 3"]})
    result = grader.score([prompt] * 3, answers, return_embeddings=True)
    assert result.scores == (5., 2., 3.)
    assert [c["labels"] for c in calls] == [answers, ["longer answer"]]
    assert result.prompt_ids == ("q",) * 3
    assert result.embeddings.shape == (3, 16)
    assert len({len(c["ids"]) for c in calls}) == 2
    assert any(calls[0]["mask"][:, 0].eq(0))  # Unequal inputs are left-padded.
    logs = events(tmp_path)
    assert [e["batch_size"] for e in logs if e.get("embedding_only")] == [3, 3, 3]
    for batch_id in {e["batch_id"] for e in logs}:
        group = [e for e in logs if e["batch_id"] == batch_id]
        assert sum(e["seconds"] for e in group) == pytest.approx(group[0]["batch_seconds"])
    cost = summarize_grading_costs([tmp_path / "grading_cost.jsonl"])["unassigned/proxy"]
    assert cost["generation_calls"] == len(calls) == 2
    assert cost["generation_attempts"] == 4 and cost["invalid_attempts"] == 1
    assert cost["embedding_forwards"] == 1 and cost["embedding_samples"] == 3
    count = len(calls)
    assert grader.score([prompt] * 3, answers, return_embeddings=True).scores == result.scores
    assert len(calls) == count


def test_cached_and_duplicate_rows_do_not_use_gpu_again(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    calls = scripted(loaded, monkeypatch, {"cached": ["SCORE: 4"], "new": ["SCORE: 2"]})
    grader.score([prompt], ["cached"])
    result = grader.score([prompt] * 4, ["cached", "new", "cached", "new"])
    assert result.scores == (4., 2., 4., 2.)
    assert [c["labels"] for c in calls] == [["cached"], ["new"]]
    assert len(list((tmp_path / "grade_cache").glob("*.json"))) == 2


def test_batch_limit_and_original_order(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    grader.batch_size = 2
    answers = [f"item{i}" for i in range(5)]
    calls = scripted(loaded, monkeypatch, {a: [f"SCORE: {i + 1}"] for i, a in enumerate(answers)})
    assert grader.score([prompt] * 5, answers).scores == (1., 2., 3., 4., 5.)
    assert [len(c["labels"]) for c in calls] == [2, 2, 1]


def test_overlong_and_malformed_rows_do_not_discard_valid_results(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    calls = scripted(loaded, monkeypatch, {"bad": ["unknown"], "good": ["SCORE: 4"]})
    answers = ["bad", "x" * 3000, "good"]
    results, errors = grader.score_partial([prompt] * 3, answers)
    assert results[0] is results[1] is None
    assert "Malformed" in errors[0] and "context limits" in errors[1]
    assert results[2].scores == (4.,) and errors[2] is None
    assert [c["labels"] for c in calls] == [["bad", "good"], ["bad"]]
    assert len(list((tmp_path / "grade_cache").glob("*.json"))) == 1


def test_eos_padding_and_actual_token_counts(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    grader.settings["grading_budgets"] = [32]
    # PAD is also an alternate EOS in real Qwen checkpoints.
    loaded.model.generation_config.eos_token_id = [loaded.tokenizer.eos_token_id, loaded.tokenizer.pad_token_id]
    outputs = {"short": "SCORE: 4", "long": "Explanation\nSCORE: 5"}
    calls = scripted(loaded, monkeypatch, {k: [v] for k, v in outputs.items()})
    assert grader.score([prompt] * 2, list(outputs)).scores == (4., 5.)
    assert len(calls) == 1
    logs = events(tmp_path)
    assert [e["generated_tokens"] for e in logs] == [len(loaded.tokenizer.encode(text, add_special_tokens=False)) + 1 for text in outputs.values()]
    assert all(e["finish_reason"] == "eos" for e in logs)


def test_real_embeddings_match_unpadded_singletons(loaded, tmp_path, monkeypatch):
    singleton = deepcopy(loaded)
    grader, prompt = make_grader(loaded, tmp_path / "batch")
    reference, _ = make_grader(singleton, tmp_path / "single")
    answers = ["a", "a much longer answer", "different"]
    scripts = {a: ["SCORE: 4"] for a in answers}
    scripted(loaded, monkeypatch, scripts)
    scripted(singleton, monkeypatch, scripts)
    batched = grader.score([prompt] * 3, answers, return_embeddings=True)
    separate = [reference.score([prompt], [a], return_embeddings=True) for a in answers]
    torch.testing.assert_close(batched.embeddings, torch.cat([b.embeddings for b in separate]), atol=1e-6, rtol=1e-5)
    assert batched.token_counts == tuple(b.token_counts[0] for b in separate)


def test_cached_grades_get_batched_embeddings_without_generation(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    answers = ["first", "second"]
    calls = scripted(loaded, monkeypatch, {a: ["SCORE: 4"] for a in answers})
    grader.score([prompt] * 2, answers)
    result = grader.score([prompt] * 2, answers, return_embeddings=True)
    assert len(calls) == 1 and result.embeddings.shape[0] == 2
    assert [e["batch_size"] for e in events(tmp_path) if e.get("embedding_only")] == [2, 2]


def test_native_greedy_generation_matches_individual_calls(loaded, tmp_path):
    reference_model = deepcopy(loaded)
    grader, prompt = make_grader(loaded, tmp_path / "batch")
    reference, _ = make_grader(reference_model, tmp_path / "single")
    for current in (grader, reference):
        current.settings["grading_budgets"] = [4]
    answers = ["a", "a longer answer"]
    # This random tiny model need not produce a valid grade. Compare the actual
    # Transformers greedy decoding, not scripted outputs or manufactured labels.
    grader.score_partial([prompt] * 2, answers)
    for answer in answers:
        reference.score_partial([prompt], [answer])
    batched, single = events(tmp_path / "batch"), events(tmp_path / "single")
    assert [e["grading_text"] for e in batched] == [e["grading_text"] for e in single]
    assert [e["generated_tokens"] for e in batched] == [e["generated_tokens"] for e in single]
    assert {e["batch_size"] for e in batched} == {2}
    assert {e["batch_size"] for e in single} == {1}


def test_success_cache_survives_later_retry_failure(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    calls = scripted(loaded, monkeypatch, {"good": ["SCORE: 5"], "bad": ["unknown"]})
    original = loaded.model.generate
    def sometimes(**kwargs):
        if calls:
            raise torch.OutOfMemoryError("retry OOM")
        return original(**kwargs)
    monkeypatch.setattr(loaded.model, "generate", sometimes)
    with pytest.raises(torch.OutOfMemoryError, match="retry OOM"):
        grader.score_partial([prompt] * 2, ["good", "bad"])
    assert len(list((tmp_path / "grade_cache").glob("*.json"))) == 1
    assert grader.score([prompt], ["good"]).scores == (5.,)


def test_loaded_grader_uses_configured_scoring_batch_size(loaded, tmp_path, monkeypatch):
    from dataclasses import replace
    from reward_gap.gsm8k import graders
    from reward_gap.gsm8k.config import load_gsm_config
    config = load_gsm_config("configs/gsm8k_b200_seed42.json")
    monkeypatch.setattr(graders, "load_policy_model", lambda *args: replace(loaded, revision="p1"))
    grader = graders.LanguageGrader.load(config, "proxy", {}, tmp_path)
    assert grader.batch_size == config.base.scoring.batch_size == 8
