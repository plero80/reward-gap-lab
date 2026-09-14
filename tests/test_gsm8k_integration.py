"""Exercise native TRL PPO and real tiny-model embeddings on CPU."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
from test_policy import loaded, actor_for, prompts
from test_gsm8k import write_prepared
from reward_gap._trl_bridge import RewardBridge
from reward_gap.config import TrainingConfig, RuntimeConfig
from reward_gap.formatting import format_policy_batch
from reward_gap.gsm8k.config import COHORTS, GSMConfig, load_gsm_config
from reward_gap.gsm8k.data import Question
from reward_gap.gsm8k.experiment import GSMExperiment
from reward_gap.gsm8k.graders import LanguageGrader, POOLING
from reward_gap.ppo import PPOTrainer
from reward_gap.scorers import ScoreBatch


def test_reward_bridge_uses_token_completion_not_answer_text(loaded):
    calls = []
    class Reward:
        def score_rollouts(self, records, answers, **metadata):
            calls.append((answers, metadata))
            return SimpleNamespace(prompt_ids=tuple(p.prompt_id for p in records), rewards=(1., 2.))
    actor = actor_for(loaded)
    batch = format_policy_batch(actor.tokenizer, prompts(), max_prompt_tokens=64, max_new_tokens=4, context_window=128)
    bridge = RewardBridge(actor.tokenizer, Reward(), eos_ids=actor.eos_ids)
    bridge.bind(prompts(), batch)
    token = actor.tokenizer.encode("a", add_special_tokens=False)[0]
    eos = actor.eos_ids[0]
    prefixes = batch.input_ids.masked_fill(~batch.attention_mask.bool(), 0)
    ids = torch.cat((prefixes, torch.tensor([[token, eos, 0, 0], [token] * 4])), dim=1)
    mask = torch.cat((batch.attention_mask, torch.tensor([[1, 1, 0, 0], [1] * 4])), dim=1)
    result = bridge(ids, attention_mask=mask)
    assert calls[0][1] == {"response_lengths": [2, 4], "finish_reasons": ["eos", "length"]}
    assert calls[0][0] == ["a", "aaaa"]
    assert result.hidden_states[0][:, -1, 0].tolist() == [1., 2.]
    bridge.bind(prompts(), batch)
    mask[1, batch.input_ids.shape[1] + 1] = 0
    with pytest.raises(ValueError, match="Unexpected PAD"):
        bridge(ids, attention_mask=mask)


def make_grader(loaded, tmp_path):
    loaded = replace(loaded, revision="p1")
    loaded.model.config.max_position_embeddings = 4096
    question = Question("q", "What is 2 + 3?", "Add to get 5. #### 5", "5")
    grader = LanguageGrader(loaded, "proxy", {"q": question},
                            {"grading_budgets": [16, 32], "grading_max_input": 2048}, tmp_path)
    return grader, question.prompt()


def test_grader_retries_caches_and_pools_actual_postnorm_prefix(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    captured, calls = [], []
    def capture(module, args, output):
        captured.append(output.last_hidden_state[0, -1].detach().float())
    handle = loaded.model.base_model.register_forward_hook(capture)
    def generate(**kwargs):
        calls.append(kwargs)
        output = "uncertain" if len(calls) == 1 else "Correct.\nSCORE: 5"
        tokens = torch.tensor([loaded.tokenizer.encode(output, add_special_tokens=False)])
        return torch.cat((kwargs["input_ids"], tokens), dim=1)
    monkeypatch.setattr(loaded.model, "generate", generate)
    result = grader.score([prompt], [r"\boxed{5}"], return_embeddings=True)
    handle.remove()
    assert result.scores == (5.,) and result.embedding_pooling == POOLING
    assert [c["generation_config"].max_new_tokens for c in calls] == [16, 32]
    assert len(captured) == 1
    expected = captured[0] / captured[0].norm()
    torch.testing.assert_close(result.embeddings[0].float(), expected)
    text = loaded.tokenizer.decode(calls[0]["input_ids"][0], skip_special_tokens=False)
    assert "reference_solution" in text and "candidate" in text and text.endswith("[ASSISTANT]")
    again = grader.score([prompt], [r"\boxed{5}"], return_embeddings=True)
    assert len(calls) == 2 and torch.equal(result.embeddings, again.embeddings)
    costs = [json.loads(line) for line in (tmp_path / "grading_cost.jsonl").read_text().splitlines()]
    assert sum(c.get("valid_grade") is False for c in costs) == 1
    assert costs[-1]["cache_hit"]


def test_grader_can_add_embedding_to_cached_grade_without_regrading(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    calls = []
    def generate(**kwargs):
        calls.append(1)
        tokens = torch.tensor([loaded.tokenizer.encode("SCORE: 4", add_special_tokens=False)])
        return torch.cat((kwargs["input_ids"], tokens), dim=1)
    monkeypatch.setattr(loaded.model, "generate", generate)
    grader.score([prompt], ["5"])
    result = grader.score([prompt], ["5"], return_embeddings=True)
    assert len(calls) == 1 and result.scores == (4.,)
    assert result.embeddings.norm(dim=1).item() == pytest.approx(1.)


def test_all_malformed_grades_fail_instead_of_fabricating_reward(loaded, tmp_path, monkeypatch):
    grader, prompt = make_grader(loaded, tmp_path)
    calls = []
    def generate(**kwargs):
        calls.append(1)
        tokens = torch.tensor([loaded.tokenizer.encode("No score", add_special_tokens=False)])
        return torch.cat((kwargs["input_ids"], tokens), dim=1)
    monkeypatch.setattr(loaded.model, "generate", generate)
    with pytest.raises(ValueError, match="Malformed proxy grades"):
        grader.score([prompt], ["5"])
    assert len(calls) == 2 and not list(tmp_path.glob("grade_cache/*.json"))


@pytest.fixture
def setup(loaded, tmp_path):
    original = load_gsm_config(Path(__file__).parents[1] / "configs/gsm8k_smoke.json")
    settings = {**original.settings, "cohorts": dict.fromkeys(COHORTS, 3), "test_limit": 3,
                "prepared_dir": str(tmp_path / "prepared"), "preparation_responses": 1,
                "questions_per_update": 1, "responses_per_question": 2, "k_values": [1, 2],
                "run_seeds": [7], "evaluate_test": True}
    config = GSMConfig(replace(original.base, seeds=(7,), training=TrainingConfig(round1_updates=1, total_updates=2,
                        rollout_batch_size=2, minibatch_size=1, ppo_epochs=1, checkpoint_every=1, learning_rate=.001),
                        runtime=RuntimeConfig(output_root=tmp_path / "outputs")), settings)
    write_prepared(config)
    calls = []
    def actor(seed):
        calls.append(("actor", seed))
        copy = deepcopy(loaded)
        copy.model.config.max_position_embeddings = 2048
        result = actor_for(copy)
        result.generation = replace(result.generation, max_prompt_tokens=1024)
        return result
    class Grader:
        def __init__(self, role):
            self.role, self.phase = role, ""
            self.loaded = SimpleNamespace(source=role, revision="p1" if role == "proxy" else "j1")
        def score(self, records, answers, *, return_embeddings=False):
            calls.append((self.role, self.phase))
            numbers = [int(p.prompt_id.rsplit("-", 1)[1]) for p in records]
            scores = tuple(float(1 + n if self.role == "proxy" else 5 - n) for n in numbers)
            return ScoreBatch(tuple(p.prompt_id for p in records), scores, tuple([10] * len(records)),
                              self.role, self.loaded.source, self.loaded.revision,
                              torch.tensor([[1., n + 1.] for n in numbers]) if return_embeddings else None,
                              POOLING if return_embeddings else None)
    def factory(cfg=None):
        return GSMExperiment(cfg or config, config.base.runtime.output_root / "test",
                             actor_factory=actor, scorer_factory=Grader)
    return config, factory, calls


def test_full_native_trl_run_same_initial_weights_and_test_only_after_training(setup, monkeypatch):
    config, factory, calls = setup
    original = PPOTrainer.update
    starts = []
    def update(self, *args, **kwargs):
        calls.append(("update", self.reward_id))
        if self.update_count == 0:
            starts.append({n: p.detach().clone() for n, p in self.actor.named_parameters() if p.requires_grad})
        result = original(self, *args, **kwargs)
        assert self._backend.args.temperature == .7
        assert self._identity()["sampling_temperature"] == .7
        if self.update_count == 1:
            assert result.library_metrics["objective/kl"] == pytest.approx(0., abs=1e-6)
        return result
    monkeypatch.setattr(PPOTrainer, "update", update)
    experiment = factory()
    result = experiment.run()
    assert result.state == "completed"
    assert sorted(p.relative_to(result.run_dir).as_posix() for p in result.run_dir.rglob("*.pt")) == [
        "seed-7/judge/final.pt", "seed-7/knn/final.pt", "seed-7/proxy/final.pt"]
    assert len(starts) == 3
    for weights in starts[1:]:
        for name in starts[0]:
            torch.testing.assert_close(weights[name], starts[0][name], rtol=0, atol=0)
    summary = json.loads(result.summary_path.read_text())
    assert set(summary["aggregate_final"]) == {"base", "proxy", "judge", "knn"}
    assert summary["aggregate_final"]["knn"]["numeric_match"]["sample_std"] is None
    assert summary["test_evaluated"]
    indices = [i for i, (role, phase) in enumerate(calls) if role == "judge" and "/evaluation/final/" in phase]
    assert min(indices) > max(i for i, c in enumerate(calls) if c[0] == "update")
    assert len([c for c in calls if c[0] == "update"]) == 6
    assert not any(role == "judge" and phase.endswith(("training/proxy", "training/knn")) for role, phase in calls)
    for arm in ("proxy", "judge", "knn"):
        audit = json.loads((result.run_dir / f"seed-7/{arm}/rollouts/update-000001.json").read_text())
        assert len(audit) == 2
        assert all(r["reward"] == pytest.approx(r["task_reward"] - r["format_penalty"] - r["length_penalty"]) for r in audit)
    count = len(calls)
    assert factory().run().state == "completed" and len(calls) == count


def test_resume_after_training_failure_and_before_official_test(setup, monkeypatch):
    config, factory, calls = setup
    original = PPOTrainer.update
    seen, failed = [], False
    def update(self, *args, **kwargs):
        nonlocal failed
        seen.append((self.reward_id.split("/")[0], self.update_count))
        if self.reward_id.startswith("proxy/") and self.update_count == 1 and not failed:
            failed = True
            raise RuntimeError("simulated interruption")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PPOTrainer, "update", update)
    with pytest.raises(RuntimeError, match="simulated"):
        factory().run()
    result = factory().run(until="training")
    assert result.state == "paused" and seen.count(("proxy", 0)) == 1
    assert not any(role == "judge" and "/evaluation/final/" in phase for role, phase in calls)
    changed = replace(config, settings={**config.settings, "format_penalty": 1.})
    with pytest.raises(ValueError, match="protocol or data changed"):
        factory(changed).run()
    assert factory().run().state == "completed"


def test_pilot_never_grades_official_test(setup):
    config, factory, calls = setup
    config = replace(config, settings={**config.settings, "evaluate_test": False})
    result = factory(config).run()
    assert result.state == "completed"
    assert not any(role == "judge" and "/evaluation/final/" in phase for role, phase in calls)
    assert not json.loads(result.summary_path.read_text())["test_evaluated"]
