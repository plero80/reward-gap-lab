from copy import deepcopy
from dataclasses import replace
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from test_policy import loaded, actor_for, prompts
from reward_gap.config import TrainingConfig
from reward_gap.ppo import PPOError, PPOTrainer, _resolved_device
from reward_gap.formatting import format_policy_batch
from reward_gap.rewards import RewardBatch


class Reward:
    def score(self, records, answers):
        values = tuple(1. + len(a) / 10 for a in answers)
        return RewardBatch(tuple(r.prompt_id for r in records), values, values, values,
                           (0.,) * len(values), (1,) * len(values), "proxy", "test-cal", "test-proxy", "v1")


@pytest.mark.parametrize("current", [0, 1])
def test_implicit_cuda_device_matches_current_gpu_only(monkeypatch, current):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    assert _resolved_device(torch.device("cuda")) == _resolved_device(torch.device(f"cuda:{current}"))
    assert _resolved_device(torch.device("cuda")) != _resolved_device(torch.device(f"cuda:{1-current}"))
    assert _resolved_device(torch.device("cuda")) != _resolved_device(torch.device("cpu"))


def test_explicit_device_comparison_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("No implicit device to resolve"))
    assert _resolved_device(torch.device("cpu")) == torch.device("cpu")
    assert _resolved_device(torch.device("cuda:0")) == torch.device("cuda:0")
    assert _resolved_device(torch.device("cuda:1")) != _resolved_device(torch.device("cuda:0"))


def trainer(loaded, **kwargs):
    config = TrainingConfig(round1_updates=1, total_updates=2, learning_rate=0.001,
                            ppo_epochs=2, minibatch_size=1)
    return PPOTrainer(actor_for(deepcopy(loaded)), Reward(), config,
                      experiment_id="tiny-v1", reward_id="proxy-cal1", **kwargs)


def test_actual_update_changes_lora_and_value_but_not_base(loaded):
    run = trainer(loaded)
    before = {n: p.detach().clone() for n, p in run.actor.named_parameters()}
    metrics = run.update(prompts(), rollout_seed=7)
    assert metrics.update == 1 and metrics.prompt_position == 2
    changed = [n for n, p in run.actor.named_parameters() if not torch.equal(p, before[n])]
    assert any("lora_" in n for n in changed)
    assert any(n.startswith("value_head.") for n in changed)
    assert all("lora_" in n or n.startswith("value_head.") for n in changed)
    assert metrics.library_metrics["objective/kl"] == pytest.approx(0., abs=1e-6)
    assert all(p.grad is None for p in run.actor.parameters())


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_checkpoint_resume_matches_uninterrupted_training(loaded, tmp_path):
    full = trainer(loaded)
    full_metrics = full.train([prompts(), prompts()], [7, 8])
    paused = trainer(loaded)
    paused.train([prompts(), prompts()], [7, 8], checkpoint_dir=tmp_path, until_update=1)
    path = tmp_path / "update_000001.pt"
    saved_torch, saved_python = torch.get_rng_state(), random.getstate()
    resumed = trainer(loaded, seed=999)
    torch.manual_seed(999)
    random.seed(999)
    resumed.load_checkpoint(path)
    assert torch.equal(torch.get_rng_state(), saved_torch)
    assert random.getstate() == saved_python
    assert resumed.update_count == 1 and resumed.prompt_position == 2
    result = resumed.train([prompts(), prompts()], [7, 8])
    assert result == full_metrics[1:]
    assert_state_equal(full.actor.state_dict(), resumed.actor.state_dict())
    assert_state_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_state_equal(full._backend.lr_scheduler.state_dict(), resumed._backend.lr_scheduler.state_dict())


def test_checkpoint_rejects_identity_mismatch_and_overwrite(loaded, tmp_path):
    run = trainer(loaded)
    path = run.save_checkpoint(tmp_path / "initial.pt")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        run.save_checkpoint(path)
    assert path.read_bytes() == before
    other = trainer(loaded)
    other.reward_id = "different-memory"
    with pytest.raises(PPOError, match="identity differs"):
        other.load_checkpoint(path)
    assert not other._failed
    assert list(tmp_path.iterdir()) == [path]


def test_failed_update_cannot_continue_or_checkpoint(loaded, tmp_path, monkeypatch):
    run = trainer(loaded)
    batch = format_policy_batch(run.actor.tokenizer, prompts(), max_prompt_tokens=64,
                                max_new_tokens=4, context_window=128)
    run._initialize_backend(batch)
    initial = run.save_checkpoint(tmp_path / "initial.pt")
    original = run._backend.train

    def fail_after_training():
        original()
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(run._backend, "train", fail_after_training)
    with pytest.raises(RuntimeError, match="interruption"):
        run.update(prompts(), rollout_seed=7)
    assert run.update_count == 0
    with pytest.raises(PPOError, match="failed"):
        run.save_checkpoint(tmp_path / "bad.pt")
    with pytest.raises(PPOError, match="failed"):
        run.update(prompts(), rollout_seed=7)
    run.load_checkpoint(initial)
    assert run.update(prompts(), rollout_seed=7).update == 1


def test_changed_schedule_is_rejected_on_resume(loaded, tmp_path):
    run = trainer(loaded)
    run.train([prompts(), prompts()], [7, 8], until_update=1)
    path = run.save_checkpoint(tmp_path / "one.pt")
    resumed = trainer(loaded)
    resumed.load_checkpoint(path)
    with pytest.raises(PPOError, match="schedule differs"):
        resumed.train([prompts(), prompts()], [7, 99])


def test_greedy_and_wrong_batch_size_are_rejected(loaded):
    with pytest.raises(PPOError, match="sampled"):
        PPOTrainer(actor_for(deepcopy(loaded), do_sample=False), Reward(), TrainingConfig(),
                   experiment_id="test", reward_id="test")
    run = trainer(loaded)
    with pytest.raises(PPOError, match="Prompt count"):
        run.update(prompts()[:1], rollout_seed=7)


def test_bad_reward_does_not_update_weights(loaded):
    run = trainer(loaded)
    class BadReward:
        def score(self, records, answers):
            return replace(Reward().score(records, answers), rewards=(float("nan"), 1.))
    run.reward = BadReward()
    before = deepcopy(run.actor.state_dict())
    with pytest.raises(ValueError, match="Nonfinite"):
        run.update(prompts(), rollout_seed=7)
    assert_state_equal(before, run.actor.state_dict())
    assert run.update_count == 0


def test_corrupt_parameter_rejected_before_modification(loaded, tmp_path):
    run = trainer(loaded)
    path = run.save_checkpoint(tmp_path / "good.pt")
    payload = torch.load(path, weights_only=True)
    payload["parameters"]["value_head.bias"].fill_(float("nan"))
    torch.save(payload, tmp_path / "bad.pt")
    before = deepcopy(run.actor.state_dict())
    with pytest.raises(PPOError, match="Invalid checkpoint parameter"):
        run.load_checkpoint(tmp_path / "bad.pt")
    assert_state_equal(before, run.actor.state_dict())


def test_real_trl_trainer_is_called_and_optimizer_has_unique_parameters(loaded, monkeypatch):
    from trl.experimental.ppo import PPOTrainer as LibraryTrainer
    original = LibraryTrainer.train
    calls = []

    def tracked(backend):
        calls.append(backend)
        return original(backend)

    monkeypatch.setattr(LibraryTrainer, "train", tracked)
    run = trainer(loaded)
    run.update(prompts(), rollout_seed=7)
    assert calls == [run._backend]
    parameters = [p for group in run.optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in run.actor.parameters() if p.requires_grad}


def test_legacy_checkpoint_is_explicitly_rejected(loaded, tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": 1}, path)
    with pytest.raises(PPOError, match="schema"):
        trainer(loaded).load_checkpoint(path)


def test_explicit_reward_fork_preserves_weights_optimizer_and_progress(loaded, tmp_path):
    source = trainer(loaded)
    source.update(prompts(), rollout_seed=7)
    path = source.save_checkpoint(tmp_path / "round1.pt")
    target = trainer(loaded)
    target.reward_id = "knn-new-memory"
    with pytest.raises(PPOError, match="identity differs"):
        target.load_checkpoint(path)
    with pytest.raises(PPOError, match="identity differs"):
        target.fork_checkpoint(path, expected_reward_id="wrong-source")
    target.fork_checkpoint(path, expected_reward_id=source.reward_id)
    assert target.reward_id == "knn-new-memory"
    assert target.update_count == source.update_count == 1
    assert target.prompt_position == source.prompt_position
    assert_state_equal(target.actor.state_dict(), source.actor.state_dict())
    assert_state_equal(target.optimizer.state_dict(), source.optimizer.state_dict())
    assert target.forked_from["reward_id"] == source.reward_id


def test_rolling_checkpoint_replaces_only_after_success(loaded, tmp_path, monkeypatch):
    run = trainer(loaded)
    path = run.save_checkpoint(tmp_path / "latest.pt")
    original = path.read_bytes()
    original_save = torch.save
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        run.save_checkpoint(path, replace_existing=True)
    assert path.read_bytes() == original
    monkeypatch.setattr(torch, "save", original_save)
    run.update(prompts(), rollout_seed=7)
    run.save_checkpoint(path, replace_existing=True)
    assert torch.load(path, weights_only=True)["update"] == 1
    assert list(tmp_path.iterdir()) == [path]


def test_proxy_and_knn_rewards_reach_real_trl(loaded):
    from reward_gap.calibration import FrozenCalibration, ScoreScale
    from reward_gap.memory import GapMemory, MemoryContext
    from reward_gap.rewards import ProxyReward, KNNReward
    from reward_gap.scorers import ScoreBatch

    class Proxy:
        role = "proxy"
        def score(self, records, answers, *, return_embeddings=False):
            assert len(records) == len(answers)
            assert all(isinstance(answer, str) for answer in answers)
            return ScoreBatch(tuple(r.prompt_id for r in records), (14., 8.), (10, 20),
                              "proxy", "proxy", "v1", torch.eye(2) if return_embeddings else None,
                              "pool" if return_embeddings else None)

    scale = ScoreScale("proxy", "v1", 10., 2.)
    calibration = FrozenCalibration("cal1", scale, scale)
    memory = GapMemory(["a", "b"], torch.eye(2), [0.5, -2.],
                       context=MemoryContext("proxy", "v1", "pool", "cal1"), k=1)
    for strategy, expected in [(ProxyReward(Proxy(), calibration), (2., -1.)),
                               (KNNReward(Proxy(), calibration, memory), (1.5, 1.))]:
        run = trainer(loaded)
        run.reward = strategy
        result = run.update(prompts(), rollout_seed=7)
        assert run._bridge.batches[0].rewards == expected
        assert result.library_metrics["objective/scores"] == pytest.approx(sum(expected) / 2)


def test_reward_bridge_preserves_prompts_and_decodes_eos_padding(loaded):
    from reward_gap._trl_bridge import RewardBridge, RewardModel
    from trl.experimental.utils import get_reward
    actor = actor_for(loaded)
    records = prompts()
    batch = format_policy_batch(actor.tokenizer, records, max_prompt_tokens=64,
                                max_new_tokens=4, context_window=128)
    calls = []
    class InspectReward(Reward):
        def score(self, records, answers):
            calls.append((tuple(records), tuple(answers)))
            return super().score(records, answers)
    bridge = RewardBridge(actor.tokenizer, InspectReward())
    bridge.bind(records, batch)
    token = actor.tokenizer.encode("a", add_special_tokens=False)[0]
    eos, pad = actor.tokenizer.eos_token_id, actor.tokenizer.pad_token_id
    suffix = torch.tensor([[eos, pad, pad], [token, token, eos]])
    sequences = torch.cat((batch.input_ids, suffix), 1)
    _, scores, _ = get_reward(RewardModel(bridge), sequences, pad, batch.input_ids.shape[1])
    assert calls == [(tuple(records), ("", "aa"))]
    torch.testing.assert_close(scores, torch.tensor([1., 1.2]))
