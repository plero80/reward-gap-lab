from copy import deepcopy
from dataclasses import replace
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from test_policy import loaded, actor_for, prompts
from reward_gap.config import TrainingConfig
from reward_gap.ppo import PPOError, PPOTrainer, compute_gae, ppo_loss
from reward_gap.rewards import RewardBatch


class Reward:
    def score(self, records, answers):
        values = tuple(1. + len(a) / 10 for a in answers)
        return RewardBatch(tuple(r.prompt_id for r in records), values, values, values,
                           (0.,) * len(values), (1,) * len(values), "proxy", "test-cal", "test-proxy", "v1")


def trainer(loaded, **kwargs):
    config = TrainingConfig(round1_updates=1, total_updates=2, learning_rate=0.001,
                            ppo_epochs=2, minibatch_size=1)
    return PPOTrainer(actor_for(deepcopy(loaded)), Reward(), config,
                      experiment_id="tiny-v1", reward_id="proxy-cal1", **kwargs)


def test_gae_terminal_reward_and_padding():
    rewards = torch.tensor([[0., 2., 999.], [0., 0., 3.]])
    values = torch.tensor([[0.5, 1., 888.], [1., 1., 1.]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    advantages, returns = compute_gae(rewards, values, mask, gamma=1., gae_lambda=1.)
    torch.testing.assert_close(returns, torch.tensor([[2., 2., 0.], [3., 3., 3.]]))
    torch.testing.assert_close(advantages, torch.tensor([[1.5, 1., 0.], [2., 2., 2.]]))
    # With lambda=0, each advantage is the one-step TD residual.
    advantages, _ = compute_gae(rewards, values, mask, gamma=0.5, gae_lambda=0.)
    torch.testing.assert_close(advantages, torch.tensor([[0., 1., 0.], [-0.5, -0.5, 2.]]))


def test_clipping_and_masked_loss_gradients():
    logp = torch.tensor([[2., 0.5, float("nan")]]).log().requires_grad_()
    values = torch.tensor([[2., 0., float("nan")]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    zeros = torch.zeros(1, 3)
    advantage = torch.tensor([[1., -1., 77.]])
    returns = torch.tensor([[1., 1., 77.]])
    policy, critic, clipped = ppo_loss(logp, values, zeros, zeros, advantage, returns, mask, TrainingConfig())
    assert policy.item() == pytest.approx(-0.2)
    assert critic.item() == pytest.approx(0.5)
    assert clipped.item() == 1.
    (policy + critic).backward()
    assert logp.grad[0, 2] == 0
    assert values.grad[0, 2] == 0
    assert logp.grad[0, :2].tolist() == [0., 0.]


def test_actual_update_changes_lora_and_value_but_not_base(loaded):
    run = trainer(loaded)
    before = {n: p.detach().clone() for n, p in run.actor.named_parameters()}
    metrics = run.update(prompts(), rollout_seed=7)
    assert metrics.update == 1 and metrics.prompt_position == 2
    assert metrics.optimizer_steps == 4
    changed = [n for n, p in run.actor.named_parameters() if not torch.equal(p, before[n])]
    assert any("lora_" in n for n in changed)
    assert any(n.startswith("value_head.") for n in changed)
    assert all("lora_" in n or n.startswith("value_head.") for n in changed)
    assert metrics.mean_sampled_kl == pytest.approx(0., abs=1e-6)
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
    assert_state_equal(full._shuffle.get_state(), resumed._shuffle.get_state())


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
    initial = run.save_checkpoint(tmp_path / "initial.pt")
    original = run.optimizer.step
    calls = 0

    def fail_after_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        original(*args, **kwargs)
        if calls == 2:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(run.optimizer, "step", fail_after_step)
    with pytest.raises(RuntimeError, match="interruption"):
        run.update(prompts(), rollout_seed=7)
    assert run.update_count == 0
    with pytest.raises(PPOError, match="failed"):
        run.save_checkpoint(tmp_path / "bad.pt")
    with pytest.raises(PPOError, match="failed"):
        run.update(prompts(), rollout_seed=7)
    monkeypatch.setattr(run.optimizer, "step", original)
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
    with pytest.raises(PPOError, match="Nonfinite"):
        run.update(prompts(), rollout_seed=7)
    assert_state_equal(before, run.actor.state_dict())
    assert run.update_count == 0


def test_kl_penalty_and_terminal_answer_reward(loaded, monkeypatch):
    import reward_gap.ppo as module
    run = trainer(loaded)
    captured = {}
    original = compute_gae

    def inspect(rewards, values, mask, **kwargs):
        captured["rewards"] = rewards.clone()
        captured["mask"] = mask.clone()
        return original(rewards, values, mask, **kwargs)

    # Force a known sampled log-ratio on all response tokens.
    monkeypatch.setattr(run.actor, "reference_log_probs",
                        lambda rollout: run.actor.statistics(rollout).log_probs - 0.2)
    monkeypatch.setattr(module, "compute_gae", inspect)
    rollout = run.actor.generate(prompts(), seed=7)
    answer_rewards = Reward().score(prompts(), rollout.answers).rewards
    run.update(prompts(), rollout_seed=7)
    mask = captured["mask"]
    expected = torch.zeros_like(captured["rewards"])
    expected[mask] = -run.config.kl_coefficient * 0.2
    for i, length in enumerate(rollout.response_lengths):
        expected[i, length - 1] += answer_rewards[i]
    torch.testing.assert_close(captured["rewards"], expected)


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
