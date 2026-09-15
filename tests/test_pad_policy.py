"""A PAD-dominant real model must work in generation and native TRL PPO."""

from copy import deepcopy
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from test_policy import loaded, actor_for, prompts
from test_ppo import trainer
from reward_gap.failures import SampleError
from reward_gap.policy import PAD_LOGIT_MASK
from reward_gap.ppo import PPOError, load_policy_checkpoint


def prefer_pad(module, inputs, logits):
    # Reproduce the real failure, rather than suppressing PAD in the fixture.
    result = logits.clone()
    result[..., 0] = 50.
    return result


@pytest.mark.parametrize("sampled", [False, True])
def test_pad_dominant_model_generates_valid_answers_with_opt_in(loaded, sampled):
    assert not loaded.model.lm_head._forward_hooks
    loaded.model.lm_head.register_forward_hook(prefer_pad)
    loaded.model.generation_config.eos_token_id = [loaded.tokenizer.eos_token_id, loaded.tokenizer.pad_token_id]
    original = actor_for(deepcopy(loaded), do_sample=sampled, suppress_pad_token=False)
    with pytest.raises(SampleError, match="Unexpected PAD"):
        original.generate(prompts(), seed=7)
    fixed = actor_for(deepcopy(loaded), do_sample=sampled)
    rollout = fixed.generate(prompts(), seed=7)
    mask = rollout.response_mask[:, rollout.prompt_width:]
    assert not rollout.response_ids[mask].eq(fixed.tokenizer.pad_token_id).any()
    assert all(reason in ("eos", "length") for reason in rollout.finish_reasons)
    # The mask is applied to reference logits too, even with adapters disabled.
    torch.testing.assert_close(fixed.statistics(rollout).log_probs,
                               fixed.reference_log_probs(rollout), atol=1e-6, rtol=1e-6)


def test_masked_log_probs_equal_base_conditioned_on_nonpad_tokens(loaded):
    loaded.model.lm_head.register_forward_hook(prefer_pad)
    base = deepcopy(loaded.model).eval()
    actor = actor_for(loaded)
    rollout = actor.generate(prompts(), seed=123)
    inputs, active = actor._inputs(rollout)
    with torch.no_grad():
        raw = base(**inputs, use_cache=False, return_dict=True).logits[:, rollout.prompt_width - 1:-1].float()
        # The fixture PAD ID is zero. Compute the conditional distribution by
        # excluding that column, independently of the production mask value.
        nonpad = raw[..., 1:]
        ids = (rollout.response_ids - 1).clamp_min(0)
        expected = nonpad.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    actual = actor.statistics(rollout).log_probs
    torch.testing.assert_close(actual[active], expected[active], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("temperature", [.7, 1.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_ppo_masks_generation_reference_and_update_logits(loaded, temperature, dtype):
    loaded.model.to(dtype=dtype)
    loaded.model.lm_head.register_forward_hook(prefer_pad)
    run = trainer(loaded, temperature=temperature)
    try:
        metrics = run.update(prompts(), rollout_seed=7)
        assert metrics.update == 1 and not metrics.skipped
        if dtype == torch.float32:
            assert metrics.library_metrics["objective/kl"] == pytest.approx(0., abs=1e-5)
        assert metrics.library_metrics["policy/entropy_avg"] >= 0
        logged = run._backend.state.log_history[-1]
        assert all(math.isfinite(v) for v in logged.values() if isinstance(v, (int, float)))
        assert all(torch.isfinite(p).all() for p in run._parameters.values())
    finally:
        run.release()


def test_mask_is_in_checkpoint_identity_and_cannot_change_on_resume(loaded, tmp_path):
    source = trainer(loaded)
    path = source.save_checkpoint(tmp_path / "masked.pt")
    payload = torch.load(path, weights_only=True)
    assert payload["identity"]["pad_logit_mask"] == PAD_LOGIT_MASK
    assert payload["identity"]["generation"]["suppress_pad_token"] is True
    target = actor_for(deepcopy(loaded), suppress_pad_token=False)
    with pytest.raises(PPOError, match="differs"):
        load_policy_checkpoint(target, path)
    # An old unmasked checkpoint is also rejected by a masked actor.
    old = deepcopy(payload)
    old["identity"].pop("pad_logit_mask")
    old["identity"]["generation"].pop("suppress_pad_token")
    torch.save(old, tmp_path / "old.pt")
    with pytest.raises(PPOError, match="differs"):
        source.load_checkpoint(tmp_path / "old.pt")


def test_legacy_config_and_checkpoint_generation_identity_is_unchanged(loaded, tmp_path):
    from reward_gap.config import GenerationConfig, TrainingConfig
    from reward_gap.ppo import PPOTrainer
    from test_ppo import Reward
    generation = GenerationConfig()
    assert generation.to_dict() == {"max_prompt_tokens": 512, "max_new_tokens": 256, "do_sample": True}
    actor = actor_for(deepcopy(loaded), suppress_pad_token=False)
    config = TrainingConfig(round1_updates=1, total_updates=2, ppo_epochs=2, minibatch_size=1)
    old = PPOTrainer(actor, Reward(), config, experiment_id="old", reward_id="old")
    path = old.save_checkpoint(tmp_path / "old.pt")
    identity = torch.load(path, weights_only=True)["identity"]
    assert "pad_logit_mask" not in identity and "suppress_pad_token" not in identity["generation"]
    old.load_checkpoint(path)
