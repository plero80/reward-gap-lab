"""Check GSM8K inference and completion-aware reward integration before PPO."""

import tempfile
from pathlib import Path

import torch

from reward_gap.artifacts import atomic_write_json
from reward_gap.formatting import format_policy_batch
from reward_gap.gsm8k.experiment import GSMExperiment


def preflight(config):
    root = config.base.runtime.output_root
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="gsm8k-preflight-", dir=root))
    report = {"state": "running", "config": config.to_dict()}
    try:
        if config.base.runtime.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment")
        experiment = GSMExperiment(config, folder)
        experiment._open_gsm()
        actor = experiment._actor(config.base.seeds[0])
        for cohort in experiment.cohorts.values():
            for question in cohort:
                format_policy_batch(actor.tokenizer, [question.prompt()], max_prompt_tokens=config.base.generation.max_prompt_tokens,
                                    max_new_tokens=config.base.generation.max_new_tokens, context_window=actor.context_window)
        experiment.proxy, experiment.judge = experiment._scorer("proxy"), experiment._scorer("judge")
        experiment.cohorts["calibration"] = experiment.cohorts["calibration"][:2]
        experiment._phase("preflight")
        data = experiment._labels(actor, "calibration", config.base.seeds[0], 1)
        report.update(state="passed", examples=data["rows"], models=experiment.status["models"],
                      note="Inference passed; PPO optimization and peak training memory still require the smoke run.")
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        atomic_write_json(folder / "report.json", report)
        raise ValueError(f"GSM8K preflight failed: {exc}; report: {folder / 'report.json'}") from exc
    return atomic_write_json(folder / "report.json", report)
