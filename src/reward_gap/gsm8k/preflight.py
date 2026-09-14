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
        matched = "teacher_comparison" in config.settings
        if matched:
            from reward_gap.gsm8k.teachers import TeacherExperiment
            experiment = TeacherExperiment(config, folder)
        else:
            experiment = GSMExperiment(config, folder)
        experiment._open_gsm()
        actor = experiment._actor(config.base.seeds[0])
        for cohort in experiment.cohorts.values():
            for question in cohort:
                format_policy_batch(actor.tokenizer, [question.prompt()], max_prompt_tokens=config.base.generation.max_prompt_tokens,
                                    max_new_tokens=config.base.generation.max_new_tokens, context_window=actor.context_window)
        if matched:
            from reward_gap.gsm8k.teachers import TeacherExperiment, release_models
            assert isinstance(experiment, TeacherExperiment)
            experiment.proxy = experiment._scorer("proxy")
            experiment.cohorts["calibration"] = experiment.cohorts["calibration"][:2]
            shared = experiment._save_shared(folder, actor, "calibration", config.base.seeds[0])
            del actor
            del experiment.proxy
            release_models()
            checked = {}
            for name in ("4b", "30b"):
                teacher = experiment._teacher(name)
                try:
                    teacher.phase = "preflight"
                    checked[name] = experiment._grade_shared(folder / name, teacher, name, shared["shared"])
                    from reward_gap.gsm8k.experiment import read
                    grades = read(checked[name]["grades"])["rows"]
                    if not any(row["raw_judge"] is not None for row in grades):
                        raise ValueError(f"No usable {name} grades in preflight; see sample_failures.jsonl")
                finally:
                    del teacher
                    release_models()
            report.update(state="passed", teachers=checked, models=experiment.status["models"],
                          note="Both teacher inference paths passed sequentially; PPO still needs the smoke run.")
            return atomic_write_json(folder / "report.json", report)
        experiment.proxy, experiment.judge = experiment._scorer("proxy"), experiment._scorer("judge")
        experiment.cohorts["calibration"] = experiment.cohorts["calibration"][:2]
        experiment._phase("preflight")
        data = experiment._labels(actor, "calibration", config.base.seeds[0], 1)
        if not data["rows"]:
            raise ValueError("No usable paired grades in preflight; see sample_failures.jsonl")
        report.update(state="passed", examples=data["rows"], failed_examples=data["failed_rows"], models=experiment.status["models"],
                      note="Inference passed; PPO optimization and peak training memory still require the smoke run.")
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        atomic_write_json(folder / "report.json", report)
        raise ValueError(f"GSM8K preflight failed: {exc}; report: {folder / 'report.json'}") from exc
    return atomic_write_json(folder / "report.json", report)
