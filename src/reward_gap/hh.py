"""One HH-RLHF experiment: two PPO rounds plus paired gap-predictor analysis."""

import gc
from pathlib import Path

import torch
from filelock import FileLock

from reward_gap.artifacts import atomic_write_json
from reward_gap.calibration import FrozenCalibration
from reward_gap.experiment import ExperimentError, ExperimentResult, FollowupExperiment, _read
from reward_gap.hh_analysis import HHAnalysis, PROTOCOL
from reward_gap.memory import GapMemory, MemoryContext


class HHExperiment(FollowupExperiment):
    retain_analysis_checkpoints = True

    def _open(self):
        protocol = self.run_dir / "hh_protocol.json"
        if (self.run_dir / "status.json").exists() and (not protocol.exists() or _read(protocol) != PROTOCOL):
            raise ExperimentError("HH protocol changed or source is a legacy run; use hh-evaluate for old checkpoints")
        super()._open()
        atomic_write_json(protocol, PROTOCOL)

    def run(self, *, until="complete"):
        if until not in ("round1", "training", "complete"):
            raise ExperimentError("until must be round1, training or complete")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.run_dir / ".run.lock"), timeout=0):
            self._open()
            was_complete = self.status["state"] == "completed"
            try:
                if not was_complete:
                    self.status.update(state="running", error=None)
                    self._status()
                    proxy = self._scorer("proxy")
                    actor = self._actor(self.config.seeds[0])
                    fitted = self._stage("calibration", lambda folder: self._calibration(folder, actor, proxy))
                    calibration = FrozenCalibration.load(fitted["calibration"])
                    initial = self._stage("initial-memory", lambda folder: self._initial_memory(folder, actor, proxy, calibration))
                    memory = GapMemory.load(initial["memory"], context=MemoryContext(**initial["context"]))
                    del actor
                    for seed in self.config.seeds:
                        self._round1(seed, proxy, calibration, memory)
                    if until != "round1":
                        for seed in self.config.seeds:
                            self._round2(seed, proxy, calibration, memory)
                    del proxy, memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if until == "complete" or was_complete:
                    # Full final analysis follows training of every branch/seed.
                    analysis_dir = self.run_dir / "analysis"
                    with FileLock(str(self.run_dir / ".analysis.lock"), timeout=0):
                        summary = HHAnalysis(self, analysis_dir).run_analysis()
                    atomic_write_json(self.run_dir / "summary.json", summary)
                    for name in ("report.md", "report.html", "policy_metrics.csv", "predictor_metrics.csv",
                                 "policy_outcomes.png", "policy_outcomes.pdf", "memory_comparison.png", "memory_comparison.pdf"):
                        (self.run_dir / name).write_bytes((analysis_dir / name).read_bytes())
                    self.status.update(state="completed", current_stage=None)
                else:
                    self.status.update(state="paused", current_stage=None)
                self.status.pop("error", None)
                self._status()
            except BaseException as exc:
                self.status.update(state="failed", error=str(exc))
                self._status()
                raise
        return ExperimentResult(self.run_dir, self.run_dir / "status.json",
                                self.run_dir / "summary.json" if self.status["state"] == "completed" else None,
                                self.status["state"])


def evaluate_hh(config, source_dir, output_dir):
    """Inference only on retained unified or original two-round HH checkpoints."""
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
        raise ExperimentError("Use a separate output directory for evaluation-only analysis")
    if not (source_dir / "status.json").is_file():
        raise ExperimentError("Source HH run has no saved status")
    source = FollowupExperiment(config, source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(source_dir / ".run.lock"), timeout=0), FileLock(str(output_dir / ".run.lock"), timeout=0):
        source.status = _read(source_dir / "status.json")
        return HHAnalysis(source, output_dir).run_analysis()
