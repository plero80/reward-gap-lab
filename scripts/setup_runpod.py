"""Create an isolated Linux training environment and verify GPU execution.

Run on the pod: python scripts/setup_runpod.py
No model weights or datasets are downloaded by this script.
"""

import argparse
import configparser
import os
from pathlib import Path
import subprocess
import sys
import venv


GPU_CHECK = """
import json
import torch
from torch.version import cuda

print('PyTorch:', torch.__version__, 'CUDA build:', cuda, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable: check the installed PyTorch build and pod GPU driver')
result = (torch.ones(3, device='cuda:0') * 2).tolist()
assert result == [2., 2., 2.], result

import peft
from transformers import BloomPreTrainedModel
from reward_gap.policy import PPOActor
from reward_gap.ppo import PPOTrainer
from reward_gap.scorers import RewardScorer

print(json.dumps({'gpu': torch.cuda.get_device_name(0),
                  'cuda_calculation': result,
                  'project_imports': 'passed'}, indent=2))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if sys.platform != "linux" or sys.version_info < (3, 12):
        parser.error("Run this script on the Linux GPU pod with Python 3.12 or newer")

    root = Path(__file__).resolve().parents[1]
    environment = root / ".venv-runpod"
    python = environment / "bin" / "python"
    if environment.exists():
        configuration = configparser.ConfigParser()
        settings = environment / "pyvenv.cfg"
        if not settings.is_file() or not python.is_file():
            parser.error(f"Incomplete environment at {environment}; inspect it before retrying")
        configuration.read_string("[venv]\n" + settings.read_text())
        if configuration.get("venv", "include-system-site-packages", fallback="true").lower() != "false":
            parser.error("Existing .venv-runpod inherits system packages; an isolated environment is required")
    else:
        print(f"Creating {environment}", flush=True)
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)

    child_env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        child_env.pop(key, None)
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env["PIP_REQUIRE_VIRTUALENV"] = "true"

    def run(*args: str, capture: bool = False):
        return subprocess.run([str(python), *args], cwd=root, env=child_env,
                              check=True, text=True, capture_output=capture)

    try:
        # Keep version choices in pyproject.toml. The separate environment is
        # intentional: the template may contain a different torch/vision/audio set.
        run("-m", "pip", "install", "-e", ".[training,test]")
        run("-m", "pip", "check")
        evidence = root / "outputs" / "setup"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "environment.txt").write_text(
            run("-m", "pip", "freeze", capture=True).stdout, encoding="utf-8")
        run("-c", GPU_CHECK)
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, "Setup check failed. Resolve the error above before running preflight.\n")

    print("Setup passed. Activate the environment in your terminal:")
    print("  source .venv-runpod/bin/activate")
    print("Then run:")
    print("  python -m reward_gap.cli preflight --config configs/smoke_gpu.json")


if __name__ == "__main__":
    main()
