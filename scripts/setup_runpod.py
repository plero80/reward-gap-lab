"""Reuse the pod's existing GPU stack; install project libraries on local disk.

Run on the pod: python scripts/setup_runpod.py
The old project/.venv-runpod directory is never changed or deleted.
"""

import argparse
import configparser
import json
import os
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_ENV = Path("/tmp/reward-gap-lab-venv")


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def gpu_package(name):
    return canonical(name).startswith(("torch", "triton", "nvidia-", "cuda-"))


GPU_PROBE = r"""
import importlib.metadata as metadata
import json
import re
import torch
from pip._vendor.packaging.requirements import Requirement

if not torch.cuda.is_available():
    raise RuntimeError('Existing PyTorch cannot use CUDA; no GPU packages will be installed')
assert (torch.ones(3, device='cuda:0') * 2).tolist() == [2., 2., 2.]
packages = {}
for dist in metadata.distributions():
    name = re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()
    if name.startswith(('torch', 'triton', 'nvidia-', 'cuda-')) and name not in packages:
        packages[name] = dist.version
for name in packages:
    for text in metadata.requires(name) or []:
        requirement = Requirement(text)
        dependency = re.sub(r'[-_.]+', '-', requirement.name).lower()
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        if dependency.startswith(('torch', 'triton', 'nvidia-', 'cuda-')):
            installed = packages.get(dependency)
            if installed is None or not requirement.specifier.contains(installed, prereleases=True):
                raise RuntimeError(f'Existing GPU stack has an unsatisfied dependency: {name} needs {text}; refusing downloads')
print(json.dumps({'torch': torch.__version__, 'torch_file': torch.__file__,
                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                  'packages': packages}))
"""

IMPORT_CHECK = """
import peft
from transformers import BloomPreTrainedModel
from reward_gap.policy import PPOActor
from reward_gap.ppo import PPOTrainer
from reward_gap.scorers import RewardScorer
from reward_gap.gsm8k.teachers import TeacherExperiment
import matplotlib
print('Project imports passed')
"""


def check_environment(folder, base_python):
    """Never reuse a partially installed stack from the old isolated venv."""
    python = folder / "bin" / "python"
    config = configparser.ConfigParser()
    settings = folder / "pyvenv.cfg"
    if not settings.is_file() or not python.is_file():
        raise ValueError(f"Incomplete environment at {folder}; choose a new --env-dir")
    config.read_string("[venv]\n" + settings.read_text())
    if config.get("venv", "include-system-site-packages", fallback="false").lower() != "true":
        raise ValueError("Environment does not reuse system packages; choose a new --env-dir")
    executable = config.get("venv", "executable", fallback="")
    if not executable or Path(executable).resolve() != base_python.resolve():
        raise ValueError("Environment belongs to a different Python installation; choose a new --env-dir")


def checked_plan(report):
    """Install only explicit non-GPU wheels, with --no-deps in the install step."""
    wheels = []
    for item in report["install"]:
        name = item["metadata"]["name"]
        if gpu_package(name):
            raise ValueError(f"Dependency resolution tried to install {name}; keeping the GPU stack unchanged")
        if canonical(name) == "reward-gap-lab":
            continue
        info = item["download_info"]
        url = info["url"]
        if not url.split("?", 1)[0].endswith(".whl"):
            raise ValueError(f"Expected a binary wheel for {name}")
        digest = info.get("archive_info", {}).get("hashes", {}).get("sha256")
        if not digest:
            raise ValueError(f"Missing wheel hash for {name}")
        wheels.append(f"{name} @ {url}#sha256={digest}")
    return wheels


def assert_same_gpu(before, after):
    if (before["packages"] != after["packages"]
            or before["torch_file"] != after["torch_file"]):
        raise ValueError("Environment shadows the pod GPU packages; choose a new --env-dir")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV,
                        help="Environment on local disk (default: /tmp/reward-gap-lab-venv)")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info < (3, 12):
        parser.error("Run on the Linux GPU pod with Python 3.12 or newer")
    root = Path(__file__).resolve().parents[1]
    folder = args.env_dir.resolve()
    if folder == (root / ".venv-runpod").resolve():
        parser.error("Use a new directory; the previous installer may still be writing .venv-runpod")
    base_python = Path(getattr(sys, "_base_executable", sys.executable))
    python = folder / "bin" / "python"
    child_env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        child_env.pop(key, None)
    child_env["PYTHONNOUSERSITE"] = "1"

    def run(executable, *arguments, capture=False):
        return subprocess.run([str(executable), *arguments], cwd=root, env=child_env,
                              check=True, text=True, capture_output=capture)

    def probe(executable):
        return json.loads(run(executable, "-c", GPU_PROBE, capture=True).stdout)

    def pip(*arguments):
        return run(python, "-m", "pip", "--isolated", "--require-virtualenv", *arguments)

    try:
        print("Checking the pod's existing PyTorch/CUDA before downloading anything...", flush=True)
        original = probe(base_python)
        print(json.dumps(original, indent=2), flush=True)
        if not folder.exists():
            run(base_python, "-m", "venv", "--system-site-packages", str(folder))
        check_environment(folder, base_python)
        assert_same_gpu(original, probe(python))

        evidence = root / "outputs" / "setup"
        evidence.mkdir(parents=True, exist_ok=True)
        constraints = folder / "gpu-constraints.txt"
        constraints.write_text("".join(f"{n}=={v}\n" for n, v in sorted(original["packages"].items())))
        plan = folder / "install-plan.json"
        # Installed GPU packages are constrained to their existing versions.
        # Resolve first; never let the installation step resolve extra packages.
        print("Resolving project libraries while holding existing GPU packages fixed...", flush=True)
        pip("install", "--dry-run", "--report", str(plan), "--only-binary=:all:",
            "--no-build-isolation", "--constraint", str(constraints), "-e", ".[research,test]")
        wheels = checked_plan(json.loads(plan.read_text()))
        if wheels:
            requirements = folder / "approved-wheels.txt"
            requirements.write_text("\n".join(wheels) + "\n")
            pip("install", "--no-deps", "--only-binary=:all:", "-r", str(requirements))
        pip("install", "--no-deps", "--no-build-isolation", "-e", ".[research,test]")
        assert_same_gpu(original, probe(python))
        pip("check")
        run(python, "-c", IMPORT_CHECK)
        (evidence / "environment.txt").write_text(
            run(python, "-m", "pip", "freeze", capture=True).stdout, encoding="utf-8")
        (evidence / "gpu.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        parser.exit(exc.returncode, "Setup failed; no GPU-package replacement is permitted. Read the error above.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Setup failed: {exc}\n")
    print(f"Setup passed. Activate in each new terminal:\n  source {folder}/bin/activate")
    print("The environment uses local temporary storage; recreate it after a pod restart if missing.")


if __name__ == "__main__":
    main()
