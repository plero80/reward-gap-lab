"""Installer protections without network access or changing the test environment."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("runpod_setup", Path(__file__).parents[1] / "scripts/setup_runpod.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def wheel(name):
    return {"metadata": {"name": name}, "download_info": {
        "url": f"https://example.invalid/{name}-1-py3-none-any.whl",
        "archive_info": {"hashes": {"sha256": "a" * 64}}}}


@pytest.mark.parametrize("name", ["torch", "torchvision", "torchaudio", "triton", "nvidia_cublas_cu12", "cuda-toolkit"])
def test_reject_gpu_wheels_before_installing_anything(name):
    with pytest.raises(ValueError, match="keeping the GPU stack unchanged"):
        setup.checked_plan({"install": [wheel("numpy"), wheel(name)]})


def test_approved_plan_pins_exact_wheel_and_digest():
    result = setup.checked_plan({"install": [wheel("numpy"), {"metadata": {"name": "reward-gap-lab"}}]})
    assert result == ["numpy @ https://example.invalid/numpy-1-py3-none-any.whl#sha256=" + "a" * 64]


@pytest.fixture
def installer(tmp_path, monkeypatch):
    folder = tmp_path / "local-env"
    base = tmp_path / "base-python"
    old = tmp_path / ".venv-runpod" / "installation-in-progress"
    old.parent.mkdir()
    old.write_text("leave this alone")
    snapshot = {"torch": "2.8.0+cu128", "torch_file": "/usr/local/lib/torch/__init__.py",
                "cuda": "12.8", "gpu": "test", "packages": {
                    "torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "triton": "3.4.0"}}
    state = SimpleNamespace(calls=[], plan={"install": [wheel("numpy")]}, probe_error=False)
    monkeypatch.setattr(setup, "__file__", str(tmp_path / "scripts/setup_runpod.py"))
    monkeypatch.setattr(setup.sys, "platform", "linux")
    monkeypatch.setattr(setup.sys, "_base_executable", str(base))
    monkeypatch.setattr(setup.sys, "argv", ["setup_runpod.py", "--env-dir", str(folder)])

    def run(command, **kwargs):
        state.calls.append(command)
        if setup.GPU_PROBE in command:
            if state.probe_error:
                raise subprocess.CalledProcessError(1, command, stderr="CUDA unavailable")
            return SimpleNamespace(stdout=json.dumps(snapshot))
        if "venv" in command:
            (folder / "bin").mkdir(parents=True)
            (folder / "bin/python").touch()
            (folder / "pyvenv.cfg").write_text(f"include-system-site-packages = true\nexecutable = {base}\n")
        if "--dry-run" in command:
            Path(command[command.index("--report") + 1]).write_text(json.dumps(state.plan))
        return SimpleNamespace(stdout="torch==2.8.0+cu128\n")

    monkeypatch.setattr(setup.subprocess, "run", run)
    return state, folder, old


def test_reuses_existing_gpu_installs_non_gpu_wheels_without_resolving_again(installer):
    state, folder, old = installer
    setup.main()
    assert setup.GPU_PROBE in state.calls[0]
    resolve = next(c for c in state.calls if "--dry-run" in c)
    assert "--constraint" in resolve and "--no-build-isolation" in resolve
    constraints = (folder / "gpu-constraints.txt").read_text()
    assert "torch==2.8.0+cu128\n" in constraints
    installs = [c for c in state.calls if "install" in c and "--dry-run" not in c]
    assert len(installs) == 2
    assert all("--no-deps" in c and "--require-virtualenv" in c for c in installs)
    assert old.read_text() == "leave this alone"


def test_incompatible_gpu_plan_stops_before_package_installation(installer):
    state, _, _ = installer
    state.plan = {"install": [wheel("torch")]}
    with pytest.raises(SystemExit) as error:
        setup.main()
    assert error.value.code == 1
    assert not any("install" in c and "--dry-run" not in c for c in state.calls)


def test_broken_existing_gpu_stops_before_creating_environment_or_using_pip(installer):
    state, folder, _ = installer
    state.probe_error = True
    with pytest.raises(SystemExit):
        setup.main()
    assert len(state.calls) == 1 and not folder.exists()


def test_refuses_existing_isolated_environment(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").touch()
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    with pytest.raises(ValueError, match="does not reuse"):
        setup.check_environment(tmp_path, Path("/usr/local/bin/python"))


def test_refuses_shadowed_torch_even_if_version_matches():
    before = {"packages": {"torch": "2.8.0"}, "torch_file": "/system/torch/__init__.py"}
    with pytest.raises(ValueError, match="shadows"):
        setup.assert_same_gpu(before, {**before, "torch_file": "/venv/torch/__init__.py"})
