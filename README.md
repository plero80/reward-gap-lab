# Reward Gap Lab

Experiments testing whether a k-nearest-neighbor reward-gap memory
can improve language-model training with PPO.

A frozen proxy model provides rewards. A stronger frozen judge
labels disagreements. A memory of these disagreements predicts
corrections to the proxy reward.

The judge is a reference model, not human ground truth.

## Status

Under construction. Configuration validation, atomic JSON saving, HH-RLHF
prompt preparation, and basic model loading are implemented. Actual experiment
checkpoint execution on the GPU, scoring, LoRA/value-head integration, and
training are still pending.

## Development setup

Use Python 3.11 or newer.

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Verify the package can be imported:

```powershell
.\.venv\Scripts\python.exe -c "import reward_gap; print(reward_gap.__file__)"
```

Once tests exist, run them with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Full experiments will target a Linux GPU environment.
Small correctness tests should work on CPU.

### Model-loading dependencies and tests

To install the tested model-library versions and run all tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,models]"
.\.venv\Scripts\python.exe -m pytest -q
```

For GPU experiments, install a CUDA-compatible PyTorch build for the target
machine. The local loader checks were run with CPU PyTorch 2.14.0 and
Transformers 5.17.0. The optional `models` dependency group records those tested
versions; model tests skip when the optional libraries are absent.

The model tests generate tiny checkpoints locally and do not download pretrained
models. `reward_gap.models` supports decoder-only policy checkpoints and scalar
sequence-classification reward checkpoints. Pass model sources and loading
options explicitly; checkpoint-specific formatting and scoring need separate
validation once the experiment models are selected.

### Selected experiment models

| Role | Checkpoint |
|---|---|
| Policy | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Proxy | [Skywork/Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B) |
| Judge | [Skywork/Skywork-Reward-V2-Qwen3-4B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B) |

The choices are recorded in `configs/smoke.json` and `configs/smoke_gpu.json`.
The CPU configuration keeps downloads disabled. The GPU configuration uses
`cuda:0`, BF16, and enabled downloads for the planned RTX PRO 6000 environment.
Both select the same prepared smoke cohorts. This is a loading configuration,
not an implemented training command.

On the GPU machine, after installing a compatible CUDA PyTorch build:

```python
from reward_gap.config import load_config
from reward_gap.models import load_experiment_model

config = load_config("configs/smoke_gpu.json")
policy = load_experiment_model(config, "policy")
# Load "proxy" or "judge" in the same way when needed.
```

Calling the loader can download model weights into `model_cache/`. Merely
loading the JSON downloads nothing. The base policy and both reward models
start frozen; trainable LoRA and value-head layers will be added later.

Skywork's model card specifies no system prompts for reward chat formatting
and recommends staying within 16,384 tokens. These requirements belong in the
upcoming formatting/scoring implementation. Full checkpoint loading and GPU
execution have not yet been verified here.

## Project layout

- `src/reward_gap/`: reusable experiment code.
- `configs/`: experiment settings.
- `notebooks/`: controls and exploration.
- `tests/`: small behavioral tests.
- `docs/`: architecture, protocol, and experiment summaries.
- `scripts/`: maintenance tools.
- `data/`, `model_cache/`, `outputs/`: local artifacts excluded from Git.

See [the planned architecture](docs/architecture.md).

## Prepare HH-RLHF data

From the project root, allow the first dataset download explicitly:

```powershell
.\.venv\Scripts\python.exe -m reward_gap.data --config configs/smoke.json --download
```

This reads all four HH-RLHF preference subsets, creates six separate prompt
cohorts, and saves their manifest and training schedule under
`data/prepared/smoke/`. It refuses to overwrite an existing prepared directory.
Choose another `data.prepared_dir` in the configuration for a new preparation.

See [data preparation details](docs/data.md) for the split rules, counts,
offline use, and how to read the saved prompts. No extra dependencies or
model downloads are needed for data preparation.
