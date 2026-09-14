# Reward Gap Lab

Experiments testing whether a k-nearest-neighbor reward-gap memory
can improve language-model training with PPO.

A frozen proxy model provides rewards. A stronger frozen judge
labels disagreements. A memory of these disagreements predicts
corrections to the proxy reward.

The judge is a reference model, not human ground truth.

## Status

Under construction. Configuration validation, atomic JSON saving, HH-RLHF
prompt preparation, basic model loading, policy/reward formatting, and frozen
Qwen3 reward scoring, a Qwen2 LoRA actor with a value head, calibrated reward
strategies, a TRL-backed PPO trainer, evaluation, refresh and the two-round
experiment coordinator are implemented. Tiny-model end-to-end execution is
tested; production GPU validation, plots and human review remain pending.

## Development setup

Use Python 3.12 or newer, matching the tested training environment and its pinned NumPy dependency.

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

### Runpod setup

Keep the checkout under `/workspace/reward-gap-lab` on your attached persistent
storage. Use Python 3.12 or newer. In the pod terminal, run:

```bash
cd /workspace/reward-gap-lab
python scripts/setup_runpod.py
source .venv-runpod/bin/activate
```

Continue only if setup reports success. This creates a separate Linux environment
and installs `.[training,test]` from `pyproject.toml`. The separate environment is
needed because the template's PyTorch, torchvision, and torchaudio versions may
conflict with our pinned PyTorch version. It does not inherit or uninstall the
template's libraries. It will download its own PyTorch build and dependencies;
allow disk space for them. Do not copy the Windows `.venv` to the pod.

Setup checks dependency compatibility, project imports, and a CUDA calculation,
and saves the installed package list to `outputs/setup/environment.txt`. A CUDA
failure needs a compatible PyTorch build and host GPU driver before proceeding;
the script does not install drivers. GPU training still needs the smoke test.

After activation, prepare data once if it is not already present, then preflight:

```bash
python -m reward_gap.cli prepare --config configs/smoke_gpu.json --download
python -m reward_gap.cli preflight --config configs/smoke_gpu.json
```

Only after preflight passes:

```bash
python -m reward_gap.cli run --config configs/smoke_gpu.json --run-name workshop-01
```

Activate `.venv-runpod` again in each new terminal. Repeating the setup script
reuses this environment. No model weights or datasets are downloaded by setup;
preparation and preflight perform those downloads according to their settings.

### Model-loading dependencies and tests

To install the tested model/training libraries and run all tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,training]"
.\.venv\Scripts\python.exe -m pytest -q
```

For GPU experiments, install a CUDA-compatible PyTorch build for the target
machine. The local loader checks were run with CPU PyTorch 2.14.0 and
Transformers 5.17.0. PEFT 0.20.0 supplies the LoRA adapter.
The optional `models` dependency group records those tested
versions; model tests skip when the optional libraries are absent.
The `training` extra adds pinned TRL 0.29.1, Accelerate, Datasets and NumPy.
TRL's PPO API is experimental, so upgrades require rerunning integration tests.

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
start frozen; `PPOActor` adds trainable LoRA and value-head layers to the policy.

Skywork's model card specifies no system prompts for reward chat formatting
and recommends staying within 16,384 tokens. These requirements belong in the
upcoming formatting/scoring implementation. Full checkpoint loading and GPU
execution have not yet been verified here.

### Format inputs

`format_policy_batch()` prepares prompts for generation;
`format_reward_batch()` prepares prompts plus generated answers for one scorer.
Supply each model's own tokenizer. Both return CPU token tensors and masks,
reject overlength inputs, and leave moving tensors to the caller.

After loading `config` and `policy` as above:

```python
from reward_gap.data import load_prompts
from reward_gap.formatting import format_policy_batch

prompts = load_prompts(config.data.prepared_dir / "training.json")
batch = format_policy_batch(
    policy.tokenizer, prompts[:2],
    max_prompt_tokens=config.generation.max_prompt_tokens,
    max_new_tokens=config.generation.max_new_tokens,
    context_window=policy.model.config.max_position_embeddings,
)
inputs = batch.to(policy.model.device).model_inputs()
```

This produces model inputs without generating an answer. A prompt exceeding
the configured limits raises an error; the formatter does not shorten it silently.

### Score generated answers

On the configured model device, use one scorer for the proxy and another for
the judge. `RewardScorer.load()` follows the same cache and download settings
as model loading:

```python
from reward_gap.scorers import RewardScorer

proxy = RewardScorer.load(config, "proxy")
# Supply one actual generated answer string for each prompt in this batch.
result = proxy.score(prompt_batch, generated_answers, return_embeddings=True)
scores = result.scores
embeddings = result.embeddings
```

Scores are raw scalar outputs, without sigmoid or calibration. The optional
proxy embeddings are unit-length vectors on CPU, pooled from the final hidden
layer at the last non-padding position. The judge returns scores only.
`scoring.batch_size` limits the number of answers processed in one forward pass;
it is separate from the PPO rollout batch size. Scoring does not update weights
or save results automatically.

### Generate PPO rollouts with the actor

After installing the `models` extra in the target environment:

```python
import torch
from reward_gap.config import load_config
from reward_gap.data import load_prompts
from reward_gap.policy import PPOActor

config = load_config("configs/smoke_gpu.json")
actor = PPOActor.load(config, seed=42)
prompts = load_prompts(config.data.prepared_dir / "training.json")
rollout = actor.generate(prompts[:2], seed=42)

with torch.no_grad():
    old_statistics = actor.statistics(rollout)
reference_log_probs = actor.reference_log_probs(rollout)
answers = rollout.answers
```

This generates answers and computes token statistics, but performs no PPO
update. `policy` settings specify the LoRA rank, scale, and projection modules.
`generation.do_sample=true` uses the full softmax at temperature 1 so sampled
and recomputed probabilities match. Greedy generation is available for
evaluation by setting it to false; the future PPO trainer must require sampled
rollouts. The trainer must also advance rollout seeds between updates.

The rollout retains original token IDs, attention/response masks, EOS status,
and length-limit status. Statistics are aligned to each generated token:
its log probability and the value of the state before that token. Calling
`statistics()` outside `torch.no_grad()` allows gradients for a future update.
The reference call temporarily disables LoRA and never records gradients.
Adapter/value-head checkpoint restoration and optimizer state are handled by
the TRL integration in ppo.py.

## Project layout

- `src/reward_gap/`: reusable experiment code.
- `configs/`: experiment settings.
- `notebooks/`: controls and exploration.
- `tests/`: small behavioral tests.
- `docs/`: architecture, protocol, and experiment summaries.
- `scripts/`: maintenance tools.
- `data/`, `model_cache/`, `outputs/`: local artifacts excluded from Git.

See [the planned architecture](docs/architecture.md).

`reward_gap.memory.GapMemory` now builds signed gap memories from proxy
embeddings and already-normalized proxy/judge scores. It predicts gaps with
cosine nearest neighbors and temperature weights, returns neighbor diagnostics,
and creates a new snapshot when examples are appended. Save snapshots under
`outputs/` using distinct filenames such as `M0.json` and `M1.json`; existing
files cannot be overwritten. Loading and querying require a matching
`MemoryContext` (encoder, revision, pooling and calibration identity).
`FrozenCalibration` fits and saves proxy/judge score normalization constants
on a dedicated calibration cohort. `ProxyReward` returns the normalized proxy
score; `KNNReward` subtracts the memory's predicted signed gap and returns
neighbor diagnostics. Both use the frozen proxy without calling the judge
during reward calculation. Calibration-cohort orchestration, configuration/CLI
wiring remain under construction.

`reward_gap.ppo.PPOTrainer` delegates generation, KL penalties, advantages,
clipped losses and optimizer updates to Hugging Face TRL. Our adapters connect
the existing LoRA actor/value head and ProxyReward/KNNReward strategies to its
model-based API. No custom PPO loss or GAE implementation remains.

The wrapper runs one TRL rollout/update at a time while retaining its optimizer,
which preserves the project's fixed prompt/seed schedule and update-boundary
checkpoints. Call `trainer.update(prompts, rollout_seed=...)` or use `train()`
for a complete schedule. TRL controls EOS handling, masks and advantage
normalization; its defaults differ from the former custom trainer. Training
requires distinct PAD/primary EOS tokens, at least two prompts per rollout, a
batch size divisible by minibatch_size, and normalize_advantages=true.

Checkpoints contain LoRA/value weights, optimizer/scheduler state, progress and
RNG information. Reconstruct the same frozen model and reward artifacts before
loading. The new format rejects old custom-PPO checkpoints. Checkpoints refuse
overwrite by default; the coordinator explicitly replaces only rolling recovery
files. CPU tiny-model training and exact resume are tested. The coordinator
supplies end-to-end orchestration; production GPU execution remains pending.

`evaluation.evaluate()` reads a prepared validation or final-evaluation cohort,
generates answers and saves proxy/judge scores, normalized gaps and generation
details. Memory predictions and saved embeddings are optional. Evaluation never
appends examples to memory.

`refresh.refresh_memory()` reads only the prepared refresh cohort, checks its
disjointness from all other cohorts, generates and labels answers, and saves a
new extended memory without changing the parent or training policy weights.
Both operations require absolute paths from the resolved configuration and a
new output directory. They save evidence rows, a manifest and status.json;
consumers must require state=completed before using results. A failed stage must
be rerun into a new directory.

### Run the two-round experiment

After preparing data and installing `.[training]` in your GPU environment,
check the setup first:

```powershell
python -m reward_gap.cli preflight --config configs/smoke_gpu.json
```

Preflight checks prepared cohorts and schedules, CUDA availability, all policy
prompt lengths, and generation/scoring with the actual models. It generates
one training batch with at most 16 answer tokens, performs no PPO update, and
saves a separate `outputs/preflight-*/report.json` on success or failure.
Download permission follows `runtime.allow_downloads`. Passing verifies
inference; the smoke run still needs to verify PPO and its peak memory use.

Once preflight passes, run:

```powershell
python -m reward_gap.cli run --config configs/smoke_gpu.json --run-name workshop-01
```

The coordinator fits or loads calibration and M0, runs proxy-only and static
round 1, refreshes M1, then runs raw/static/refreshed round 2. It evaluates all
final policies only after all training finishes and writes `summary.json` under
the configured output root. Repeating the command resumes compatible saved
progress; changed settings or prepared inputs require a new run name.

Use `--until round1` or `--until training` to pause at those boundaries. There
is one rolling recovery checkpoint per branch, replaced at checkpoint_every;
completed runs retain only the shared corrected round-1 checkpoint and three
final branch checkpoints per seed. Full frozen model weights are not duplicated.
Stage attempts and evidence remain available for inspection. The coordinator
runs branches sequentially and releases trainer resources between stages.

Read progress with `python -m reward_gap.cli status --config configs/smoke_gpu.json
--run-name workshop-01` (on one line). After reinstalling the editable package,
`reward-gap` is also available in place of `python -m reward_gap.cli`.
The CLI also exposes `prepare --config ... --download`.

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
