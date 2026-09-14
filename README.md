# Reward Gap Lab

Experiments testing whether a k-nearest-neighbor reward-gap memory
can improve language-model training with PPO.

A frozen proxy model provides rewards. A stronger frozen judge
labels disagreements. A memory of these disagreements predicts
corrections to the proxy reward.

The judge is a reference model, not human ground truth.

For setup and copyable commands for every experiment, see
[How to run the experiments](RUN_EXPERIMENTS.md).

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
source /tmp/reward-gap-lab-venv/bin/activate
```

Continue only if setup reports success. It checks the existing PyTorch/CUDA first,
then creates a Linux environment on local disk that inherits the pod's packages.
It installs the project, research and test libraries while preserving the existing
GPU stack. It pins the installed GPU package versions during dependency resolution
and rejects plans that would install torch, torchvision, torchaudio, Triton or
CUDA/NVIDIA packages. Actual wheel installation uses `--no-deps` after checking
the plan. Missing/broken GPU dependencies cause a failure, not an automatic repair.
Do not copy the Windows `.venv` to the pod.

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

Activate `/tmp/reward-gap-lab-venv` again in each new terminal. Repeating setup
reuses it when present. Local temporary storage may be lost when the pod restarts;
rerun setup if missing. The old checkout's `.venv-runpod` is not reused or deleted.
No model weights or datasets are downloaded by setup;
preparation and preflight perform those downloads according to their settings.

### Model-loading dependencies and tests

To install the tested model/training libraries and run all tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,training]"
.\.venv\Scripts\python.exe -m pytest -q
```

For Runpod, use the setup script above to preserve the existing GPU build.
The package accepts PyTorch >=2.8,<3; the local loader checks were run with CPU PyTorch 2.14.0 and
Transformers 5.17.0. PEFT 0.20.0 supplies the LoRA adapter.
The optional `models` dependency group records those tested
versions for Transformers/PEFT; model tests skip when optional libraries are absent.
GPU execution with the pod's inherited PyTorch still needs setup/preflight and a smoke run.
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

On the GPU machine, after the setup checks pass:

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

For the larger two-round experiment, use `configs/followup.json`: three training
seeds (42/43/44), 200 updates per round, eight prompts per update, and separate
prepared cohorts under `data/prepared/followup/`. Prepare these inputs once,
then preflight and run with a fresh name:

```bash
python -m reward_gap.cli prepare --config configs/followup.json --download
python -m reward_gap.cli preflight --config configs/followup.json
python -m reward_gap.cli run --config configs/followup.json --run-name followup-full-01
```

These are workshop starting settings; actual GPU training remains to be checked.
See [the full-run instructions](RUN_EXPERIMENTS.md#full-200-updates-per-round-seeds-42-43-and-44)
for cohort sizes, shared calibration/M0, status and checkpoint retention.

## RQ1: Can memory predict disagreement?

The `rq1` command implements section 3 of the workshop plan. It compares kNN,
zero gap, training-mean gap, a linear ridge gap regressor, and a linear judge-score
student on the same held-out answers and frozen proxy embeddings. The default
plan now trains proxy PPO and corrected PPO from identical initial weights,
then checks whether the original frozen predictors still work on their answers.
All learned predictors share the same initial training labels. Validation selects
ridge regularization and F1 detector cutoffs before any final labels.

In the activated Runpod environment, install the plotting extra:

```bash
python scripts/setup_runpod.py
source /tmp/reward-gap-lab-venv/bin/activate
```

For a quick check using the already-prepared smoke cohorts (four PPO updates
per branch, evaluation after update 1 and update 4):

```bash
python -m reward_gap.cli rq1 --config configs/smoke_gpu.json --plan configs/rq1.json --run-name rq1-smoke-01
```

For a 100-update exploratory experiment, prepare the separate RQ1 PPO cohorts once,
then run preflight and, once it passes, the prediction experiment:

```bash
python -m reward_gap.cli prepare --config configs/rq1_ppo_gpu.json --download
python -m reward_gap.cli preflight --config configs/rq1_ppo_gpu.json
python -m reward_gap.cli rq1 --config configs/rq1_ppo_gpu.json --plan configs/rq1.json --run-name rq1-ppo-01
```

The RQ1 configuration requests at least 128 calibration, 512 predictor-training,
256 validation, and 512 final-test prompts, using one answer per prompt. These
are starting workshop settings, not a power calculation. Long prompts raise an
explicit error rather than being silently truncated. The new preset also requests
1,024 training prompts and trains each branch for 100 updates of eight prompts.
The `refresh` cohort remains unused: RQ1 tests the original memory throughout.
Each run uses one seed; repeat with separate seeds/configurations to assess
run variability. These budgets do not guarantee a measurable distribution shift.

The default `configs/rq1.json` enables `train_ppo` and evaluates after update 1
and after `training.total_updates`. Add increasing update numbers to
`ppo_evaluation_updates` for extra measurement points. Every point uses the same
held-out prompts and generation seeds; final-test scores never change the reward
or stopping budget. Only two final checkpoints remain, under `seed-42/rq1-proxy/`
and `seed-42/rq1-corrected/`. Intermediate answers, predictions and plots remain
available. Repeating the command resumes interrupted training and evaluation
without rebuilding the memory. Use a new run name for this expanded protocol.

`configs/rq1_initial.json` preserves the initial-policy-only check and can still
use `configs/rq1_gpu.json` or an existing smoke preparation.

To reuse existing checkpoints instead of training policies, finish the PPO source run. Edit the paths
in `configs/rq1_shift.json` if needed, then use a fresh RQ1 run:

```bash
python -m reward_gap.cli rq1 --config configs/rq1_gpu.json --plan configs/rq1_shift.json --run-name rq1-shift-01
```

That plan evaluates the initial policy, corrected round-1 checkpoint, and the
proxy/static/refreshed final policies from `outputs/smoke-01`. Keep the source
run's `inputs.json` with its checkpoints so RQ1 can verify test conversations
were excluded from PPO development/training. Base model revisions and adapter
settings must match. Use the same RQ1 generation settings across every policy.
The command checks all requested checkpoint paths before generating labels.

Default high-gap labels use `g > theta`, with theta fixed to the nonnegative
95th percentile of calibration gaps. Set an explicit nonnegative `theta` in
the plan to override that rule. Normalization, predictors, and detector cutoffs
remain fixed across policy checkpoints. `answers_per_prompt` in the plan can
increase sampling; related answers always remain in their original cohort.

Under `outputs/<run-name>/`, read `report.md`, `metrics.csv`, and `summary.json`.
PNG/PDF figures show predicted versus actual gaps and performance across
checkpoints. The table reports MAE, RMSE, R2, AUROC, AP, precision, recall, and
positive prevalence. Undefined metrics are explicitly marked. Saved stage
artifacts include answers, embeddings, predictors, memory, and per-method
predictions. Completed stages are reused after failure; changing the plan or
inputs requires a new run name. The external-checkpoint plan creates no extra
policy checkpoints; integrated PPO retains only its two final checkpoints.

An initial-only run does not test PPO distribution shift. A high AUROC alone
does not demonstrate accurate numerical correction, and disagreement with the
judge is not independently verified reward hacking. The ridge student is a
small frozen-feature baseline, not a fine-tuned reward language model.

## Experiment 2: GSM8K policy improvement

This implements the follow-up described in `README_GSM8K.md`: Base, Proxy PPO,
Judge PPO, and static kNN PPO. It uses Qwen2.5-1.5B-Instruct and
Qwen3-4B-Instruct-2507 as frozen reference-aware language-model graders.
These are separate from the Skywork models in the HH-RLHF experiments.

After the Runpod setup script succeeds (it includes `.[research,test]`), run the
GSM8K smoke setup (prepare only once for each prepared directory):

```bash
python -m reward_gap.cli gsm8k-prepare --config configs/gsm8k_smoke.json --download
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_smoke.json
```

After preflight passes:

```bash
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_smoke.json --run-name gsm8k-smoke-01
```

The smoke run checks two PPO updates per arm. It cannot test the reported
formatting decline around update 75. Next prepare the full-sized development
cohorts and run the 100-update pilot:

```bash
python -m reward_gap.cli gsm8k-prepare --config configs/gsm8k_pilot.json --download
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_pilot.json --run-name gsm8k-pilot-01
```

Inspect `outputs/gsm8k-pilot-01/report.md`, `metrics.csv`, and the monitor plots.
The pilot disables official-test evaluation. Shared format/completion penalties
start at 0.5 each; these are development candidates, not established best values.
Select them using development results, then freeze them in the full configuration
and pin model/dataset revisions from the pilot's saved artifacts before running:

```bash
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_full.json --run-name gsm8k-full-01
```

The full configuration reuses the pilot's fixed partitions, runs seeds 42/43/44
with fresh calibration and memory per seed, and trains each arm for 400 updates.
Each update has eight questions with two responses each. Final evaluation uses
one greedy answer per official test question after all training is finished.
The primary endpoint is numeric-match rate, alongside strict accuracy, format,
unresolved answers, truncation, judge/proxy grades, actual high gaps, and PPO KL.
The new numeric checker is versioned independently from the historical post-hoc
checker; no historical result is claimed for this code.

Use `gsm8k-status --config ... --run-name ...` to read progress. Repeating the
same run resumes completed stages and the latest recovery checkpoint. `--until
training` stops before official-test evaluation. Only three final checkpoints
per seed remain; monitor answers and metrics are saved without retaining every
intermediate policy. Configuration changes require a new run name.

## GSM8K: compare 4B and 30B memory teachers

The `gsm8k_teachers_*` presets implement Section 11 of `README_GSM8K_extra.md`.
They compare Base, Proxy PPO, Judge-4B PPO, kNN PPO with 4B labels, and kNN PPO with
`Qwen/Qwen3-30B-A3B-Instruct-2507` labels. The original GSM8K presets still run
their existing Proxy/Judge/kNN comparison.

The two teachers grade the exact same saved initial-policy answers. Both memories
reuse the same proxy grades, proxy normalization, embeddings and question IDs.
Each teacher gets its own frozen normalization and gap labels. Both memories use
fixed k=32 and temperature=0.05; this primary comparison does not tune retrieval
separately. These retrieval values come from the earlier exploratory experiment.

The 30B checkpoint has roughly 61 GB of BF16 weights before inference overhead;
it is loaded sequentially with the 4B teacher after releasing the policy/proxy.
Both teachers are unloaded for static kNN training and numeric evaluation.
Judge-4B PPO loads the frozen 4B judge during its own training stage and uses
its normalized grade, followed by the same format/completion penalties. It trains
the same 0.5B policy from the same initial weights and with the same PPO budget.
Actual GPU capacity and grading reliability need the new preflight and smoke run.
See the [official model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).

For the smoke comparison, prepare once (skip this command if the shared
`data/prepared/gsm8k-smoke` directory is already prepared), then run preflight:

```bash
python -m reward_gap.cli gsm8k-prepare --config configs/gsm8k_teachers_smoke.json --download
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_teachers_smoke.json
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_teachers_smoke.json --run-name teachers-smoke-01 --until preparation
```

Preparation writes `teacher_checks.json` (grading and gap-prediction diagnostics),
`blinded_review.json`, and `review_key.json`. Keep the key away from reviewers.
Inspect invalid-grade/retry counts, wrong numeric answers graded >=4, unresolved
cases, and reasoning disagreements. The grader rubric/parser must be validated
on development data before claiming a teacher-quality result. Then continue:

```bash
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_teachers_smoke.json --run-name teachers-smoke-01
```

`gsm8k_teachers_pilot.json` runs 100 updates per arm with official-test evaluation
disabled. `gsm8k_teachers_full.json` runs 400 updates per arm for seeds 42/43/44,
followed by official-test evaluation. These two presets share
`data/prepared/gsm8k-followup`, which must be prepared once using either preset.
Fix common penalty strengths on development data and pin model/dataset revisions
before the full comparison. The supplied 0.5/0.5 penalties remain development
candidates. Changed protocol choices require a new run name.

The primary result is the paired numeric-match difference between kNN–30B and
kNN–4B, with per-seed results and final means/sample standard deviations. The
report also includes format, strict correctness, unresolved answers, truncation,
training KL and grading cost. Final answers use one common numeric checker;
teacher grades of trained-policy answers are not computed in this primary run.
Shared initial monitor answers are graded by both teachers before PPO for
development diagnostics. Predicted gaps are not reported as actual teacher gaps.

Read `outputs/<run-name>/report.md`, `summary.json`, `metrics.csv`, and the
`teacher-monitor-<seed>.png`/PDF plots. Only four final policy checkpoints per
seed remain. Cached teacher labels are reused after interruption, and reviewer
notes are preserved. Use a new run name if an earlier teacher-comparison run used
the three-arm protocol; the new Judge-4B comparison records protocol v2.
Direct-30B-judge PPO, independently tuned retrieval, memory
refresh and learned students are separate optional extensions.

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
