# How to run the experiments

This guide uses the current repository commands and presets. Run the commands
in a **Linux GPU terminal**, such as your Runpod terminal, from the project root.
Run one command at a time and wait for it to succeed before continuing.

## 1. Choose the experiment

An **arm** is one training condition. For example, Proxy PPO and Judge-4B PPO
train separate copies of the same starting policy using different rewards.
The Base condition is the policy before PPO.

| Experiment | Question | Conditions |
| --- | --- | --- |
| RQ1: HH-RLHF gap prediction | Can memory predict proxy/judge disagreement before and after PPO? | Frozen gap predictors evaluated on initial, Proxy PPO and corrected PPO answers |
| Experiment 2: GSM8K | Does correcting the proxy improve math answers? | Base, Proxy PPO, Judge-4B PPO, kNN-4B PPO |
| Experiment 2 extension: GSM8K teachers | Does a 30B teacher produce a more useful correction memory than a 4B teacher? | Base, Proxy PPO, Judge-4B PPO, kNN-4B PPO, kNN-30B PPO |
| Original HH-RLHF two-round experiment | Does refreshing memory help continued PPO? | Proxy, static-memory and refreshed-memory final policies |

Each experiment can run independently. The 30B extension does not require a
completed standard GSM8K run. Start with the smoke preset for the experiment
you want, then move to its larger preset. Smoke runs check execution; their
small samples and training budgets are not enough for research conclusions.

## 2. Set up the GPU environment once

The project must already be copied or cloned onto the GPU machine, including
the new teacher-comparison files if you want that experiment.

```bash
cd /workspace/reward-gap-lab
python --version
```

Use Python 3.12 or newer. Check the pod's GPU stack and set up project libraries:

```bash
python scripts/setup_runpod.py
```

Continue only after `Setup passed`. The script includes research/plotting and
test dependencies. Activate the environment:

```bash
source /tmp/reward-gap-lab-venv/bin/activate
```

Setup reuses the existing PyTorch/CUDA and places additional libraries on local
disk, avoiding large package extraction on `/workspace` network storage. It
checks the GPU before dependency resolution, constrains installed GPU versions,
and refuses an installation plan containing GPU packages. If the existing stack
is incompatible, it stops rather than replacing it. The actual installation
uses the checked non-GPU wheels with dependency installation disabled.
No experiment datasets or model weights are downloaded during setup.
Do not copy your Windows `.venv` to Linux.

In **each new terminal**, run these two lines again:

```bash
cd /workspace/reward-gap-lab
source /tmp/reward-gap-lab-venv/bin/activate
```

Use persistent storage for the checkout, `data/`, `model_cache/` and `outputs/`.
Verify that your pod's `/workspace` is backed by the storage you intend to keep.
The environment under `/tmp` may disappear after a pod restart; rerun setup if
missing. An older `.venv-runpod` directory is left untouched. If the old installer
is still running, stop it with Ctrl+C in its original terminal before using the
updated script. Do not rerun the old script or activate that incomplete environment.
If setup fails while resolving dependencies, share that error before changing
GPU libraries. The package now accepts PyTorch >=2.8,<3; only 2.14 has been used
for local CPU tests, so the pod's inherited version must pass the runtime checks.

## 3. Understand prepare, preflight and run

- **Prepare:** download dataset files if allowed, partition them, and save the
  fixed experiment inputs. It does not train a policy.
- **Preflight:** validate inputs and check model inference on the GPU. It can
  download model weights because the GPU presets allow downloads. It does not
  perform a PPO update or prove that the full training workload fits.
- **Run:** perform the experiment and save its progress, answers and results.
  Calibration and initial memory construction happen here, before PPO.

Prepare each directory only once. If it is already completely prepared with
matching settings, skip preparation and run preflight. Preparation refuses to
overwrite an existing directory. If settings differ, select a new prepared
directory in a copied config; do not mix partitions from different settings.

| Presets | Shared prepared directory | Manifest |
| --- | --- | --- |
| `smoke_gpu.json` | `data/prepared/smoke/` | `input_manifest.json` |
| `followup.json` | `data/prepared/followup/` | `input_manifest.json` |
| `rq1_ppo_gpu.json` | `data/prepared/rq1-ppo/` | `input_manifest.json` |
| `rq1_gpu.json` | `data/prepared/rq1/` | `input_manifest.json` |
| `gsm8k_smoke.json`, `gsm8k_teachers_smoke.json` | `data/prepared/gsm8k-smoke/` | `manifest.json` |
| All GSM8K pilot/full presets, including teachers | `data/prepared/gsm8k-followup/` | `manifest.json` |

Sharing prepared questions does not share a trained policy or a run directory.
For example, standard GSM8K and the teacher comparison build their own run
artifacts from those questions.

## 4. RQ1: can memory predict disagreement?

This uses the HH-RLHF dataset, a Qwen2.5-0.5B policy, a Skywork 0.6B proxy reward
model, and a Skywork 4B judge reward model.

`--config` chooses models, data and the training budget. `--plan` chooses the
RQ1 analysis. The default `rq1.json` trains both Proxy PPO and corrected PPO,
and evaluates the original frozen predictors before PPO, after update 1, and
after the last update. It does not refresh the memory.

### Smoke: four updates per PPO arm, seed 42

Skip the first command if the matching smoke data is already prepared:

```bash
python -m reward_gap.cli prepare --config configs/smoke_gpu.json --download
python -m reward_gap.cli preflight --config configs/smoke_gpu.json
python -m reward_gap.cli rq1 --config configs/smoke_gpu.json --plan configs/rq1.json --run-name rq1-smoke-01
```

### Larger run: 100 updates per PPO arm, seed 42

```bash
python -m reward_gap.cli prepare --config configs/rq1_ppo_gpu.json --download
python -m reward_gap.cli preflight --config configs/rq1_ppo_gpu.json
python -m reward_gap.cli rq1 --config configs/rq1_ppo_gpu.json --plan configs/rq1.json --run-name rq1-ppo-01
```

This requests at least 128 calibration prompts, 512 memory/predictor-training
prompts, 256 validation prompts, 512 final-evaluation prompts and 1,024 PPO
training prompts. Each PPO update uses eight prompts. Whole conversation groups
stay together, so actual cohort counts may exceed the requested minimums.

Read progress:

```bash
python -m reward_gap.cli status --config configs/rq1_ppo_gpu.json --run-name rq1-ppo-01
```

Read `outputs/rq1-ppo-01/report.md`, `metrics.csv`, `summary.json` and the PNG/PDF
plots. They compare predicted gaps with actual proxy/judge gaps across policies.
Only two final policy checkpoints remain per run:
`seed-42/rq1-proxy/final.pt` and `seed-42/rq1-corrected/final.pt`.

### Optional: initial-policy-only analysis

This skips PPO, so it cannot answer what happens after training:

```bash
python -m reward_gap.cli prepare --config configs/rq1_gpu.json --download
python -m reward_gap.cli preflight --config configs/rq1_gpu.json
python -m reward_gap.cli rq1 --config configs/rq1_gpu.json --plan configs/rq1_initial.json --run-name rq1-initial-01
```

### Optional: analyze existing two-round checkpoints

Finish the original two-round experiment in section 7 first. The supplied
`configs/rq1_shift.json` points to checkpoints under `outputs/smoke-01/`.
If your source run has another name, edit those paths before starting.
Keep the source run's `inputs.json`; the analysis checks for data overlap.
The base model revisions and adapter settings must match the source run.

Prepare and preflight `configs/rq1_gpu.json` as above, then run:

```bash
python -m reward_gap.cli rq1 --config configs/rq1_gpu.json --plan configs/rq1_shift.json --run-name rq1-shift-01
```

This analyzes initial and saved trained policies without running new PPO.

## 5. Experiment 2: standard GSM8K comparison

This uses `openai/gsm8k`, a Qwen2.5-0.5B policy, a Qwen2.5-1.5B proxy grader,
and a Qwen3-4B-Instruct-2507 judge grader. These graders generate scores using
the question, reference solution and policy answer. They are different models
from the Skywork reward models used in RQ1.

The conditions are Base, Proxy PPO, Judge-4B PPO and kNN-4B PPO. The memory is
fixed during PPO. Judge PPO trains the small policy; the judge stays frozen.

### Smoke: two updates per arm, seed 42

```bash
python -m reward_gap.cli gsm8k-prepare --config configs/gsm8k_smoke.json --download
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_smoke.json
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_smoke.json --run-name gsm8k-smoke-01
```

The smoke preset uses two questions with two responses each per PPO update.
It evaluates development questions, with official-test evaluation disabled.

### Pilot: 100 updates per arm, seed 42

Prepare the larger partitions once, then check and run:

```bash
python -m reward_gap.cli gsm8k-prepare --config configs/gsm8k_pilot.json --download
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_pilot.json
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_pilot.json --run-name gsm8k-pilot-01
```

Inspect `outputs/gsm8k-pilot-01/report.md`, `metrics.csv` and
`monitor-seed-42.png`. The pilot checks progress every 25 updates and does not
evaluate the official test set. Use development results to choose settings
before the full run; the supplied format/length penalties of 0.5 are candidates.

### Full: 400 updates per arm, seeds 42, 43 and 44

Reuse the pilot's prepared directory. If you skipped the pilot, prepare it once
with `gsm8k-prepare --config configs/gsm8k_full.json --download` first.

```bash
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_full.json
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_full.json --run-name gsm8k-full-01
```

Each arm starts fresh for each seed and uses eight questions with two responses
each per update. This is a new experiment, not continuation from pilot weights.
After all training, it generates one greedy answer per official test question.
The main outcome is numeric-match rate; reports also include formatting,
unresolved answers, length limits and grader diagnostics.

Read progress:

```bash
python -m reward_gap.cli gsm8k-status --config configs/gsm8k_full.json --run-name gsm8k-full-01
```

Results are under `outputs/gsm8k-full-01/`. Three final policy checkpoints
remain per seed, under `seed-<seed>/proxy/`, `judge/` and `knn/` as `final.pt`.

## 6. GSM8K extension: compare 4B and 30B teachers

These presets automatically include **all five conditions**:
Base, Proxy PPO, Judge-4B PPO, kNN-4B PPO and kNN-30B PPO.
There is no extra flag to enable Judge-4B PPO.

The extra teacher is `Qwen/Qwen3-30B-A3B-Instruct-2507`. Both teachers label the
same initial-policy answers. Both memories use the same proxy embeddings and
fixed retrieval settings (`k=32`, temperature `0.05`), with separate frozen
teacher normalization. The 30B teacher is used for preparation; it is not
trained, and this experiment does not include direct Judge-30B PPO.

The code loads the teachers sequentially during preparation. Judge-4B PPO
loads the 4B judge again during its training stage. Check actual GPU capacity
with this experiment's preflight and smoke run; success on the standard GSM8K
experiment does not establish that the 30B teacher fits.

### Smoke: two updates per arm, seed 42

Skip preparation if the standard GSM8K smoke data is already prepared:

```bash
python -m reward_gap.cli gsm8k-prepare --config configs/gsm8k_teachers_smoke.json --download
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_teachers_smoke.json
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_teachers_smoke.json --run-name teachers-smoke-01
```

Official-test evaluation is disabled in this preset.

### Pilot: 100 updates per arm, seed 42

Skip preparation if `data/prepared/gsm8k-followup/` already contains the matching
standard GSM8K pilot/full preparation:

```bash
python -m reward_gap.cli gsm8k-prepare --config configs/gsm8k_teachers_pilot.json --download
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_teachers_pilot.json
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_teachers_pilot.json --run-name teachers-pilot-01 --until preparation
```

The last command deliberately pauses after labeling and memory construction.
Inspect these files in `outputs/teachers-pilot-01/`:

- `teacher_checks.json`: teacher grading and initial gap-prediction diagnostics.
- `blinded_review.json`: answers and anonymous teacher grades for your review.
- `review_key.json`: the teacher identities; keep it separate from blinded review.

Check failed grading/retries, wrong numeric answers receiving high grades, and
reasoning disagreements before interpreting teacher quality. Review is manual;
the CLI does not automatically apply your notes to training. To continue PPO:

```bash
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_teachers_pilot.json --run-name teachers-pilot-01
```

The pilot does not evaluate the official test set. The preparation pause is
optional: omit `--until preparation` to run preparation and training together.

### Full: 400 updates per arm, seeds 42, 43 and 44

Reuse the pilot's prepared questions. If not prepared yet, run
`gsm8k-prepare --config configs/gsm8k_teachers_full.json --download` first.

```bash
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_teachers_full.json
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_teachers_full.json --run-name teachers-full-01
```

All PPO arms use identical starting trainable weights, prompt schedules and
training budgets within each seed. Final evaluation uses the same numeric
answer checker for every condition. The primary comparison is kNN-30B minus
kNN-4B numeric-match rate, paired within each seed. Actual teacher grades for
trained-policy evaluation answers are not computed in this comparison.

Read progress:

```bash
python -m reward_gap.cli gsm8k-status --config configs/gsm8k_teachers_full.json --run-name teachers-full-01
```

Read `report.md`, `summary.json`, `metrics.csv` and `teacher-monitor-42.png`
(also 43/44 and PDF) in `outputs/teachers-full-01/`. Four final checkpoints
remain per seed, under `proxy/`, `judge4/`, `knn4/` and `knn30/` as `final.pt`.

Use a new run name if an older teacher-comparison run omitted Judge-4B PPO.
The expanded comparison uses protocol `gsm8k_matched_teacher_labels_judge4_v2`.

## 7. Original HH-RLHF two-round experiment

This is the coordinator that tests memory refresh. It is separate from RQ1
and the GSM8K experiments.

### Smoke: two updates per round, seed 42

```bash
python -m reward_gap.cli prepare --config configs/smoke_gpu.json --download
python -m reward_gap.cli preflight --config configs/smoke_gpu.json
python -m reward_gap.cli run --config configs/smoke_gpu.json --run-name smoke-01
```

Skip preparation if you already prepared these inputs for the RQ1 smoke run.
The coordinator builds calibration and M0, trains round 1, extends the memory
to M1 using refresh answers, and trains the second-round branches. Final
evaluation compares proxy, static-memory and refreshed-memory policies.

```bash
python -m reward_gap.cli status --config configs/smoke_gpu.json --run-name smoke-01
```

Read `outputs/smoke-01/summary.json` and the evaluation rows/manifests it links.
Completed runs retain `seed-42/corrected_round1.pt` and final checkpoints under
`seed-42/raw/`, `seed-42/static/` and `seed-42/iterative/`.

### Full: 200 updates per round, seeds 42, 43 and 44

Use `configs/followup.json`. It starts fresh policies and prepares a separate
larger dataset; it does not continue from smoke-run weights or reuse smoke
partitions. The raw dataset cache and model cache can be reused.

In your activated GPU environment (inside `tmux` if you will disconnect), run
each command separately and proceed only after it succeeds:

```bash
python -m reward_gap.cli prepare --config configs/followup.json --download
python -m reward_gap.cli preflight --config configs/followup.json
python -m reward_gap.cli run --config configs/followup.json --run-name followup-full-01
```

Skip preparation only if `data/prepared/followup/` already contains the matching
completed preparation. Full preparation also saves a 400-update training
schedule for each of the three seeds.

| Setting | Full preset |
| --- | --- |
| Training seeds | 42, 43, 44 |
| Round 1 | 200 updates |
| Round 2 | 200 additional updates; each final policy has 400 total |
| Prompts per PPO update | 8 |
| Calibration prompts | At least 128 |
| Initial-memory prompts | At least 512 |
| Training prompt pool | At least 4,096 |
| Refresh prompts | At least 512 |
| Reserved validation prompts | At least 256 |
| Final-evaluation prompts | At least 512 |
| Recovery checkpoint interval | Every 10 updates; the rolling file is replaced |

Models, learning rate, KL coefficient and memory retrieval settings match the
smoke preset. Prompt/scoring limits are larger to accommodate longer HH-RLHF
conversations; overlength inputs still raise an error rather than being silently
truncated. These are starting workshop settings, not a demonstrated optimal
budget or a guarantee of a detectable effect. Run preflight and verify GPU
training before leaving the full experiment unattended.

Within this run, calibration and initial memory M0 are built once using the
first seed and shared across training seeds. Each seed creates its own M1 from
its corrected round-1 policy. The validation cohort is reserved; this coordinator
does not use it for automatic tuning or early stopping.

The three final conditions are:

- **Raw:** proxy rewards for all 400 updates.
- **Static:** M0 corrections for all 400 updates.
- **Iterative:** M0 corrections for the first 200 updates, then M1 corrections
  for the remaining 200.

Static and iterative share the same corrected round-1 checkpoint before they
split. All final evaluations wait until all seeds and branches finish training.

```bash
python -m reward_gap.cli status --config configs/followup.json --run-name followup-full-01
```

Read `outputs/followup-full-01/summary.json` for per-seed, per-branch metrics
and links to evaluation artifacts. The coordinator does not generate the GSM8K
plots or an automatic `report.md`. Completed runs retain four checkpoints per
seed (12 total): the shared corrected round-1 checkpoint and three final branch
checkpoints. They do not retain every intermediate policy.

Repeat the same run command to resume. Optionally launch with `--until round1`
to pause after the first round, or `--until training` to pause before final
evaluation; repeat without that option to continue.

## 8. Resume, pause and find your results

`--run-name` selects the output directory. For example, `teachers-full-01`
writes under `outputs/teachers-full-01/` with the supplied configs.

To resume after an interruption, activate the same environment and repeat
the **same run command**, with the same config, plan (for RQ1) and run name.
Completed stages are reused; training resumes from the saved checkpoint.
Work after the latest checkpoint may need to repeat. Do not launch two
processes into the same run directory.

Use a new run name for changed scientific settings, model revisions or inputs.
A pilot and a full run should have different names. Review development results
and freeze choices before using final-test results. For repeatable larger runs,
replace model/dataset `main` revisions with the resolved revisions saved in
development artifacts, before starting the new run.

The available deliberate stopping points are:

| Command | Option | Stops after |
| --- | --- | --- |
| `run` | `--until round1` | First training round |
| `run` | `--until training` | All training, before final evaluation |
| `gsm8k-run` with a teacher preset | `--until preparation` | Shared answers, teacher labels and memories |
| `gsm8k-run` | `--until training` | Training, before official-test evaluation when enabled |

To continue, repeat the command without `--until`. These options are selected
when launching; there is no separate live `pause` command. RQ1 has no `--until`
option. A GSM8K pilot can be `completed` without official-test results because
its config sets `evaluate_test=false`.

Check `status.json` for `running`, `paused`, `failed` or `completed`. A running
stage may take a while during generation or grading. A failed status includes
the failing stage/error. Reports may be incomplete until the run returns.

Keep the run directory and its prepared inputs together for inspection/resume.
Checkpoints contain trainable policy/value-head and recovery state; reconstructing
the policy also needs its original frozen base model. Keep large data, model
caches and run outputs in artifact storage outside Git.

## 9. If a command fails

| Symptom | Next step |
| --- | --- |
| `No module named reward_gap` or a missing dependency | Rerun the updated setup script, then activate `/tmp/reward-gap-lab-venv` as in section 2. |
| Config path cannot be found | Run `cd /workspace/reward-gap-lab` before the command. |
| Prepared directory already exists | Reuse it if settings match. Preflight checks it; do not run preparation over it. |
| Prepared inputs differ from config | Select a new prepared directory for the changed settings and prepare it. |
| Prompt exceeds the configured token limit | Inspect the reported length and generation/scoring settings. The code refuses silent truncation. Use suitable limits within the model context window, then rerun preflight. |
| CUDA or out-of-memory error | Read the setup/preflight error first. Verify the GPU/environment and workload size; a successful preflight alone does not test PPO peak memory. |
| Run protocol/configuration changed | Use a fresh run name for the new experiment; preserve the old results. |

If a grader returns `SCORE: 3` followed by its explanation, use the current
grading parser (`gsm8k_score_boundary_complete_v2`). It accepts one standalone
`SCORE: N` line, with an integer from 1 to 5, at either the beginning or end of
a completed response. It rejects multiple score fields and retries responses
that hit their token limit without completing. The rubric itself is unchanged.
Logs/cache entries record `grade_format`, and generation attempts record
`finish_reason`. Scores are never inferred from the numerical answer.

Runs created before this parser change require a fresh run name. After updating
the source files on the pod, reuse your installed environment and prepared data:

```bash
python -m reward_gap.cli gsm8k-preflight --config configs/gsm8k_smoke.json
```

Only after it passes:

```bash
python -m reward_gap.cli gsm8k-run --config configs/gsm8k_smoke.json --run-name gsm8k-smoke-02
```

CPU tests use tiny local models. The commands here describe the implemented
GPU workflow; this guide does not claim that the full production runs have
already passed on your GPU.

### Isolated missing answers or grades (GSM8K)

Both GSM8K runners use failure policy `gsm8k_skip_unusable_samples_v1`:

- Malformed or truncated grades retry with the configured grading token budgets.
  A grading timeout also consumes one of those attempts. If none succeeds, the
  grade stays missing; it is never replaced with zero or a guessed score.
- A recoverable generation failure retries the affected batch as individual
  questions once. Persistently missing generations are recorded. A normal EOS
  with empty answer text is a real response: the numeric checker marks it unresolved.
- Calibration and memory use valid paired labels. The 4B/30B comparison keeps
  the same intersection of examples for both teachers. Their `fit/.../coverage.json`
  files list exclusions. Fitting still requires enough usable, nonconstant scores
  and enough neighbors for the chosen memory settings.
- Evaluation retains every scheduled question. Missing generations count as
  unresolved/incorrect. Missing grades do not remove valid numeric answers from
  the accuracy denominator. Grader/gap statistics use valid pairs and report
  `graded_count` and `failed_count`; unavailable statistics are JSON `null`.
- During PPO, an ungradable response skips its **whole scheduled batch before
  optimization**. No policy/value weights or optimizer state are updated. The
  next scheduled batch continues. This preserves TRL's fixed batch requirements.
  `metrics.json` marks `skipped`, with `mean_reward: null`, and `summary.json`
  reports `scheduled_batches`, `optimized_batches`, and `skipped_batches` per arm.
  Plot/update numbers therefore describe scheduled batches, not successful updates.

Failures are saved in `outputs/<run-name>/sample_failures.jsonl`; raw grader
attempts remain in `grading_cost.jsonl`. Skips may differ between arms, so check
their optimized batch counts before interpreting a comparison. Recovery uses
the existing checkpoint retention policy and adds no per-prompt checkpoints.

CUDA/OOM errors, broken files, invalid configuration and programming errors still
stop the run. Five consecutive unusable PPO batches stop after saving a recovery
checkpoint, and a run with no successful PPO optimization cannot report success.
This prevents a broken grader from consuming the entire budget silently.

Update the project source on the pod, keep the existing environment/data/model
cache, and rerun preflight. Runs started before this failure policy need a fresh
run name (for example `gsm8k-smoke-recovery-01`); do not mix old and new results.
