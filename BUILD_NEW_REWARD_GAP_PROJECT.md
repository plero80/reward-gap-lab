# Build a new reward-gap experiment project from scratch

This is a construction guide for a **new directory and a new GitHub repository**. It proposes a cleaner implementation of the same core idea: train a language-model policy with PPO, compare proxy reward with a stronger judge, and test whether refreshing a kNN reward-gap memory improves training.

The structure and APIs below are a proposed new design, not files that already exist in the current project. Following this guide requires implementing the components; the setup commands only create the initial directories and package marker. No repository has been initialized by writing this guide.

## 1. Understand what you are building

The experiment has three model roles:

1. **Policy:** generates answers. PPO trains its LoRA adapter and a separate value head.
2. **Proxy:** a frozen reward model that scores answers and provides embeddings.
3. **Judge:** a frozen, stronger reward model used to measure disagreement and label memory examples.

Using calibration constants fitted on a separate calibration cohort:

```text
z_proxy = (proxy_score - proxy_mean) / proxy_std
z_judge = (judge_score - judge_mean) / judge_std
true_gap = z_proxy - z_judge

memory example = (normalized proxy embedding, true_gap)
predicted_gap = weighted average of gaps from nearby memory examples
corrected_reward = z_proxy - predicted_gap
```

Positive predicted gaps reduce reward. Negative predicted gaps increase reward under the signed correction. A capped version limits the correction according to a separately specified rule.

The judge is a reference model, not human ground truth. Evaluate answer quality and completion alongside disagreement; retain a separate human-review stage.

## 2. Create a separate directory

On Windows, open PowerShell. Choose a parent folder outside the existing project. These commands use your Documents directory; change that parent if desired. `New-Item` deliberately fails if the destination already exists, so you do not mix projects.

```powershell
$projectParent = Join-Path $env:USERPROFILE 'Documents'
$newProject = Join-Path $projectParent 'reward-gap-lab'
New-Item -ItemType Directory -Path $newProject -ErrorAction Stop
Set-Location -LiteralPath $newProject

New-Item -ItemType Directory -Path src, configs, notebooks, tests, docs, scripts
New-Item -ItemType Directory -Path src/reward_gap
New-Item -ItemType File -Path src/reward_gap/__init__.py
git init -b main
```

Use Windows for editing if you prefer. Target a Linux GPU environment for the full experiment, and explicitly support CPU execution for small correctness tests.

## 3. Use this final folder design

Build toward this tree gradually. Do not create empty implementations for every file at once.

```text
reward-gap-lab/
├── README.md                       # Setup and the one command to run
├── pyproject.toml                  # Package metadata, dependencies, CLI
├── .gitignore
├── configs/
│   ├── smoke.json                  # Very small development experiment
│   └── followup.json               # Full two-round experiment
├── notebooks/
│   ├── RUN_EXPERIMENT.ipynb        # Thin controls over the CLI
│   └── EXPLORE_RESULTS.ipynb       # Read completed artifacts and plot
├── src/reward_gap/
│   ├── __init__.py
│   ├── cli.py                      # prepare / preflight / run / status / pause
│   ├── config.py                   # Typed settings, defaults and validation
│   ├── artifacts.py                # Atomic writes, hashes, run directories
│   ├── data.py                     # Load and partition conversation groups
│   ├── models.py                   # Resolve pinned model files and tokenizers
│   ├── formatting.py               # Consistent chat formatting
│   ├── policy.py                   # PPOActor and its value head
│   ├── scorers.py                  # Frozen proxy/judge RewardScorer
│   ├── calibration.py             # Fit/load frozen score normalization
│   ├── memory.py                  # GapMemory: build, predict, append, save
│   ├── rewards.py                 # ProxyReward and KNNReward strategies
│   ├── ppo.py                     # PPOTrainer and checkpoint restoration
│   ├── evaluation.py              # Generate, score and save evaluation rows
│   ├── refresh.py                 # Label refresh answers and lock memory
│   ├── experiment.py              # FollowupExperiment: stage order and forks
│   ├── reporting.py               # Metrics, plots, report and export
│   └── review.py                  # Blinded packs and imported human ratings
├── tests/                         # Small behavioral tests, no large downloads
├── scripts/                       # Explicit maintenance tools only
├── docs/
│   ├── architecture.md
│   ├── protocol.md
│   └── experiments/               # Short plans and result summaries
├── data/                          # Local datasets; excluded from Git
├── model_cache/                   # Downloaded model weights; excluded
└── outputs/                       # Run artifacts and checkpoints; excluded
```

The package owns reusable code. The notebook owns user interaction. Configuration files own supported choices. Outputs contain the evidence needed to inspect or resume a run.

Prefer explicit package imports such as `from reward_gap.policy import PPOActor`. Install the package into the environment; avoid notebook `sys.path` edits and duplicate files named `study_train.py` in separate experiment directories.

## 4. Establish the first commit

Create `.gitignore` with:

```gitignore
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/
build/
dist/
.ipynb_checkpoints/
*.executed.ipynb
.env
.env.*
!.env.example
/data/
/model_cache/
/outputs/
*.pt
*.safetensors
```

Keep source notebooks with outputs cleared. Keep code, configuration, tests, dependency declarations, protocol documentation and small selected result summaries in Git. Keep datasets, full generated answers, weights, training checkpoints and raw reviewer exports in artifact storage.

In `README.md`, write the project objective and state that it is under construction. In `docs/architecture.md`, record the tree above and the class responsibilities below.

Create `pyproject.toml` using a `src` package layout. Declare a Python version, tested dependencies, a test extra and the future command entry point:

```toml
[project.scripts]
reward-gap = "reward_gap.cli:main"
```

This is only the entry-point fragment, not a complete `pyproject.toml`. Add it when `cli.py:main` exists. Dependencies will include PyTorch, Transformers, PEFT, NumPy and reporting/testing packages. Select a CUDA-compatible PyTorch build for the actual GPU; record the resolved environment for each run.

Review and commit only the files you created:

```powershell
git add .gitignore README.md pyproject.toml src/reward_gap/__init__.py docs/architecture.md
git diff --cached
git commit -m "chore: initialize reward-gap package and architecture"
```

If Git requests your identity, configure your own name and email for this repository and retry. A commit saves a local version; a push uploads committed history to GitHub.

Create an empty GitHub repository named `reward-gap-lab`, without an initial README or other generated files. Authenticate Git using your normal GitHub setup. Replace `YOUR_ACCOUNT` below:

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/reward-gap-lab.git
git push -u origin main
```

This initial-import sequence follows [GitHub's existing-code instructions](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## 5. Define one configuration contract

Implement `config.py` before model loading. Use typed dataclasses or a schema library. Reject unknown fields and invalid combinations instead of silently accepting typos.

The following is a **design example**, not a runnable configuration. Replace every placeholder with a real pinned identifier. Your implementation must support each field before you use it.

```json
{
  "schema_version": 1,
  "experiment": "followup",
  "seeds": [42, 43, 44],
  "models": {
    "policy": {"id": "POLICY_REPO", "revision": "COMMIT_HASH"},
    "proxy": {"id": "PROXY_REPO", "revision": "COMMIT_HASH"},
    "judge": {"id": "JUDGE_REPO", "revision": "COMMIT_HASH"}
  },
  "data": {"manifest": "data/input_manifest.json"},
  "policy": {"lora_rank": 8, "lora_alpha": 16},
  "training": {
    "round1_updates": 100,
    "total_updates": 200,
    "rollout_batch_size": 32,
    "learning_rate": 0.000003,
    "kl_coefficient": 0.05,
    "checkpoint_every": 10
  },
  "generation": {"max_prompt_tokens": 512, "max_new_tokens": 256},
  "scoring": {"max_tokens": 4096},
  "memory": {"k": 31, "temperature": 0.05, "correction": "signed"},
  "refresh": {"answers_per_seed": 1024},
  "evaluation": {"final_prompts": 1024},
  "branches": ["raw", "static_knn", "iterative_knn"],
  "runtime": {"device": "cuda", "output_root": "outputs", "allow_downloads": true}
}
```

Define the remaining PPO settings explicitly in the schema: value learning rate, clipping, discount, GAE lambda, minibatches, microbatches, epochs, gradient clipping and generation sampling. Persist **all resolved defaults** with the run so the saved configuration is complete.

Resolve configuration-relative paths consistently: for example, all relative project paths resolve against the directory containing `pyproject.toml`. Do not let the notebook's working directory change their meaning.

One top-level JSON can select the complete experiment, but it still depends on code, model files and data. A JSON cannot implement a new algorithm on its own.

Implement `artifacts.py` alongside configuration:

- Derive a run identity from resolved scientific configuration, source version/hash, input hashes, model revisions and relevant package versions.
- Keep execution controls such as log verbosity separate from scientific settings.
- Save `resolved_config.json`, `manifest.json` and `status.json` atomically.
- Do not overwrite a completed run when its identity changes.

**Verify:** malformed settings fail with helpful errors; scientific changes alter run identity; runtime-only changes follow the documented resume rules.

**Commit:** `feat: add validated configuration and experiment identity`

## 6. Implement data and input preparation

In `data.py`, define a stable prompt record: `prompt_id`, `conversation_group`, and conversation messages. Partition by conversation group so alternate responses or duplicate conversations cannot leak between splits.

Create and seal these distinct cohorts:

| Cohort | Use |
|---|---|
| Calibration | Fit score means/stds and any fixed high-gap threshold. |
| Initial memory | Generate and label examples for M0. |
| PPO training | Matched training prompt schedule across branches within each seed. |
| Refresh | Generate fresh examples from the round-one corrected policy. |
| Validation | Choose optional capped-ablation parameters. |
| Final evaluation | Measure completed policies after all training decisions are locked. |

Write `data/input_manifest.json` listing source dataset revision, split procedure, seeds, counts and file hashes. Input preparation is an explicit operation, not a notebook secretly writing Python files.

**Verify:** group separation, sufficient prompt counts, deterministic schedules and detection of changed input files.

**Commit:** `feat: prepare reproducible data cohorts and input manifests`

## 7. Implement the actor and frozen scorers

Define these interfaces:

| Class | File | State | Main methods |
|---|---|---|---|
| `PPOActor` | `policy.py` | Base model, LoRA adapter, tokenizer, value head | `load`, `generate`, `statistics` |
| `RewardScorer` | `scorers.py` | Frozen model and its tokenizer | `load`, `score` |

`PPOActor.generate()` must return decoded answers **and** token IDs, attention masks, response masks and EOS information. PPO needs token-level information, not only answer strings.

`statistics()` returns response-token log probabilities and value estimates. Define the KL reference as the original frozen policy; with LoRA, disabling the adapter can provide this reference without a second full model, provided parity is verified.

`RewardScorer.score()` returns raw scores and token counts, plus normalized embeddings when requested. Use the same frozen proxy encoder and pooling rule for memory construction and every query.

Use explicit full-answer scoring. If an input exceeds the configured scoring limit, raise an error instead of silently truncating. Keep model loading in `models.py` and conversation formatting in `formatting.py`.

**Verify:** small models generate valid masks, frozen parameters stay frozen, reference behavior is correct and score batching preserves results within documented numerical tolerance.

**Commit:** `feat: add LoRA policy and frozen reward scorers`

## 8. Implement calibration, memory and reward strategies

| Class | File | State | Main methods |
|---|---|---|---|
| `GapMemory` | `memory.py` | Unit vectors, signed gaps, encoder provenance | `predict`, `append`, `save`, `load` |
| `ProxyReward` | `rewards.py` | Proxy scorer and frozen calibration | `score` |
| `KNNReward` | `rewards.py` | Proxy scorer, calibration, memory, correction rule | `score` |

Start with these two reward strategies. Both implement a common interface, for example:

```python
score(prompts, answers) -> RewardBatch
```

`RewardBatch` should contain training rewards and diagnostics such as proxy scores, predicted gaps and neighbor distances. The trainer uses the reward field without needing to know which strategy produced it.

Fit calibration once on its dedicated cohort. Generate initial-memory responses from the initial policy, score them with proxy and judge, and store all signed normalized gaps in M0. Save calibration and memory separately, with hashes and provenance.

For kNN, define cosine similarity on unit vectors, neighbor count, temperature weighting and deterministic tie behavior. Reject an undersized or incompatible memory. A refresh creates a new immutable memory version; it does not overwrite M0.

**Verify:** known synthetic neighbors give expected predictions; negative gaps remain negative; raw reward has no correction; ordinary training reward calculation never calls the judge.

**Commit:** `feat: add calibrated proxy and signed kNN rewards`

## 9. Implement one reusable PPO trainer

Create `PPOTrainer` in `ppo.py`. It receives an actor, a reward strategy and training settings:

```python
trainer = PPOTrainer(actor=actor, reward=reward_strategy, config=training_config)
metrics = trainer.update(prompt_batch, rollout_seed=rollout_seed)
```

One update must:

1. Generate answers from the current actor.
2. Obtain scalar rewards from the supplied strategy.
3. Compute old-policy and frozen-reference log probabilities and value estimates.
4. Add the KL penalty and place answer reward at the response terminal position.
5. Compute masked advantages and returns.
6. Optimize clipped policy and value objectives across the configured minibatches/epochs.
7. Return metrics and update progress only after the update finishes.

Save LoRA weights, value-head weights, optimizer state, RNG states, update number, prompt-schedule position and configuration/memory identities in resumable checkpoints. An adapter-only export is for generation; it is insufficient for exact training continuation.

**Verify:** actual tiny-model updates change trainable weights, padding does not affect loss, and uninterrupted versus checkpoint-resumed training agrees under controlled conditions.

**Commit:** `feat: implement PPO updates and complete checkpoint resume`

## 10. Build the experiment coordinator

Create `FollowupExperiment` in `experiment.py`. Its `run()` method owns stage ordering. Keep PPO math out of this class.

```text
Validate configuration and artifacts
    → Load or prepare frozen calibration and M0
    → Save an initial training state for each seed
    → Round 1: train raw and static kNN from that seed's initial state
    → Generate refresh answers from the round-one static checkpoint
    → Proxy/judge-score refresh answers; create and lock M1
    → Round 2: continue raw from its own round-one checkpoint
    → Round 2: fork static and iterative from the SAME corrected checkpoint
         static uses M0; iterative uses M1
    → Lock every final policy and memory
    → Evaluate all final branches on the fixed final cohort
    → Write reports and review packs
```

Restore the same policy, value head, optimizer and appropriate RNG state at corrected forks. Match training prompts and rollout seed schedules between compared branches. The intended main difference is memory content.

The fork diagram describes independent conditions; it does not require simultaneous GPU processes. Run branches sequentially at first.

Unlike the old project, a new project need not have legacy adapters. Begin with newly trained policies. Add an optional legacy-checkpoint evaluator later only if you have compatible saved adapters and their provenance.

**Verify:** a tiny end-to-end run follows the declared stages; fork states match; static memory is unchanged; final labels are unavailable to training/selection code; interrupted stages resume safely.

**Commit:** `feat: orchestrate matched two-round memory-refresh experiment`

## 11. Implement evaluation and reporting

In `evaluation.py`, save one record per generated answer with prompt ID, policy/seed/checkpoint identity, answer, proxy/judge scores, true and predicted gap, token length, EOS status and relevant KL diagnostics. Save complete batches atomically and validate dependencies before reusing them.

In `reporting.py`, aggregate proxy reward, judge reward, gap, high-gap rate, response length/completion and memory-prediction error. Preserve seed-level results and use matched prompt comparisons where applicable.

In `review.py`, generate randomized blinded answer pairs with the private mapping stored separately. Import human ratings explicitly; never present missing ratings as an automated result.

Write artifacts under:

```text
outputs/<run_id>/
├── resolved_config.json
├── manifest.json
├── status.json
├── calibration/
├── memories/
├── runs/                   # Per-seed/per-branch checkpoints
├── evaluations/
├── reports/
└── review/
```

**Verify:** incomplete runs are labeled incomplete, metrics use completed cases, and blinded exports exclude private mappings and model scores.

**Commit:** `feat: add evaluation reports and blinded review artifacts`

## 12. Add the CLI, then a thin notebook

Implement these proposed commands in `cli.py`:

```text
reward-gap prepare --config configs/followup.json
reward-gap preflight --config configs/followup.json
reward-gap run --config configs/followup.json
reward-gap status --run outputs/<run_id>
reward-gap pause --run outputs/<run_id>
reward-gap resume --run outputs/<run_id>
```

These commands become usable only after you implement them. Install the package in the intended environment using `python -m pip install -e ".[test]"` once the package metadata and test extra are complete.

First make foreground execution work. Then add an explicit `--detach` option: the launcher starts a separate worker, records process identity, redirects logs and prevents duplicate workers. Pause must be cooperative at complete batches or PPO updates. Resume reads the original resolved configuration and validates dependencies.

Keep `RUN_EXPERIMENT.ipynb` to these cells:

1. Set the project/config path and display resolved settings.
2. Call preflight and show its actual result.
3. Start or resume through the CLI.
4. Read live status and link existing reports.
5. Optionally request pause or import completed human ratings.

The notebook must not create `.py` source files or contain another implementation of the training loop. Prefer selecting a checked-in config over rewriting it silently. If notebook overrides are supported, save and display the exact resolved config for the run.

**Verify:** notebook and CLI select the same configuration; failed preflight prevents launch; duplicate starts are rejected; status checks the worker, not merely a stale PID file.

**Commit:** `feat: expose experiment CLI and notebook controls`

## 13. Run a smoke experiment before a full experiment

Create `configs/smoke.json` with one seed, a few updates, short generations and small disjoint cohorts. Keep memory size at least `k`, or explicitly select a smaller `k` for this smoke configuration. Use compatible small models when possible; also run a short preflight with the actual pinned production models on the GPU.

Verify model loading, reward/embedding parity, one real PPO update, checkpoint restoration, stage completion and report generation. Treat smoke results as software validation, not scientific evidence.

Write the intended full comparison and fixed settings in `docs/protocol.md`. Commit the full configuration before launching it.

**Commit:** `experiment: define smoke and full follow-up protocols`

## 14. Repeat this workflow for every change or experiment

### A. Start from a known version

Finish or cooperatively pause any worker using the checkout before changing its files. On your development machine, check your worktree before switching branches:

```powershell
git status
git switch main
git pull --ff-only
git switch -c experiment/refresh-size-2048
```

If `git status` shows unfinished changes, commit them on their current branch or otherwise resolve them before switching. Use a new descriptive branch name for each task.

### B. Write the question first

Create `docs/experiments/refresh-size-2048.md` containing the hypothesis, comparison, changed settings, fixed settings and planned evaluation. For example: increase refresh answers from 1,024 to 2,048 while holding the other settings fixed.

### C. Change the smallest appropriate component

| Desired change | Edit |
|---|---|
| Existing supported hyperparameter | Experiment JSON |
| New correction rule | `rewards.py` and configuration validation |
| New embedding/memory algorithm | `memory.py`, with compatible memory provenance |
| PPO optimization behavior | `ppo.py` |
| New stage or comparison | `experiment.py`, or a new coordinator using the same components |
| New metric or plot | `evaluation.py` / `reporting.py` |
| Notebook controls | Notebook / CLI, without copying model code |

A changed embedding model or calibration requires compatible rebuilt memory. Do not continue with old memory just because its array shape matches.

### D. Validate, inspect, then commit

Run tests appropriate to the change and a smoke experiment when training behavior changes. Review configuration and notebook diffs, including saved outputs.

```powershell
python -m pytest tests
git diff
git add configs/followup.json docs/experiments/refresh-size-2048.md
git diff --cached
git commit -m "experiment: increase refresh cohort to 2048 answers"
git push -u origin experiment/refresh-size-2048
git rev-parse HEAD
```

The staged filenames above illustrate a config-only change. Add the specific implementation/test files you actually changed. Do not make a commit saying tests passed unless they did.

### E. Execute the committed version on the GPU machine

Use a separate checkout for an incompatible new run. On that machine, fetch and check out the exact commit recorded above; install the package, provide the sealed input artifacts, and run preflight before the full experiment. These are the new project's proposed CLI commands, not commands implemented by the existing project.

Record the Git commit, resolved configuration, input/model hashes and environment in the run manifest. A Git commit alone cannot restore model files or training state.

### F. Observe, pause or resume

Inspect worker status and logs. Resume with the original run's stored configuration and checkpoints. Changing scientific settings creates a new run rather than redefining the old one.

### G. Record the result in a second commit

Update the experiment note with:

- Code commit actually executed and run ID.
- Exact artifact location and status: complete, failed or paused.
- Relevant metrics with sample counts and uncertainty where available.
- Validation performed and remaining limitations, including human-review status.

Keep bulky artifacts outside Git; retain full checkpoints if exact continuation matters. Copy only selected small figures/tables into the documentation if useful.

```powershell
git add docs/experiments/refresh-size-2048.md
git diff --cached
git commit -m "docs: record refresh-size experiment results"
git push
```

The result note references the **earlier code/config commit**, not its own later documentation commit. Open a pull request when the change and evidence are ready; after review/merge, begin the next task from updated `main`.

## 15. Add extensions after the main experiment works

Add a capped reward ablation with an explicitly defined formula and validation-only selection. Add a judge-trained reward strategy for a teacher-access comparison. Add stress or repeated-refresh coordinators when you need different experimental stage sequences.

Reuse `PPOActor`, `PPOTrainer`, scorers, memory, evaluation and artifact handling. Give each new experiment its own validated configuration and coordinator, rather than copying the entire project into another folder.

The intended construction order is: **package → config/artifacts → data → models → rewards/memory → PPO → experiment → evaluation → CLI/notebook → smoke run → full run**. Each working stage gets a focused commit, and each actual experiment gets a pre-run specification plus a post-run result record.
