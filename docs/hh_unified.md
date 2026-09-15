# Combined HH-RLHF experiment

`hh-run` combines the original two-round HH experiment and RQ1's predictor
comparisons into one run. It shares calibration, initial answers, M0 memory,
and policy checkpoints. GSM8K remains a separate, single static-memory PPO
experiment; this change adds no GSM8K training stages.

## What is trained

1. Generate initial calibration answers and freeze proxy/judge normalization.
2. Build initial memory M0 from a separate cohort of initial-policy answers.
3. Train proxy-only and M0-corrected policies through round 1.
4. Label the corrected round-1 policy's answers on the separate refresh cohort.
   Append those examples to M0 to form M1; retain the same encoder/calibration.
5. Continue proxy PPO. Fork the corrected checkpoint, including optimizer state,
   into static-M0 and iterative-M1 policies for round 2. Their prompt schedules,
   seeds and total update budgets match.
6. After all seeds finish training, fit and freeze prediction baselines/cutoffs
   using development labels, then evaluate all retained policy checkpoints.

The full preset runs 200 updates in each round. Each final policy has a budget
of 400 updates. Because the two corrected branches share round 1, the actual
training work is 1,000 PPO updates per seed (400 proxy + 200 shared corrected
+ 200 static continuation + 200 refreshed continuation).

## What is compared

Policies: initial, proxy/corrected at update 1 and round 1, then final proxy,
static and iterative. The smoke preset may share update-1/round-1 boundaries.
No final test labels are generated until every seed finishes training.

Each policy generates one saved final answer set, including normalized proxy
embeddings. Every predictor uses that identical set:

| Predictor | Labels used |
| --- | --- |
| kNN M0 | Initial-memory labels |
| kNN M1 | Initial-memory plus refresh labels |
| Frozen ridge gap predictor | Initial-memory gaps |
| Updated ridge gap predictor | The exact M1 gaps/examples |
| Frozen linear judge student | Initial-memory normalized judge scores |
| Updated linear judge student | The exact M1 examples and judge scores |
| Zero/mean gap | Constant controls |

The linear models are prediction baselines, not additional PPO reward arms.
Frozen and updated baselines use the same initial-policy validation answers;
regularization and F1-maximizing detector cutoffs are frozen before final labels.
The high-gap threshold is `max(0, calibration 95th percentile)` for all policies.
Actual positives use `gap > theta`; detection uses `prediction >= validated
cutoff`. Constant baselines may flag every answer; the report labels this
explicitly. M1 on initial/round-1 answers is a retrospective diagnostic.

Policy outcomes: normalized proxy and judge means, signed mean gap, high-gap
rate, response length and EOS fraction. Predictor outcomes: MAE, MSE/RMSE,
R2, Pearson/Spearman, AUROC/AP, prevalence, precision/recall/F1 and all four
confusion counts. Each predictor result identifies policy checkpoint, PPO
update, memory, calibration, threshold and detector comparison rule.

Judge agreement is not independently measured task quality or verified reward
hacking. Single-seed differences do not estimate between-run uncertainty.
The report records matched predictor label counts. It does not promise a
complete GPU-time or judge-call cost comparison across the entire workload.

## Run one seed on a GPU pod

### H200 NVL

Use **one H200 NVL** with the existing `hh_seed42.json` full preset. It uses
`cuda:0`, BF16, eight responses per PPO update, minibatches of two, and scoring
batches of two. The implementation uses one GPU; NVLink does not distribute
this run across additional GPUs automatically. Throughput and peak memory
still need measurement on the pod.

Use `hh_h200_smoke.json` for the initial check on this GPU. It copies the full
run's model, batch and token limits (8,192 prompt tokens, 256 answer tokens,
16,384 scorer tokens) with small cohorts and two updates per round. It uses
its own prepared directory because the generic smoke preset's training
schedule has a different batch size. This checks the full configuration on
small cohorts, not every possible long prompt in the full dataset.

After setup and activation below, run these commands one at a time:

```bash
python -m reward_gap.cli prepare --config configs/hh_h200_smoke.json --download
python -m reward_gap.cli preflight --config configs/hh_h200_smoke.json
python -u -m reward_gap.cli hh-run --config configs/hh_h200_smoke.json --run-name hh-h200-smoke-01
```

Then use the full seed-42 commands below. No B200/GSM8K preset is needed for HH.

### Setup and full run

Update the checkout before starting a **new** run. In the pod terminal:

```bash
cd /workspace/reward-gap-lab
git pull --ff-only origin main
python scripts/setup_runpod.py
```

After `Setup passed`, start a tmux session (install tmux if missing):

```bash
tmux new -s hh
cd /workspace/reward-gap-lab
source /tmp/reward-gap-lab-venv/bin/activate
mkdir -p outputs/logs
set -o pipefail
```

First validate the full code path with the smoke preset. Run commands one at
a time, continuing only on success. Skip prepare if that matching directory
is already prepared:

```bash
python -m reward_gap.cli prepare --config configs/hh_smoke.json --download
python -m reward_gap.cli preflight --config configs/hh_smoke.json
python -u -m reward_gap.cli hh-run --config configs/hh_smoke.json --run-name hh-smoke-01
```

Then run seed 42 with the full budget:

```bash
python -m reward_gap.cli prepare --config configs/hh_seed42.json --download
python -m reward_gap.cli preflight --config configs/hh_seed42.json
python -u -m reward_gap.cli hh-run --config configs/hh_seed42.json --run-name hh-seed42 2>&1 | tee -a outputs/logs/hh-seed42.log
```

Repeat the final command to resume the same run. `--until round1` or
`--until training` deliberately pauses at a boundary. Seeds 43/44 have their
own configs and prepared directories/schedules; use `hh-seed43`/`hh-seed44`
as run names. Raw downloads/model weights are cached and reused. Alternatively
`hh_full.json` runs all three seeds in one directory, sharing calibration/M0.
Separate single-seed runs build their own calibration/M0.

Detach with Ctrl+B then D; return with `tmux attach -t hh`. The pod must stay
running. If restarted, rebuild the `/tmp` environment as needed, then repeat
the same run command. Changing config or model revisions requires a fresh run.

## View results

Open `outputs/hh-seed42/report.html` in a browser for filterable tables, or
`report.md` in Markdown preview. `policy_metrics.csv` and
`predictor_metrics.csv` contain the complete tables. The `analysis/` folder
contains saved answers, embeddings, neighbor evidence and frozen predictors.
All summary paths point to the evidence used for each result.
`policy_outcomes.png`/PDF and `memory_comparison.png`/PDF show the policy
scores and the paired M0/M1 prediction results. Keep the plots beside the HTML
report when copying it, or open the PNG/PDF files directly.

New unified runs retain initial, proxy/corrected update-1, proxy/corrected
round-1 and three final checkpoints per seed. One rolling recovery checkpoint
per active branch is replaced periodically and removed at its boundary.

## Evaluate an existing two-round HH run without retraining

Use its original config and prepared inputs. For example, for a completed
legacy `followup.json` run named `followup-01`:

```bash
python -u -m reward_gap.cli hh-evaluate --config configs/followup.json --source-run followup-01 --run-name followup-01-analysis
```

This writes a separate analysis directory and leaves source artifacts intact.
It checks exact source config/cohorts and fingerprints calibration, memory,
label and checkpoint files. It performs model inference and predictor fitting,
but never calls PPO updates. Saved analysis answer stages are reused on resume.
An old coordinator's missing raw round-1/update-1 checkpoints are listed as
unavailable. Its initial policy can be reconstructed from the same base and
seeded adapter setup. The retained corrected-round-1 and final checkpoints
are required. A legacy RQ1-only run lacks M1 and cannot supply this full
comparison; keep its results as the earlier frozen-memory study.

## Audit fixes and historical compatibility

All reward strategies now reject interior PAD or noncontiguous PPO responses
before scoring. Standalone generation also rejects PAD before primary EOS.
HH uses the tokenizer's primary EOS consistently in preflight, preparation,
PPO and evaluation. Checkpoints record the effective EOS and response contract;
new training refuses older checkpoints with a different contract.
Evaluation-only loading explicitly permits historical EOS metadata while
recording the new inference rule. This does not repair historical PPO updates.

GSM8K's same-threshold detector now uses strict `>` for both actual and
predicted gaps; ranking metrics are unchanged by this boundary fix. Existing
saved metrics are not silently rewritten. The standard GSM8K protocol still
uses one frozen-memory kNN stage per seed. Its scheduled/optimized/skipped
batch counts remain essential when comparing training budgets.
