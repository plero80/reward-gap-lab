# Audit follow-up: combined HH-RLHF and correctness fixes

Addresses the user-supplied audit of commit `2727ad2` dated 2026-09-15.

| Finding | Change | Verification |
| --- | --- | --- |
| HH reward text can extend beyond TRL's PAD endpoint | `RewardBridge` validates contiguous responses for every reward strategy before scoring, including HH proxy/kNN | Tests invoke real TRL `get_reward()` with an interior PAD in both HH arms and assert no scoring occurs |
| HH preparation/evaluation and PPO have different EOS rules | `PPOActor` selects primary tokenizer EOS at construction; standalone generation rejects PAD before EOS; checkpoint identity records response contract | Scripted alternate-EOS and PAD tests; explicit historical inference test; incompatible optimizer resume rejected |
| GSM8K equality creates false positives | Actual and predicted gaps both use strict `> theta`; snapshot records the detector version | Perfect predictions `[0,1,1,2]` at theta 1 give TP=1, TN=3, FP=FN=0 and precision/recall=1 |
| Separate HH/RQ1 runs duplicate work and omit paired M0/M1 analysis | `hh-run` shares calibration, M0 and PPO checkpoints, then evaluates all predictors on each identical answer set | Tiny-model end-to-end test checks actual PPO update counts, shared answer counts and matched label budgets |
| Missing policy trajectory | Retain initial, update-1, both round-1 and three final checkpoints; evaluate after all seeds finish training | Boundary-progress and no-final-labels-before-training tests |
| Updated student needs matched labels | Updated gap ridge and judge student use exactly the M1 examples, frozen variants exactly M0 | Memory IDs/gaps are checked against saved label rows; report records label counts |
| Incomplete HH metrics and confusing constant-baseline recall | Separate policy and predictor tables, full regression/detection metrics, explicit all-positive/never-positive flags; HTML/Markdown/CSV and PNG/PDF plots | Tests check metric counts, constant-baseline confusion values and generated artifacts |
| Existing retained policies need inference-only evaluation | `hh-evaluate` checks source config/input identity and fingerprints source artifacts, writes separately and never trains PPO | Legacy-source test forbids PPO updates and checks source bytes remain unchanged; changed-source resume is rejected |

## Scope and interpretation

- Standard GSM8K still has one static kNN-PPO stage per seed. The teacher
  comparison retains its existing numeric endpoint. No GSM memory-refresh
  stage or Judge-30B training arm was added.
- Historical RQ1-only runs do not contain M1. The unified protocol needs a
  new training run; retained original two-round HH runs can be analyzed
  without PPO retraining. Missing historical raw round-1 checkpoints are
  disclosed, not recreated by additional training.
- Historical PPO cannot be repaired by changing inference. New runs use
  explicit response and detector versions to avoid mixing corrected and old
  stages; historical policy weights may be loaded for labeled inference only.
- Final test answers never tune predictor parameters or detector thresholds.
  M0/M1 metrics share each policy's saved answer set. Differences between final
  policies remain a separate outcome from differences between memories.
- The report does not establish independently verified answer quality, reward
  hacking, or a statistically reliable improvement from one seed. It also does
  not add full GPU-time accounting. GSM scheduled/optimized/skipped counters
  retain their existing semantics.

## Validation

The complete offline suite passed 403 tests after the protocol/report changes.
Additional focused checks cover historical inference and intermediate checkpoint
retention. Commands (PowerShell environment variables shown in the test logs):

```text
python -m pytest -q --tb=short
python -m pytest tests/test_hh.py tests/test_ppo.py::test_historical_eos_metadata_only_allowed_for_explicit_inference -q --tb=short
```

The local tests use tiny CPU models and cached/offline resources. No production
GPU experiment or existing user result was rerun. Full and single-seed configs
were loaded through the actual config parser, and both CLI help routes checked.
See [the run guide](hh_unified.md) for commands and result files.
