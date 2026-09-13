# Planned architecture


## Folder structure

reward-gap-lab/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── smoke.json
│   └── followup.json
├── notebooks/
│   ├── RUN_EXPERIMENT.ipynb
│   └── EXPLORE_RESULTS.ipynb
├── src/reward_gap/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── artifacts.py
│   ├── data.py
│   ├── models.py
│   ├── formatting.py
│   ├── policy.py
│   ├── scorers.py
│   ├── calibration.py
│   ├── memory.py
│   ├── rewards.py
│   ├── ppo.py
│   ├── evaluation.py
│   ├── refresh.py
│   ├── experiment.py
│   ├── reporting.py
│   └── review.py
├── tests/
├── scripts/
├── docs/
│   ├── architecture.md
│   ├── protocol.md
│   └── experiments/
├── data/
├── model_cache/
└── outputs/

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| cli.py | Commands: prepare, preflight, run, status, pause |
| config.py | Typed settings, defaults, and validation |
| artifacts.py | Atomic writes, hashes, and run directories |
| data.py | Load data and partition conversation groups |
| models.py | Resolve pinned model files and tokenizers |
| formatting.py | Consistent chat formatting |
| policy.py | PPOActor, policy adapter, and value head |
| scorers.py | Frozen proxy and judge RewardScorer |
| calibration.py | Fit and load frozen score normalization |
| memory.py | GapMemory: build, predict, append, and save |
| rewards.py | ProxyReward and KNNReward strategies |
| ppo.py | PPOTrainer and checkpoint restoration |
| evaluation.py | Generate, score, and save evaluation rows |
| refresh.py | Label refresh answers and lock memory |
| experiment.py | FollowupExperiment: stage ordering and forks |
| reporting.py | Metrics, plots, reports, and exports |
| review.py | Blinded review packs and imported human ratings |

## Design rules

- The package owns reusable code.
- Notebooks provide thin controls over the CLI and explore saved results.
- Configuration files own supported experiment choices.
- Outputs preserve evidence needed to inspect or resume a run.
- Use explicit imports, such as `from reward_gap.policy import PPOActor`.
- Install the package; avoid notebook modifications to `sys.path`.
- Keep tests small and independent of large model downloads.
- Keep datasets, model weights, and run artifacts outside Git.
- Clear notebook outputs before committing.

## Reward correction

Fit normalization constants on a separate calibration cohort.

    true_gap = z_proxy - z_judge
    predicted_gap = weighted average of nearby memory gaps
    corrected_reward = z_proxy - predicted_gap

Memory stores normalized proxy embeddings and their labeled gaps.
The judge labels memory examples and supports evaluation.
Ordinary training reward calculation does not call the judge.

The judge is a reference model, not human ground truth.
Evaluate answer quality and include separate human review.