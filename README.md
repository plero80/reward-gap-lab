# Reward Gap Lab

Experiments testing whether a k-nearest-neighbor reward-gap memory
can improve language-model training with PPO.

A frozen proxy model provides rewards. A stronger frozen judge
labels disagreements. A memory of these disagreements predicts
corrections to the proxy reward.

The judge is a reference model, not human ground truth.

## Status

Under construction. Configuration validation, atomic JSON saving, and HH-RLHF
prompt preparation are implemented. Model loading and training are not yet implemented.

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
