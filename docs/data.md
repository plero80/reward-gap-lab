# HH-RLHF input preparation

The source is [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).
The default configuration includes the four preference subsets: helpful-base,
helpful-online, helpful-rejection-sampled, and harmless-base. It does not use
red-team-attempts, whose records have a different format.

## Preparation

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m reward_gap.data --config configs/smoke.json --download
```

The command downloads original compressed JSONL files into `data.cache_dir`.
The source revision is resolved once and recorded in the manifest. This is
the dataset's own version identifier, not a new content-hashing system.
Without `--download` or `runtime.allow_downloads`, preparation requires a
resolved revision and all required files already in the cache. For offline
use, copy `revision` from the previous manifest into `data.revision` and use
a new `data.prepared_dir`. Paths inside configuration resolve against the
project root, not the current notebook directory.

Preparation reads the full selected train/test files even for the smoke
configuration, so deduplication and test overlap exclusion happen before
sampling. A smoke run's small cohort counts do not limit the initial download.

## Prompt extraction and grouping

Each preference row has chosen and rejected transcripts. The parser removes
the final assistant answer from both and requires their remaining message
sequences to match. Earlier assistant turns remain as context. Chosen answers
are not supervised training targets in this preparation pipeline.

Rows with malformed or ambiguous turns are skipped with reason counts in the
manifest. Invalid JSON or unreadable files fail preparation. Inspect these
counts before interpreting an experiment: filtering changes the source sample.
The role-marker format cannot distinguish every quoted dialogue from actual
turns; strict alternation checks catch many, but not all, ambiguous cases.

Preference records lack a reliable original conversation ID. The grouping rule
uses the first user message with whitespace collapsed and case folded. This
keeps repeated openings, alternate continuations, and later turns together.
It can overgroup unrelated conversations with identical openings and cannot
guarantee detection of paraphrased duplicates. Exact repeated prompt message
sequences are deduplicated within each source split.

All groups appearing in the original test split are excluded from the training
pool, including test groups not selected for final evaluation. Development
cohorts come from train; final evaluation comes only from test. Whole groups
are shuffled with `data.split_seed` and allocated in this fixed order:
calibration, initial_memory, training, refresh, validation, final_evaluation.

`data.minimum_prompts` specifies minimum numbers of prompts, not groups.
Whole-group allocation may exceed a requested count. Insufficient remaining
data fails clearly. Only validation may have a zero count. These are workshop
development defaults, not sample-size recommendations for a full study.

## Prepared files

Each cohort is a JSON list containing `prompt_id`, `conversation_group`, and
`messages` (role/content objects). IDs refer to source subset, split, and line;
the manifest identifies the dataset revision needed to interpret them.

```python
from reward_gap.config import load_config
from reward_gap.data import load_prompts

config = load_config("configs/smoke.json")
training = load_prompts(config.data.prepared_dir / "training.json")
```

One `training_schedule_seed<seed>.json` is saved per experiment seed. All
training branches for that seed should read the same schedule. Each schedule
contains one list of prompt IDs per update, with the configured rollout batch
size. Shuffling draws without replacement within each epoch; crossing an epoch
boundary may repeat a prompt within a batch. Round one uses the first
`round1_updates` batches; round two continues from the following batch.
Changing update counts or batch size requires new preparation.

`input_manifest.json` records source revision/subsets, split procedure and seed,
source row/filter counts, duplicate/overlap removals, actual cohort counts,
unused prompt counts, and schedule filenames. The resolved configuration is
also saved. There is no code/package provenance database or SHA run identity.

The destination directory must be new. Files are saved atomically and the
manifest is written last. If preparation is interrupted, its incomplete
directory is not overwritten on retry; choose a new destination. A missing
manifest means preparation did not complete. Treat completed inputs as fixed
for an experiment; the workshop pipeline does not detect later manual edits
using checksums. All dataset and prepared files remain under ignored `data/`.
