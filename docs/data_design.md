# What was implemented for HH-RLHF, and why

This document explains the data preparation work completed on September 13,
2026. It describes the implementation that exists, its design choices, and
its limits. See [data.md](data.md) for operating instructions.

## 1. The goal

The experiment needs conversations that a policy model can answer. It also
needs separate conversations for calibration, reward memory, training, and
evaluation, so those activities do not accidentally share the same material.

HH-RLHF stores preference comparisons: a chosen transcript and a rejected
transcript. We convert these into prompts and prepare fixed groups of inputs
for the later experiment. This work does not load models, train PPO, score
answers, or produce experimental quality results.

The design follows the workshop scope: readable JSON files, saved settings,
repeatable splits, and protection against overwriting prepared data. It does
not build a SHA-based experiment identity or a tracking service.

## 2. Files changed

| File | What changed | Why |
|---|---|---|
| `src/reward_gap/data.py` | Added downloading, parsing, validation, grouping, splitting, schedules, saving, and reading prepared prompts | Keep reusable data logic in the installed package |
| `src/reward_gap/config.py` | Added `DataConfig` and validation for its fields | Keep supported experiment choices in configuration |
| `configs/smoke.json` | Added dataset subsets, revision, paths, split seed, and minimum cohort sizes | Make a development preparation possible without editing Python |
| `tests/test_data.py` | Added parser, split, schedule, configuration, saved-file, and offline preparation tests | Check behavior without requiring network access or models |
| `README.md` | Added preparation instructions and updated implementation status | Give the project a usable entry point |
| `docs/data.md` | Added usage and format documentation | Explain how later code should consume prepared inputs |

The empty `src/data.py` was removed. The implementation belongs at
`src/reward_gap/data.py`, so imports use `reward_gap.data`.

The existing `artifacts.py` is reused for atomic JSON writes. It did not need
another saving implementation for this task. No new Python dependencies were
added: JSON, gzip, HTTP downloads, and seeded shuffling use Python's standard
library. Downloaded datasets and prepared inputs are under the ignored `data/`
directory and are not intended for Git.

## 3. How data moves through the implementation

```text
configs/smoke.json
        |
        v
load_config(): validate settings and resolve project-relative paths
        |
        v
Resolve dataset version and download/read cached train/test files
        |
        v
read_hh_file() -> extract_prompt(): validated conversation prompts
        |
        v
partition_prompts(): remove duplicates, reserve test groups, split groups
        |
        v
prompt_schedule(): repeatable training batches for each experiment seed
        |
        v
Save cohort files, schedules, settings, and input_manifest.json
        |
        v
load_prompts(): read a prepared cohort for later experiment code
```

`prepare_data()` coordinates these operations. `main()` exposes a small
command-line interface for data preparation only. It is not the future full
experiment CLI in `cli.py`.

## 4. The configuration choices

| Setting | Meaning |
|---|---|
| `data.revision` | Which HH-RLHF version to read; `main` is resolved to a concrete source revision during preparation |
| `data.subsets` | Which of the four supported preference subsets to include |
| `data.split_seed` | Seed controlling how conversation groups are allocated |
| `data.cache_dir` | Where downloaded original files are kept |
| `data.prepared_dir` | New directory for the prepared cohorts and manifest |
| `data.minimum_prompts` | Minimum prompt count for each of the six cohorts |

The initial selection includes `helpful-base`, `helpful-online`,
`helpful-rejection-sampled`, and `harmless-base`. This is an explicit default,
not a requirement that every study combine them. The red-teaming subset is
not supported by this adapter because its records use a different schema.

The schema rejects unknown fields, unsupported or repeated subsets, invalid
seeds, missing cohort counts, and empty paths. Validation may be zero when
unused; the other five cohorts require positive counts. A configuration
without a `data` section can still load for earlier configuration-only work,
but `prepare_data()` requires it.

The split seed and experiment seeds do different jobs. `data.split_seed`
chooses the cohorts. Top-level `seeds` choose training schedules within the
already selected training cohort. Comparing branches within one experiment
seed should therefore use the same inputs in the same order.

## 5. Why the final answers are removed

The policy must generate its own next answer. Feeding it HH-RLHF's final
chosen answer would reveal an answer that is supposed to be generated.

`extract_prompt()` parses both transcripts using the Human/Assistant role
markers. It requires alternating, nonempty turns and a final assistant answer.
It removes that final answer from each transcript and verifies that the
remaining message sequences agree. Earlier assistant messages stay because
they are part of the conversation context.

The output contains `Message` objects with `role` and `content`. The enclosing
`PromptRecord` has:

- `prompt_id`: source subset, source split, and original line number.
- `conversation_group`: a key used to keep related prompts together.
- `messages`: the conversation, ending with a user message.

The source revision in the manifest gives the IDs their version context.
These IDs are not claimed to be stable across arbitrary dataset revisions.

Malformed dialogue rows are omitted and counted by reason. Broken JSON or an
unreadable file fails preparation instead of quietly losing unknown data.
This makes the amount of filtering visible without printing conversation
contents in routine progress logs.

## 6. Why splitting uses conversation groups

Two rows can be different continuations of the same conversation. Splitting
individual rows at random could put one continuation into training and another
into evaluation. Evaluation would then reuse part of the training material.

The preference files do not provide a reliable original conversation ID for
this purpose. The implemented approximation groups records by their first
user message after collapsing whitespace and ignoring letter case. The
normalization affects the grouping key, not the message text sent to a model.

This deliberately favors keeping possibly related conversations together.
It can also combine unrelated conversations that begin with the same words.
It cannot identify every paraphrase or other semantic duplicate. Consequently,
the code verifies separation under this grouping rule, not perfect semantic
independence of all conversations.

Exact repeated prompt message sequences are deduplicated within each source
split. Then every group present in the original test split is excluded from
the training pool, even if that test group is not selected for the small
final-evaluation cohort. This preserves the original test boundary under the
chosen grouping rule.

## 7. The six cohorts and their purpose

A cohort here simply means a set of prompts assigned one job.

| Cohort | Source split | Later use |
|---|---|---|
| Calibration | Train | Fit score-normalization constants |
| Initial memory | Train | Generate answers and label the initial reward-gap memory |
| Training | Train | Supply prompts for PPO updates |
| Refresh | Train | Generate fresh answers to update memory after round one |
| Validation | Train | Choose optional settings such as capped correction |
| Final evaluation | Test | Evaluate policies after training decisions are fixed |

Groups are sorted before seeded shuffling. The development cohorts are then
filled in a fixed order; final evaluation draws from its separate test pool.
Sorting means changing the input list order alone does not change the split.

Counts are minimums because a whole group is allocated at once. Requesting 32
prompts can produce 33. This preserves group boundaries. If the remaining
groups cannot satisfy a request, preparation raises an error. It does not
silently borrow prompts from evaluation or reuse another cohort.

Allocation is not stratified by subset, topic, or number of conversation turns.
Small smoke cohorts are development inputs, not representative evaluation
samples. A full workshop result should use deliberately chosen larger counts
and review the composition of the selected data.

## 8. Why training schedules are saved

A schedule is a list of batches, each containing prompt IDs for one update.
One schedule is generated for each top-level experiment seed.

The schedule shuffles the training prompts, uses each once per epoch, and
reshuffles when an epoch ends. A batch crossing an epoch boundary can contain
a repeated prompt. The number of batches is `training.total_updates`; their
size is `training.rollout_batch_size`.

Saving this schedule lets raw-reward and corrected-reward branches use matched
training prompts. Otherwise, a result could partly reflect different prompt
orders rather than the reward strategy. Later training code must actually
consume the saved schedule; this data layer cannot enforce that by itself.

Round one uses the first `round1_updates` batches. Round two continues with the
following batches. Changing update counts or batch size requires new prepared
inputs with a matching schedule.

## 9. Saved files and overwrite protection

Preparation creates six cohort JSON files, a schedule for each seed, a copy
of resolved settings, and `input_manifest.json`.

The manifest is a readable inventory: dataset version, subsets, split seed,
split procedure, requested and actual counts, filtered rows, removed duplicates,
test-overlap exclusions, unused prompts, and output filenames. It is not a
database, run identity, or training checkpoint.

The prepared directory must not already exist. It is created with
`exist_ok=False` before writing prepared files. This prevents a new preparation
from overwriting old inputs and also prevents two preparations from both
claiming the same destination.

Each JSON file uses the existing atomic writer: write a temporary file, then
replace the destination after the write succeeds. The manifest is saved last.
If preparation crashes before that, the directory has no completion manifest.
Use a new destination on retry; automatic recovery is not implemented.

This does not provide multi-file transactions, automatic training resumption,
or detection of later manual edits. Those mechanisms were intentionally left
outside the current workshop scope.

## 10. What was verified

After implementation, the full local test suite passed: **55 tests**. Tests
include removing only the final answer, rejecting mismatched contexts,
reporting malformed records, group separation, test overlap exclusion,
deduplication, deterministic splitting, insufficient data, repeatable schedules,
configuration validation, offline preparation, and refusal to overwrite output.

These tests use small local fixtures for speed. Separately, the actual dataset
was downloaded and prepared successfully using source revision
`09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`.

The saved manifest recorded 169,352 source rows and 718 skipped malformed or
ambiguous dialogue rows. Deduplication removed 815 train prompts and 3 test
prompts. The grouping rule excluded 17,677 remaining train prompts because
their group also appeared in test. Most eligible prompts remained unused by
the small smoke configuration.

| Cohort | Requested minimum | Prepared prompts | Conversation groups |
|---|---:|---:|---:|
| Calibration | 16 | 16 | 7 |
| Initial memory | 32 | 33 | 14 |
| Training | 32 | 33 | 12 |
| Refresh | 16 | 19 | 6 |
| Validation | 8 | 8 | 4 |
| Final evaluation | 16 | 16 | 16 |

The saved files were read back to check counts, disjoint group and prompt IDs
across cohorts, and that schedule IDs refer only to training prompts. These
checks establish data preparation behavior; they do not establish model quality
or the scientific success of the reward-gap experiment.

## 11. Known limits and next work

- Grouping is an approximation based on opening messages, not a recovered
  original conversation ID or semantic deduplication system.
- Role-marker parsing cannot disambiguate every quoted Human/Assistant
  exchange from real conversation turns. Strict checks filter many ambiguous
  rows but cannot guarantee all surviving interpretations are correct.
- Filtering and conservative grouping change the usable dataset distribution.
  Inspect the recorded counts when selecting a full experiment configuration.
- Preparation reads and holds the selected source prompts in memory before
  sampling. It worked on this HH-RLHF dataset; it is not a general streaming
  pipeline for datasets of arbitrary size.
- Cached files are checked when first downloaded, but are not tracked with
  content hashes. A damaged existing cache file may require manual replacement.
- `revision: main` can point to a different dataset version in the future.
  Use the concrete recorded revision when repeating this preparation.
- The `load_prompts()` reader validates individual cohort records. It does
  not automatically verify every other file in the prepared directory or
  detect every manual change to a completed preparation.

Next, later model and experiment modules can read these prepared prompts.
They still need model configuration, tokenization/chat formatting, token-limit
checks, generation, scoring, PPO, evaluation, and results reporting. None of
those steps is performed by the current data-preparation command.
