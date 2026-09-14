"""Row-level inference recovery and explicit evaluation coverage."""

from dataclasses import replace

from reward_gap.failures import SampleError

FAILURE_POLICY = "gsm8k_skip_unusable_samples_v1"


def score_partial(scorer, prompts, answers, *, return_embeddings=False):
    if hasattr(scorer, "score_partial"):
        return scorer.score_partial(prompts, answers, return_embeddings=return_embeddings)
    # Alternate scorers can expose the usual strict ScoreBatch API.
    try:
        batch = scorer.score(prompts, answers, return_embeddings=return_embeddings)
    except SampleError:
        results, errors = [], []
        for prompt, answer in zip(prompts, answers, strict=True):
            try:
                results.append(scorer.score([prompt], [answer], return_embeddings=return_embeddings))
                errors.append(None)
            except SampleError as exc:
                results.append(None)
                errors.append(str(exc))
        return results, errors
    if batch.prompt_ids != tuple(p.prompt_id for p in prompts):
        raise ValueError("Grader changed prompt alignment")
    return [replace(batch, prompt_ids=(p.prompt_id,), scores=(batch.scores[i],),
                    token_counts=(batch.token_counts[i],),
                    embeddings=batch.embeddings[i:i + 1] if batch.embeddings is not None else None)
            for i, p in enumerate(prompts)], [None] * len(prompts)


def mean_present(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def aggregate_present(rows, field):
    import numpy as np
    values = [row[field] for row in rows if row.get(field) is not None]
    return {"mean": float(np.mean(values)) if values else None,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "count": len(values)}
