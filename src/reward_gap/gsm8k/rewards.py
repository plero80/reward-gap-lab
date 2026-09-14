"""Common format/completion penalties applied after each arm's task reward."""

from types import SimpleNamespace

from reward_gap.gsm8k.answers import boxed
from reward_gap.memory import MemoryContext
from reward_gap.failures import SampleError


class MathReward:
    recover_sample_failures = True
    def __init__(self, arm, proxy, judge, calibration, memory, settings):
        if arm not in ("proxy", "judge", "knn"):
            raise ValueError("Unknown GSM8K PPO arm")
        self.arm, self.proxy, self.judge = arm, proxy, judge
        self.calibration, self.memory, self.settings = calibration, memory, settings
        self.rows = []

    def score(self, prompts, answers):
        raise ValueError("GSM8K training rewards need token-derived completion metadata")

    def score_rollouts(self, prompts, answers, *, response_lengths, finish_reasons):
        if any(len(values) != len(prompts) for values in (answers, response_lengths, finish_reasons)):
            raise ValueError("Rollout metadata does not align")
        scorer = self.judge if self.arm == "judge" else self.proxy
        try:
            batch = scorer.score(prompts, answers, return_embeddings=self.arm == "knn")
        except SampleError as exc:
            self.rows.extend({"question_id": p.prompt_id, "answer": a, "arm": self.arm,
                              "response_tokens": n, "finish_reason": reason,
                              "reward": None, "status": "batch_skipped", "error": str(exc)}
                             for p, a, n, reason in zip(prompts, answers, response_lengths, finish_reasons, strict=True))
            raise
        if batch.prompt_ids != tuple(p.prompt_id for p in prompts):
            raise ValueError("Grader changed prompt alignment")
        task = list(self.calibration.normalize_judge(batch) if self.arm == "judge" else self.calibration.normalize_proxy(batch))
        gaps, neighbors = [0.] * len(task), [None] * len(task)
        if self.arm == "knn":
            context = MemoryContext(batch.source, batch.revision, batch.embedding_pooling, self.calibration.calibration_id)
            gaps, neighbors = self.memory.predict(batch.embeddings, batch.prompt_ids, context=context)
            task = [value - gap for value, gap in zip(task, gaps, strict=True)]
        rewards = []
        for i, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
            if finish_reasons[i] not in ("eos", "length"):
                raise ValueError("Unknown finish reason")
            format_penalty = self.settings["format_penalty"] * (boxed(answer)[0] is None)
            length_penalty = self.settings["length_penalty"] * (finish_reasons[i] == "length")
            reward = task[i] - format_penalty - length_penalty
            rewards.append(reward)
            self.rows.append({"question_id": prompt.prompt_id, "answer": answer, "arm": self.arm,
                              "raw_grade": batch.scores[i], "task_reward": task[i], "predicted_gap": gaps[i],
                              "format_penalty": format_penalty, "length_penalty": length_penalty,
                              "reward": reward, "response_tokens": response_lengths[i],
                              "finish_reason": finish_reasons[i], "neighbors": neighbors[i]})
        return SimpleNamespace(prompt_ids=tuple(p.prompt_id for p in prompts), rewards=tuple(rewards))
