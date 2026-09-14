"""Exact cosine retrieval excluding every response to the query question."""

from dataclasses import asdict

import torch

from reward_gap.memory import GapMemory, MemoryContext


class QuestionMemory:
    def __init__(self, rows, embeddings, context, *, k, temperature):
        self.base = GapMemory([r["example_id"] for r in rows], embeddings, [r["gap"] for r in rows],
                              context=context, k=k, temperature=temperature)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["example_id"])
        self.rows = [rows[i] for i in order]
        self.vectors = embeddings[order].double().cpu()
        self.vectors = self.vectors / self.vectors.norm(dim=1, keepdim=True)
        self.context, self.k, self.temperature = context, k, temperature

    def predict(self, embeddings, question_ids, *, context):
        if context != self.context or embeddings.ndim != 2 or len(embeddings) != len(question_ids):
            raise ValueError("Memory context or query alignment differs")
        values, neighbors = [], []
        for query, question_id in zip(embeddings.double().cpu(), question_ids, strict=True):
            norm = query.norm()
            if not torch.isfinite(query).all() or norm <= 0:
                raise ValueError("Invalid memory query")
            eligible = [i for i, row in enumerate(self.rows) if row["question_id"] != question_id]
            if len(eligible) < self.k:
                raise ValueError("Too few neighbors after same-question exclusion")
            similarities = self.vectors[eligible] @ (query / norm)
            order = torch.argsort(similarities, descending=True, stable=True)[:self.k]
            chosen = [eligible[i] for i in order.tolist()]
            weights = torch.softmax((similarities[order] - similarities[order].max()) / self.temperature, dim=0)
            values.append(float(weights @ torch.tensor([self.rows[i]["gap"] for i in chosen], dtype=torch.float64)))
            neighbors.append({"example_ids": [self.rows[i]["example_id"] for i in chosen],
                              "question_ids": [self.rows[i]["question_id"] for i in chosen],
                              "weights": weights.tolist(), "similarities": similarities[order].tolist()})
        return values, neighbors

    def to_dict(self):
        return {"rows": self.rows, "embeddings": self.vectors.tolist(), "context": asdict(self.context),
                "k": self.k, "temperature": self.temperature}

    @classmethod
    def from_dict(cls, data):
        return cls(data["rows"], torch.tensor(data["embeddings"], dtype=torch.float64),
                   MemoryContext(**data["context"]), k=data["k"], temperature=data["temperature"])
