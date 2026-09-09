from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from apple_agent import textutil as T


class TfidfRetriever:
    def __init__(self, docs: list[dict], vectorizer: TfidfVectorizer, matrix):
        self.docs = docs
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def build(cls, docs: list[dict], max_features: int = 40000) -> "TfidfRetriever":
        texts = [T.normalize(d["text"]) for d in docs]
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
        )
        matrix = vec.fit_transform(texts)
        return cls(docs, vec, matrix)

    def query(self, text: str, k: int = 8, intent: str | None = None) -> list[dict]:
        q = self.vectorizer.transform([T.normalize(text)])
        sims = cosine_similarity(q, self.matrix).ravel()
        if intent:
            boost = np.array(
                [0.08 if d.get("weak_intent") == intent else 0.0 for d in self.docs]
            )
            sims = sims + boost
        k = min(k, len(self.docs))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out = []
        for i in idx:
            row = dict(self.docs[i])
            row["score"] = float(sims[i])
            out.append(row)
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "tfidf.joblib")
        joblib.dump(self.matrix, directory / "matrix.joblib")
        with gzip.open(directory / "docs.jsonl.gz", "wt", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "TfidfRetriever":
        vec = joblib.load(directory / "tfidf.joblib")
        matrix = joblib.load(directory / "matrix.joblib")
        docs = []
        with gzip.open(directory / "docs.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
        return cls(docs, vec, matrix)
