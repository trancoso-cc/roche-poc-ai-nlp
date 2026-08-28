"""
local_nlp.py
------------
Offline sentiment inference using the tiny model weights in
models/tiny_sentiment.json.

No scikit-learn, no numpy, no network calls. Standard library only.
Use this when the Azure endpoint is unavailable, or as an air-gapped fallback.

Usage:
    python local_nlp.py "the response was clear and very helpful"
    python local_nlp.py            # runs a few built-in examples
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys

WEIGHTS_PATH = pathlib.Path(__file__).parent / "models" / "tiny_sentiment.json"
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


class TinySentimentModel:
    """TF-IDF + logistic regression scorer, reimplemented in plain Python."""

    def __init__(self, weights_path: pathlib.Path = WEIGHTS_PATH):
        w = json.loads(weights_path.read_text(encoding="utf-8"))
        self.name = w["name"]
        self.version = w["version"]
        self.labels = w["labels"]
        self.vocabulary = w["vocabulary"]
        self.idf = w["idf"]
        self.coef = w["coef"]
        self.intercept = w["intercept"]
        self.ngram_max = w["tokenizer"]["ngram_range"][1]

    def _terms(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(text.lower())
        terms = list(tokens)
        for n in range(2, self.ngram_max + 1):
            terms += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return terms

    def _vector(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for term in self._terms(text):
            idx = self.vocabulary.get(term)
            if idx is not None:
                counts[idx] = counts.get(idx, 0.0) + 1.0
        vec = {i: c * self.idf[i] for i, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {i: v / norm for i, v in vec.items()} if norm else vec

    def predict(self, text: str) -> dict:
        vec = self._vector(text)
        z = self.intercept + sum(v * self.coef[i] for i, v in vec.items())
        prob_positive = 1.0 / (1.0 + math.exp(-z))
        label = "positive" if prob_positive >= 0.5 else "negative"
        return {
            "text": text,
            "sentiment": label,
            "confidence": round(max(prob_positive, 1 - prob_positive), 4),
            "scores": {
                "positive": round(prob_positive, 4),
                "negative": round(1 - prob_positive, 4),
            },
            "matched_features": len(vec),
            "model": f"{self.name}@{self.version}",
        }


def main() -> None:
    model = TinySentimentModel()
    inputs = sys.argv[1:] or [
        "the documentation is well written and easy to follow",
        "the provisioning failed with an unclear error",
    ]
    for text in inputs:
        print(json.dumps(model.predict(text), indent=2))


if __name__ == "__main__":
    main()
