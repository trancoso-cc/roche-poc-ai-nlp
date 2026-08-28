"""
train_tiny_model.py
-------------------
Trains a tiny TF-IDF + Logistic Regression sentiment classifier and exports the
weights to a single portable JSON file (models/tiny_sentiment.json).

The exported file is small enough to commit to Git (<100 KB) and can be loaded
by local_nlp.py with NO scikit-learn dependency at inference time.

Run:  python train_tiny_model.py
"""

import json
import pathlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

OUT = pathlib.Path(__file__).parent / "models" / "tiny_sentiment.json"

# --- Tiny labelled corpus -------------------------------------------------
# 1 = positive, 0 = negative. Deliberately small and readable so the artefact
# is auditable: you can see exactly what the model was trained on.
DATA = [
    ("the response was clear and very helpful", 1),
    ("excellent support, resolved my issue quickly", 1),
    ("great experience overall, would recommend", 1),
    ("the documentation is well written and easy to follow", 1),
    ("fast turnaround and accurate information", 1),
    ("i am happy with the outcome", 1),
    ("the team was responsive and professional", 1),
    ("this works perfectly for our use case", 1),
    ("very satisfied with the quality of the answer", 1),
    ("the onboarding process was smooth and simple", 1),
    ("impressive accuracy and good performance", 1),
    ("thanks, this solved my problem", 1),
    ("the interface is intuitive and pleasant to use", 1),
    ("reliable service, no complaints at all", 1),
    ("a genuinely useful improvement", 1),
    ("the summary was concise and correct", 1),
    ("good job, everything went as expected", 1),
    ("clear explanation, exactly what i needed", 1),
    ("the update fixed the issue completely", 1),
    ("outstanding quality and attention to detail", 1),
    ("i appreciate how quickly this was handled", 1),
    ("the results are consistent and trustworthy", 1),
    ("simple to configure and works out of the box", 1),
    ("the training material was informative and useful", 1),
    ("well structured and easy to understand", 1),
    ("the process is efficient and saves time", 1),
    ("positive experience from start to finish", 1),
    ("the answer was accurate and well sourced", 1),
    ("stable, fast and easy to maintain", 1),
    ("really pleased with this outcome", 1),
    ("the escalation was handled professionally", 1),
    ("everything is working as documented", 1),
    ("the response was slow and unhelpful", 0),
    ("terrible experience, nothing worked", 0),
    ("the documentation is confusing and incomplete", 0),
    ("i am disappointed with the quality", 0),
    ("this is broken and needs to be fixed", 0),
    ("poor support, no one replied for days", 0),
    ("the answer was wrong and misleading", 0),
    ("frustrating process, far too complicated", 0),
    ("the system keeps failing without any warning", 0),
    ("very unhappy with the outcome", 0),
    ("unacceptable delay and no explanation", 0),
    ("the interface is clunky and hard to use", 0),
    ("it crashed again during the demo", 0),
    ("the information provided was outdated", 0),
    ("waste of time, i had to redo everything", 0),
    ("the results are inconsistent and unreliable", 0),
    ("bad performance and constant errors", 0),
    ("nobody could explain why it failed", 0),
    ("the setup was painful and poorly documented", 0),
    ("this made the problem worse", 0),
    ("the quality has clearly declined", 0),
    ("difficult to configure and badly designed", 0),
    ("i would not recommend this to anyone", 0),
    ("the output was irrelevant and confusing", 0),
    ("repeated failures with no root cause", 0),
    ("the service was unavailable all morning", 0),
    ("annoying bugs that block our work", 0),
    ("the summary missed the key points", 0),
    ("weak explanation and no supporting evidence", 0),
    ("too slow to be useful in practice", 0),
    ("the deployment failed and rolled back", 0),
    ("the provisioning failed with an unclear error", 0),
    ("the answer was helpful and grounded in the source", 1),
    ("this is a solid and dependable solution", 1),
    ("the escalation path was unclear and slow", 0),
    ("great value and strong technical depth", 1),
    ("the report contained factual errors", 0),
    ("very smooth integration with our systems", 1),
    ("the model returned an incorrect classification", 0),
]


def main() -> None:
    texts = [t for t, _ in DATA]
    labels = [y for _, y in DATA]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=False,
        norm="l2",
        smooth_idf=True,
    )
    clf = LogisticRegression(C=4.0, max_iter=2000)
    pipeline = make_pipeline(vectorizer, clf)
    pipeline.fit(texts, labels)

    print(f"train accuracy: {pipeline.score(texts, labels):.3f}")

    weights = {
        "name": "tiny-sentiment-tfidf-logreg",
        "version": "1.0.0",
        "task": "binary-sentiment",
        "labels": {"0": "negative", "1": "positive"},
        "tokenizer": {"lowercase": True, "pattern": r"(?u)\b\w\w+\b", "ngram_range": [1, 2]},
        "vectorizer": {"norm": "l2", "sublinear_tf": False},
        "vocabulary": {term: int(i) for term, i in vectorizer.vocabulary_.items()},
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "coef": [round(float(v), 6) for v in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(weights['coef'])} features)")


if __name__ == "__main__":
    main()
