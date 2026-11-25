"""Baseline 1: TF-IDF + logistic regression.

Two modes:
    * ``pair``  : input is "claim [SEP] evidence" (a naive claim-verification baseline)
    * ``single``: input is the text alone (a style baseline: fake vs real)

Character n-grams (2-5) are used alongside word n-grams because they work
for Devanagari and romanised Hindi without a language-specific tokenizer.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

SEP = " [SEP] "


def example_to_text(row: dict, mode: str) -> str:
    if mode == "pair":
        return row["claim"] + SEP + row["evidence"]
    if mode == "single":
        return row.get("text") or row["claim"]
    raise ValueError(f"mode must be 'pair' or 'single', got {mode!r}")


def build_model(max_features: int = 200_000, C: float = 4.0) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 max_features=max_features, sublinear_tf=True,
                                 token_pattern=r"(?u)[^\s.,!?।;:\"'()]+")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=max_features, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def train_tfidf(train_rows: list[dict], mode: str, **kwargs) -> Pipeline:
    model = build_model(**kwargs)
    X = [example_to_text(r, mode) for r in train_rows]
    y = [r["label"] for r in train_rows]
    model.fit(X, y)
    return model


def predict_tfidf(model: Pipeline, rows: list[dict], mode: str) -> list[str]:
    return list(model.predict([example_to_text(r, mode) for r in rows]))


def save(model: Pipeline, path: str | Path, mode: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "mode": mode}, path)


def load(path: str | Path) -> tuple[Pipeline, str]:
    obj = joblib.load(path)
    return obj["model"], obj["mode"]
