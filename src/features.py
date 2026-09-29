from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


_WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
_SENTENCE_RE = re.compile(r"[.!?]+")


def _stats(text: str) -> list[float]:
    words = _WORD_RE.findall(text)
    word_count = len(words)
    char_count = len(text)
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    sentence_count = len(sentences)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    avg_sentence_length = word_count / max(sentence_count, 1)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    unique_ratio = len({w.lower() for w in words}) / max(word_count, 1)

    return [
        float(word_count),
        float(char_count),
        float(sentence_count),
        float(avg_sentence_length),
        float(avg_word_length),
        float(unique_ratio),
        float(len(paragraphs)),
    ]


class EssayStatsTransformer(BaseEstimator, TransformerMixin):
    """Convert raw essay strings into a small numerical feature matrix."""

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]):
        rows = [_stats(str(text)) for text in X]
        return csr_matrix(np.asarray(rows, dtype=np.float64))


def build_feature_union() -> FeatureUnion:
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=20000,
    )

    stats = StandardScaler(with_mean=False)

    # stats pipeline is built as a tuple inside FeatureUnion below.
    from sklearn.pipeline import Pipeline

    stats_pipeline = Pipeline(
        [
            ("stats", EssayStatsTransformer()),
            ("scale", stats),
        ]
    )

    return FeatureUnion(
        [
            ("tfidf", tfidf),
            ("stats", stats_pipeline),
        ]
    )
