from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import EssayStatsTransformer, build_feature_union


def test_stats_shape():
    X = EssayStatsTransformer().fit_transform(["This is one sentence."])
    assert X.shape == (1, 7)
    assert np.all(X.toarray() >= 0)


def test_feature_union_runs():
    X = build_feature_union().fit_transform([
        "This is a test essay.",
        "Another essay with more words and more detail.",
    ])
    assert X.shape[0] == 2
    assert X.shape[1] > 7
