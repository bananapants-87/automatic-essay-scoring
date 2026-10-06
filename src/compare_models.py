from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor


def qwk(y_true: np.ndarray, y_pred: np.ndarray, score_min: float, score_max: float) -> float:
    from sklearn.metrics import cohen_kappa_score

    rounded = np.clip(np.rint(y_pred * 2) / 2, score_min, score_max)
    return float(
        cohen_kappa_score(
            np.rint(y_true * 2).astype(int),
            np.rint(rounded * 2).astype(int),
            weights="quadratic",
        )
    )


def make_text(df: pd.DataFrame) -> pd.Series:
    prompt = df.get("prompt", pd.Series("", index=df.index)).fillna("").astype(str)
    essay = df["essay"].fillna("").astype(str)
    return "PROMPT: " + prompt + " ESSAY: " + essay


def base_features() -> FeatureUnion:
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def model_pipelines() -> dict[str, object]:
    features = base_features()
    return {
        "Ridge": Pipeline(
            [("features", features), ("model", Ridge(alpha=3.0))]
        ),
        "LinearSVR": Pipeline(
            [
                ("features", base_features()),
                ("model", LinearSVR(C=1.0, epsilon=0.1, max_iter=10000, random_state=42)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("features", base_features()),
                ("model", ElasticNet(alpha=0.0005, l1_ratio=0.15, max_iter=5000, random_state=42)),
            ]
        ),
        "SGDRegressor": Pipeline(
            [
                ("features", base_features()),
                ("model", SGDRegressor(
                    loss="squared_error",
                    penalty="elasticnet",
                    alpha=0.0001,
                    l1_ratio=0.15,
                    max_iter=2000,
                    early_stopping=True,
                    random_state=42,
                )),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare AES regression models")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--results", default="results/model_comparison.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = {"essay", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["essay"] = df["essay"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["essay"].str.strip().ne("") & df["score"].notna()].copy()
    df = df.drop_duplicates(subset=["essay"]).reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("Need at least 30 scored essays for a meaningful comparison.")

    X = make_text(df)
    y = df["score"].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    score_min = float(y_train.min())
    score_max = float(y_train.max())
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for name, pipeline in model_pipelines().items():
        pipeline.fit(X_train, y_train)
        pred = np.asarray(pipeline.predict(X_test), dtype=float)

        rows.append(
            {
                "model": name,
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "mae": float(mean_absolute_error(y_test, pred)),
                "qwk": qwk(y_test, pred, score_min, score_max),
                "n_test": int(len(y_test)),
            }
        )

        joblib.dump(
            {
                "name": name,
                "pipeline": pipeline,
                "score_min": score_min,
                "score_max": score_max,
                "test_size": args.test_size,
                "random_state": args.random_state,
            },
            model_dir / f"scraped_{name.lower()}.joblib",
        )

    result_df = pd.DataFrame(rows).sort_values("qwk", ascending=False)
    result_df.to_csv(args.results, index=False)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
