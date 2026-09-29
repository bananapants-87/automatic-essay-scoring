from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import build_feature_union



def train_one(df: pd.DataFrame, essay_set: int, model_dir: Path, test_size: float, random_state: int) -> None:
    subset = df[df["essay_set"] == essay_set].copy()
    if len(subset) < 10:
        raise ValueError(f"Essay set {essay_set} has only {len(subset)} rows; need at least 10.")

    X = subset["essay"].astype(str)
    y = subset["domain1_score"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline(
        [
            ("features", build_feature_union()),
            ("model", Ridge(alpha=3.0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    artifact = {
        "essay_set": essay_set,
        "pipeline": pipeline,
        "score_min": float(y_train.min()),
        "score_max": float(y_train.max()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"aes_prompt_{essay_set}.joblib"
    joblib.dump(artifact, path)

    print(
        f"Essay set {essay_set}: trained on {len(X_train):,}, "
        f"held out {len(X_test):,}; score range {y.min():g}-{y.max():g}; saved {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a prompt-specific AES model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--essay-set", type=int)
    parser.add_argument("--all-sets", action="store_true")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.all_sets and args.essay_set is None:
        parser.error("Provide --essay-set N or --all-sets")

    df = pd.read_csv(args.data)
    model_dir = Path(args.model_dir)

    sets = sorted(df["essay_set"].unique()) if args.all_sets else [args.essay_set]
    for essay_set in sets:
        train_one(df, int(essay_set), model_dir, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
