from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from metrics import regression_metrics


def evaluate_one(df: pd.DataFrame, essay_set: int, model_dir: Path) -> dict[str, float]:
    artifact = joblib.load(model_dir / f"aes_prompt_{essay_set}.joblib")
    subset = df[df["essay_set"] == essay_set].copy()
    if subset.empty:
        raise ValueError(f"No rows found for essay set {essay_set}.")

    # Use the same deterministic split as train.py so this evaluates the held-out portion.
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        subset["essay"].astype(str),
        subset["domain1_score"].astype(float),
        test_size=float(artifact["test_size"]),
        random_state=int(artifact["random_state"]),
    )

    y_pred = artifact["pipeline"].predict(X_test)
    metrics = regression_metrics(
        np.asarray(y_test),
        np.asarray(y_pred),
        artifact["score_min"],
        artifact["score_max"],
    )
    metrics["n_test"] = float(len(y_test))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained AES models")
    parser.add_argument("--data", required=True)
    parser.add_argument("--essay-set", type=int)
    parser.add_argument("--all-sets", action="store_true")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    if not args.all_sets and args.essay_set is None:
        parser.error("Provide --essay-set N or --all-sets")

    df = pd.read_csv(args.data)
    model_dir = Path(args.model_dir)
    sets = sorted(df["essay_set"].unique()) if args.all_sets else [args.essay_set]

    for essay_set in sets:
        metrics = evaluate_one(df, int(essay_set), model_dir)
        print(
            f"Essay set {essay_set}: "
            f"RMSE={metrics['rmse']:.3f} | "
            f"MAE={metrics['mae']:.3f} | "
            f"QWK={metrics['qwk']:.3f} | "
            f"n_test={int(metrics['n_test'])}"
        )


if __name__ == "__main__":
    main()
