from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np



def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an essay score")
    parser.add_argument("--essay-set", type=int, required=True)
    parser.add_argument("--essay")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    model_path = Path(args.model_dir) / f"aes_prompt_{args.essay_set}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py for essay set {args.essay_set} first."
        )

    essay = args.essay if args.essay is not None else input("Paste essay:\n").strip()
    if not essay:
        raise ValueError("Essay cannot be empty.")

    artifact = joblib.load(model_path)
    raw_score = float(np.asarray(artifact["pipeline"].predict([essay])).ravel()[0])
    rounded_score = int(np.clip(np.rint(raw_score), artifact["score_min"], artifact["score_max"]))

    print(f"Essay set: {args.essay_set}")
    print(f"Raw model score: {raw_score:.2f}")
    print(f"Predicted score: {rounded_score}")
    print(f"Valid score range seen during training: {artifact['score_min']:g}-{artifact['score_max']:g}")


if __name__ == "__main__":
    main()
