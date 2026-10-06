from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_ID = "chillies/ielts-writing-task2-essays"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a licensed scored essay corpus from the web via Hugging Face."
    )
    parser.add_argument(
        "--output",
        default="data/scraped/essays.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows for a quick test.",
    )
    args = parser.parse_args()

    dataset = load_dataset(DATASET_ID, split="train")
    df = dataset.to_pandas()

    df = df[
        [
            "essay_id",
            "topic",
            "question",
            "essay_text",
            "overall_band",
            "task_achievement_band",
            "coherence_cohesion_band",
            "lexical_resource_band",
            "grammatical_range_band",
        ]
    ].rename(
        columns={
            "essay_text": "essay",
            "overall_band": "score",
            "question": "prompt",
        }
    )

    df["essay"] = df["essay"].fillna("").astype(str)
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df[
        df["essay"].str.strip().ne("")
        & df["score"].notna()
    ].copy()

    df = df.drop_duplicates(subset=["essay"]).reset_index(drop=True)

    if args.limit is not None:
        if args.limit < 30:
            raise ValueError("--limit must be at least 30 for model evaluation.")
        df = df.head(args.limit).copy()

    # Preserve provenance and the criterion-level labels for later analysis.
    df["source_url"] = "https://huggingface.co/datasets/chillies/ielts-writing-task2-essays"
    df["source_name"] = "Hugging Face / Writing9 IELTS Task 2 corpus"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved {len(df):,} scored essays to {output}")
    print(f"Score range: {df['score'].min():g}-{df['score'].max():g}")
    print("Topics:")
    print(df["topic"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
