from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"essay_id", "essay_set", "essay", "domain1_score"}


def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[["essay_id", "essay_set", "essay", "domain1_score"]].copy()
    out["essay"] = out["essay"].fillna("").astype(str)
    out["essay_set"] = pd.to_numeric(out["essay_set"], errors="raise").astype(int)
    out["domain1_score"] = pd.to_numeric(out["domain1_score"], errors="raise")
    out = out[out["essay"].str.strip().ne("")].reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASAP AES data")
    parser.add_argument("--input", required=True, help="Raw .xlsx, .xls, .tsv or .csv file")
    parser.add_argument("--output", required=True, help="Output canonical CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = prepare_dataframe(load_input(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} essays to {output_path}")
    print("Essay sets:")
    print(df["essay_set"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
