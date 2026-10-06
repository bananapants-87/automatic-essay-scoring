from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed" / "asap.csv"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metrics import regression_metrics  # noqa: E402

WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
SENTENCE_RE = re.compile(r"[.!?]+")

st.set_page_config(page_title="Automatic Essay Scoring", page_icon="📝", layout="wide")


def essay_stats(text: str) -> dict[str, float]:
    words = WORD_RE.findall(text)
    sentences = [s for s in SENTENCE_RE.split(text) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    word_count = len(words)
    sentence_count = len(sentences)
    return {
        "Words": float(word_count),
        "Characters": float(len(text)),
        "Sentences": float(sentence_count),
        "Avg. words / sentence": float(word_count / max(sentence_count, 1)),
        "Avg. word length": float(sum(len(w) for w in words) / max(word_count, 1)),
        "Unique-word ratio": float(len({w.lower() for w in words}) / max(word_count, 1)),
        "Paragraphs": float(len(paragraphs)),
    }


@st.cache_resource
def load_scraped_model(name: str):
    path = MODELS / f"scraped_{name.lower()}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_comparison() -> pd.DataFrame | None:
    path = RESULTS / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


comparison = load_comparison()
available_models = (
    comparison["model"].tolist()
    if comparison is not None and not comparison.empty
    else []
)

st.title("Automatic Essay Scoring")
st.caption("Web-collected scored essays → multiple NLP regressors → comparative scoring dashboard")

if not available_models:
    st.warning(
        "No scraped-model comparison results found yet. "
        "First scrape an authorized source, then run src/compare_models.py."
    )
    st.stop()

st.subheader("Model Performance")
st.dataframe(
    comparison[["model", "qwk", "mae", "rmse", "n_test"]]
    .rename(
        columns={
            "model": "Model",
            "qwk": "QWK",
            "mae": "MAE",
            "rmse": "RMSE",
            "n_test": "Test essays",
        }
    )
    .round({"QWK": 3, "MAE": 3, "RMSE": 3}),
    hide_index=True,
    use_container_width=True,
)

chart = comparison.set_index("model")[["qwk", "mae", "rmse"]].rename(
    columns={"qwk": "QWK", "mae": "MAE", "rmse": "RMSE"}
)
st.bar_chart(chart)

best_qwk = comparison.loc[comparison["qwk"].idxmax(), "model"]
best_mae = comparison.loc[comparison["mae"].idxmin(), "model"]
best_rmse = comparison.loc[comparison["rmse"].idxmin(), "model"]

b1, b2, b3 = st.columns(3)
b1.metric("Best QWK", best_qwk)
b2.metric("Lowest MAE", best_mae)
b3.metric("Lowest RMSE", best_rmse)

st.divider()
st.subheader("Score a New Essay")

selected = st.multiselect(
    "Models to compare",
    available_models,
    default=available_models,
)

essay = st.text_area(
    "Paste a large essay",
    height=440,
    placeholder="Paste the full essay here...",
)

if essay.strip():
    stats = essay_stats(essay)
    st.markdown("### Essay diagnostics")
    stat_cols = st.columns(4)
    items = [
        ("Words", int(stats["Words"])),
        ("Sentences", int(stats["Sentences"])),
        ("Avg. words/sentence", f"{stats['Avg. words / sentence']:.1f}"),
        ("Paragraphs", int(stats["Paragraphs"])),
        ("Characters", int(stats["Characters"])),
        ("Avg. word length", f"{stats['Avg. word length']:.1f}"),
        ("Unique-word ratio", f"{stats['Unique-word ratio']:.2f}"),
    ]
    for i, (label, value) in enumerate(items):
        stat_cols[i % 4].metric(label, value)

    predictions = []
    for name in selected:
        artifact = load_scraped_model(name)
        if artifact is None:
            continue
        raw = float(np.asarray(artifact["pipeline"].predict([essay])).ravel()[0])
        rounded = float(
            np.clip(np.rint(raw * 2) / 2, artifact["score_min"], artifact["score_max"])
        )
        predictions.append(
            {"Model": name, "Raw score": raw, "Predicted score": rounded}
        )

    if predictions:
        pred_df = pd.DataFrame(predictions)
        st.markdown("### Model predictions")
        st.dataframe(
            pred_df.style.format({"Raw score": "{:.2f}", "Predicted score": "{:.1f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(pred_df.set_index("Model")[["Predicted score"]])

        best_pred = float(pred_df["Predicted score"].median())
        st.metric("Consensus score (median)", f"{best_pred:.1f}")

    st.info(
        "The prediction is a learned estimate from the scraped training examples. "
        "QWK, MAE and RMSE are model-level test metrics; they cannot be meaningfully "
        "calculated for this one essay without a human reference score."
    )
else:
    st.info("Paste an essay to see predictions from the selected models.")
