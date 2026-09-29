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
DATA = ROOT / "data" / "processed" / "asap.csv"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metrics import regression_metrics  # noqa: E402


WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
SENTENCE_RE = re.compile(r"[.!?]+")


st.set_page_config(
    page_title="Automatic Essay Scoring",
    page_icon="📝",
    layout="wide",
)


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
        "Avg. word length": float(
            sum(len(w) for w in words) / max(word_count, 1)
        ),
        "Unique-word ratio": float(
            len({w.lower() for w in words}) / max(word_count, 1)
        ),
        "Paragraphs": float(len(paragraphs)),
    }


@st.cache_resource
def load_model(essay_set: int):
    path = MODELS / f"aes_prompt_{essay_set}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def evaluate_model(essay_set: int):
    if not DATA.exists():
        return None

    from sklearn.model_selection import train_test_split

    artifact = load_model(essay_set)
    if artifact is None:
        return None

    df = pd.read_csv(DATA)
    subset = df[df["essay_set"] == essay_set].copy()

    _, x_test, _, y_test = train_test_split(
        subset["essay"].astype(str),
        subset["domain1_score"].astype(float),
        test_size=float(artifact["test_size"]),
        random_state=int(artifact["random_state"]),
    )

    y_pred = artifact["pipeline"].predict(x_test)
    return regression_metrics(
        np.asarray(y_test),
        np.asarray(y_pred),
        artifact["score_min"],
        artifact["score_max"],
    ) | {"n_test": int(len(y_test))}


st.title("Automatic Essay Scoring")
st.caption("TF-IDF + essay statistics + Ridge regression on the ASAP dataset")

available_sets = sorted(
    int(p.stem.split("_")[-1])
    for p in MODELS.glob("aes_prompt_*.joblib")
    if p.stem.split("_")[-1].isdigit()
)

if not available_sets:
    st.error("No trained models found in models/. Train at least one essay prompt first.")
    st.stop()

essay_set = st.selectbox(
    "Essay prompt / set",
    available_sets,
    format_func=lambda x: f"Prompt {x}",
)

artifact = load_model(essay_set)
metrics = evaluate_model(essay_set)

st.subheader("Model performance")

if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QWK", f"{metrics['qwk']:.3f}")
    c2.metric("MAE", f"{metrics['mae']:.3f}")
    c3.metric("RMSE", f"{metrics['rmse']:.3f}")
    c4.metric("Test essays", f"{metrics['n_test']:,}")

    st.progress(
        min(max(float(metrics["qwk"]), 0.0), 1.0),
        text=f"Quadratic Weighted Kappa: {metrics['qwk']:.3f}",
    )

st.divider()

st.subheader("Paste your essay")

essay = st.text_area(
    "Essay text",
    height=420,
    placeholder="Paste a long essay here...",
    label_visibility="collapsed",
)

if essay.strip():
    raw_score = float(np.asarray(artifact["pipeline"].predict([essay])).ravel()[0])
    rounded_score = int(
        np.clip(
            np.rint(raw_score),
            artifact["score_min"],
            artifact["score_max"],
        )
    )

    stats = essay_stats(essay)

    left, right = st.columns([1, 2])

    with left:
        st.metric("Predicted score", f"{rounded_score:g}")
        st.caption(f"Raw model estimate: {raw_score:.2f}")
        st.caption(
            f"Valid training range: "
            f"{artifact['score_min']:g}–{artifact['score_max']:g}"
        )

        score_span = artifact["score_max"] - artifact["score_min"]
        score_position = (
            (raw_score - artifact["score_min"]) / score_span
            if score_span > 0
            else 0.0
        )
        st.progress(
            min(max(score_position, 0.0), 1.0),
            text="Position within training score range",
        )

    with right:
        st.markdown("**Essay diagnostics**")
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

    st.info(
        "The score above is a model prediction, not a definitive assessment of writing quality. "
        "QWK/MAE/RMSE shown above describe the trained model's held-out test performance; "
        "they cannot be computed for this pasted essay unless a human reference score is available."
    )
else:
    st.info("Paste an essay above to get a predicted score and essay diagnostics.")
