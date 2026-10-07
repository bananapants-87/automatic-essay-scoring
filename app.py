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
ASAP_DATA = ROOT / "data" / "processed" / "asap.csv"
SCRAPED_DATA = ROOT / "data" / "scraped" / "uol_essays.csv"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
SENTENCE_RE = re.compile(r"[.!?]+")

st.set_page_config(
    page_title="AES Analytics | Automatic Essay Scoring",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background:#f6f8fb; }
    .block-container { max-width:1450px; padding-top:2rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] { background:#111827; border-right:1px solid #263244; }
    [data-testid="stSidebar"] * { color:#e5e7eb; }
    .hero { background:linear-gradient(135deg,#111827,#1f2937); border-radius:18px;
            padding:30px 34px; margin-bottom:24px; box-shadow:0 10px 30px rgba(15,23,42,.10); }
    .eyebrow { color:#93c5fd; font-size:.76rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }
    .hero h1 { color:#fff; font-size:2.35rem; margin:7px 0 0; line-height:1.1; }
    .hero p { color:#cbd5e1; max-width:850px; margin:10px 0 0; }
    .section-label { color:#64748b; font-size:.74rem; font-weight:750; letter-spacing:.12em;
                     text-transform:uppercase; margin:18px 0 8px; }
    .score-card { background:#fff; border:1px solid #e2e8f0; border-radius:18px; padding:24px;
                  min-height:245px; box-shadow:0 6px 22px rgba(15,23,42,.06); }
    .score-label { color:#64748b; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
    .score-value { color:#0f172a; font-size:3.6rem; line-height:1; font-weight:800; margin:12px 0 6px; }
    .score-caption { color:#64748b; font-size:.9rem; }
    .pipeline { background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:16px 20px;
                margin:18px 0 24px; color:#334155; text-align:center; font-weight:650; }
    .pipeline span { color:#94a3b8; margin:0 9px; }
    .status-pill { display:inline-block; background:#dcfce7; color:#166534; border-radius:999px;
                   padding:5px 10px; font-size:.73rem; font-weight:700; }
    .footer { color:#94a3b8; font-size:.78rem; text-align:center; margin-top:28px; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #e2e8f0; border-radius:14px;
                                    padding:14px 16px; box-shadow:0 4px 14px rgba(15,23,42,.04); }
    </style>
    """,
    unsafe_allow_html=True,
)


def essay_stats(text: str) -> dict[str, float]:
    words = WORD_RE.findall(text)
    sentences = [s for s in SENTENCE_RE.split(text) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    wc, sc = len(words), len(sentences)
    return {
        "Words": float(wc),
        "Characters": float(len(text)),
        "Sentences": float(sc),
        "Paragraphs": float(len(paragraphs)),
        "Avg. words / sentence": float(wc / max(sc, 1)),
        "Avg. word length": float(sum(map(len, words)) / max(wc, 1)),
        "Unique-word ratio": float(len({w.lower() for w in words}) / max(wc, 1)),
    }


@st.cache_resource
def load_model(name: str):
    path = MODELS / f"scraped_{name.lower()}.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    path = RESULTS / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


comparison = load_comparison()
available_models = comparison["model"].tolist() if comparison is not None and not comparison.empty else []

with st.sidebar:
    st.markdown("## AES Analytics")
    st.caption("Automatic Essay Scoring")
    st.divider()
    if available_models:
        best_model = str(comparison.loc[comparison["qwk"].idxmax(), "model"])
        selected = st.multiselect("Models to run", available_models, default=available_models)
        st.caption(f"Best QWK model: **{best_model}**")
    else:
        selected = []
    st.divider()
    st.markdown("**Pipeline**")
    st.caption("Dataset / HTML → preprocessing → TF-IDF → regression → score")
    if SCRAPED_DATA.exists():
        try:
            n_scraped = len(pd.read_csv(SCRAPED_DATA))
        except Exception:
            n_scraped = None
        st.markdown('<span class="status-pill">● Scraped data available</span>', unsafe_allow_html=True)
        if n_scraped is not None:
            st.caption(f"{n_scraped:,} scraped essays")
    else:
        st.caption("Scraped dataset not found locally")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">NLP / Regression / Model Evaluation</div>
      <h1>Automatic Essay Scoring</h1>
      <p>Evaluate essays with TF-IDF-based regression models, inspect held-out performance, and generate score estimates from trained pipelines.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not available_models:
    st.warning("No trained scraped-model results found. Run `src/compare_models.py` after preparing the scored dataset.")
    st.stop()

tab_score, tab_models, tab_data, tab_about = st.tabs(["Score an Essay", "Model Performance", "Data", "How It Works"])

with tab_score:
    st.markdown('<div class="section-label">Prediction workspace</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, .85], gap="large")

    with left:
        st.markdown("### Essay input")
        essay = st.text_area(
            "Essay", height=430, label_visibility="collapsed",
            placeholder="Paste the full essay here...\n\nThe trained models will convert it to TF-IDF features and estimate a score.",
        )

    predictions: list[dict[str, object]] = []
    pred_df = pd.DataFrame()

    with right:
        st.markdown("### Prediction")
        if essay.strip():
            for name in selected:
                artifact = load_model(name)
                if artifact is None:
                    continue
                raw = float(np.asarray(artifact["pipeline"].predict([essay])).ravel()[0])
                rounded = float(np.clip(np.rint(raw * 2) / 2, artifact["score_min"], artifact["score_max"]))
                predictions.append({"Model": name, "Raw score": raw, "Predicted score": rounded})

            if predictions:
                pred_df = pd.DataFrame(predictions)
                consensus = float(pred_df["Predicted score"].median())
                best_model = str(comparison.loc[comparison["qwk"].idxmax(), "model"])
                best_row = pred_df[pred_df["Model"] == best_model]
                primary = float(best_row["Predicted score"].iloc[0]) if not best_row.empty else consensus
                artifact = load_model(best_model)
                lo = float(artifact["score_min"]) if artifact else 0.0
                hi = float(artifact["score_max"]) if artifact else 10.0
                progress = float(np.clip((primary - lo) / max(hi - lo, 1e-9), 0, 1))
                st.markdown(
                    f'<div class="score-card"><div class="score-label">Primary estimate · {best_model}</div>'
                    f'<div class="score-value">{primary:g}</div>'
                    f'<div class="score-caption">Consensus across selected models: <b>{consensus:g}</b></div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(progress, text=f"Score position · {lo:g} to {hi:g}")
                st.markdown("**Model estimates**")
                st.dataframe(pred_df.style.format({"Raw score": "{:.2f}", "Predicted score": "{:.1f}"}), hide_index=True, use_container_width=True)
            else:
                st.info("Select at least one trained model in the sidebar.")
        else:
            st.markdown(
                '<div class="score-card"><div class="score-label">Waiting for essay</div>'
                '<div class="score-value">—</div><div class="score-caption">Paste an essay on the left to generate predictions.</div></div>',
                unsafe_allow_html=True,
            )

    if essay.strip():
        stats = essay_stats(essay)
        st.markdown('<div class="section-label">Essay diagnostics</div>', unsafe_allow_html=True)
        cols = st.columns(7)
        items = [
            ("Words", f"{int(stats['Words']):,}"),
            ("Characters", f"{int(stats['Characters']):,}"),
            ("Sentences", f"{int(stats['Sentences']):,}"),
            ("Paragraphs", f"{int(stats['Paragraphs']):,}"),
            ("Avg. words / sentence", f"{stats['Avg. words / sentence']:.1f}"),
            ("Avg. word length", f"{stats['Avg. word length']:.1f}"),
            ("Unique-word ratio", f"{stats['Unique-word ratio']:.2f}"),
        ]
        for col, (label, value) in zip(cols, items):
            col.metric(label, value)
        if predictions:
            st.markdown('<div class="section-label">Prediction comparison</div>', unsafe_allow_html=True)
            st.bar_chart(pred_df.set_index("Model")[["Predicted score"]], use_container_width=True)
        st.info("This is a learned estimate. QWK, MAE and RMSE are model-level test metrics and cannot be calculated for one essay without a human reference score.")
    else:
        st.markdown('<div class="pipeline">Essay <span>→</span> preprocessing <span>→</span> TF-IDF <span>→</span> regression <span>→</span> predicted score</div>', unsafe_allow_html=True)

with tab_models:
    st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
    st.markdown("### Model performance")
    st.caption("Metrics come from the held-out test split produced by `src/compare_models.py`.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best QWK", comparison.loc[comparison["qwk"].idxmax(), "model"])
    c2.metric("Lowest MAE", comparison.loc[comparison["mae"].idxmin(), "model"])
    c3.metric("Lowest RMSE", comparison.loc[comparison["rmse"].idxmin(), "model"])
    table = comparison[["model", "qwk", "mae", "rmse", "n_test"]].rename(columns={"model":"Model", "qwk":"QWK", "mae":"MAE", "rmse":"RMSE", "n_test":"Test essays"})
    st.dataframe(table.round({"QWK":3, "MAE":3, "RMSE":3}), hide_index=True, use_container_width=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Quadratic weighted kappa")
        st.bar_chart(comparison.set_index("model")[["qwk"]], use_container_width=True)
    with b:
        st.markdown("#### Prediction error")
        st.bar_chart(comparison.set_index("model")[["mae", "rmse"]], use_container_width=True)

with tab_data:
    st.markdown('<div class="section-label">Training data</div>', unsafe_allow_html=True)
    st.markdown("### Dataset status")
    a, b = st.columns(2)
    if SCRAPED_DATA.exists():
        try:
            scraped = pd.read_csv(SCRAPED_DATA)
            a.metric("Scraped essays", f"{len(scraped):,}")
            a.caption("data/scraped/uol_essays.csv")
            st.markdown("#### Scraped dataset preview")
            cols = [c for c in ["essay", "score", "prompt", "source_url", "source_name"] if c in scraped.columns]
            st.dataframe(scraped[cols], hide_index=True, use_container_width=True)
        except Exception as exc:
            a.error(f"Could not read scraped dataset: {exc}")
    else:
        a.metric("Scraped essays", "—")
    if ASAP_DATA.exists():
        try:
            asap = pd.read_csv(ASAP_DATA)
            b.metric("ASAP-AES essays", f"{len(asap):,}")
            b.caption("data/processed/asap.csv")
        except Exception as exc:
            b.error(f"Could not read ASAP dataset: {exc}")
    else:
        b.metric("ASAP-AES essays", "—")

with tab_about:
    st.markdown('<div class="section-label">System overview</div>', unsafe_allow_html=True)
    st.markdown("### How the scoring system works")
    st.markdown(
        '<div class="pipeline">Essay text <span>→</span> prompt + essay representation <span>→</span> '
        'word & character TF-IDF <span>→</span> regression model <span>→</span> score estimate</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Feature representation")
        st.write("The training pipeline combines word-level and character-level TF-IDF features. Word n-grams capture lexical patterns; character n-grams capture subword and stylistic patterns.")
        st.markdown("#### Models")
        st.write("The current comparison includes Ridge, LinearSVR, ElasticNet, and SGDRegressor. The dashboard highlights the model with the strongest stored QWK.")
    with b:
        st.markdown("#### Evaluation")
        st.write("QWK measures agreement with human scores, while MAE and RMSE quantify prediction error. These are test-set metrics, not confidence scores for an individual essay.")
        st.markdown("#### Important limitation")
        st.write("Predictions depend on the training data and scoring rubric. Incompatible score scales should not be mixed without appropriate normalization.")

st.markdown('<div class="footer">Automatic Essay Scoring · NLP regression dashboard</div>', unsafe_allow_html=True)
