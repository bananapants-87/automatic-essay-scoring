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
    page_title="Automatic Essay Scoring",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual system: restrained, editorial, research-tool aesthetic.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink: #111827;
        --muted: #64748b;
        --line: #dfe5ec;
        --panel: #ffffff;
        --canvas: #f7f8fa;
        --accent: #2563eb;
        --accent-soft: #eff6ff;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { max-width: 1380px; padding: 2.2rem 3rem 4rem; }
    [data-testid="stHeader"] { background: rgba(247,248,250,.92); }
    [data-testid="stSidebar"] { background: #101827; border-right: 0; }
    [data-testid="stSidebar"] * { color: #dbe4ef; }
    [data-testid="stSidebar"] .stCaption { color: #94a3b8; }

    .brand { padding: .35rem 0 1.6rem; }
    .brand-mark { color: #ffffff; font-weight: 800; font-size: 1.05rem; letter-spacing: -.02em; }
    .brand-sub { color: #7f8da3; font-size: .73rem; margin-top: .2rem; letter-spacing: .05em; text-transform: uppercase; }
    .side-rule { height: 1px; background: #263244; margin: 1rem 0 1.25rem; }
    .side-label { color: #64748b; font-size: .67rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 750; margin-bottom: .55rem; }
    .side-status { color: #9fb2c9; font-size: .78rem; line-height: 1.55; }
    .side-status strong { color: #e5edf7; }

    .topline { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; margin-bottom:2.2rem; }
    .kicker { color: var(--accent); font-size: .72rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    .title { color: var(--ink); font-size: 2.55rem; font-weight: 780; letter-spacing: -.045em; line-height: 1.02; margin-top: .45rem; }
    .subtitle { color: var(--muted); font-size: .98rem; max-width: 720px; margin-top: .65rem; line-height: 1.55; }
    .system-meta { color: #64748b; font-size: .75rem; text-align:right; line-height:1.6; }
    .system-meta b { color:#334155; }

    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1.35rem 1.45rem; }
    .panel-tight { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.05rem 1.15rem; }
    .panel-title { color:#111827; font-size:.92rem; font-weight:750; margin-bottom:.15rem; }
    .panel-sub { color:#7b8797; font-size:.76rem; margin-bottom:1rem; }
    .eyebrow { color:#718096; font-size:.67rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }

    .score-wrap { border: 1px solid #dbe4ef; background: #fbfdff; border-radius: 14px; padding: 1.35rem 1.4rem; }
    .score-label { color:#64748b; font-size:.69rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
    .score { color:#0f172a; font-size:4.5rem; line-height:.95; font-weight:800; letter-spacing:-.06em; margin:.6rem 0 .3rem; }
    .score-model { color:#64748b; font-size:.78rem; }
    .score-scale { color:#94a3b8; font-size:.72rem; margin-top:.2rem; }
    .rule { height:1px; background:#e8edf3; margin:1.15rem 0; }

    .metric-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.05rem; min-height:91px; }
    .metric-label { color:#718096; font-size:.68rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
    .metric-value { color:#172033; font-size:1.35rem; font-weight:760; margin-top:.35rem; }
    .metric-note { color:#94a3b8; font-size:.7rem; margin-top:.15rem; }

    .section-head { display:flex; align-items:baseline; justify-content:space-between; margin:2.1rem 0 .8rem; }
    .section-head h2 { color:#172033; font-size:1.05rem; margin:0; font-weight:760; }
    .section-head span { color:#94a3b8; font-size:.72rem; }
    .status { display:inline-flex; align-items:center; gap:.45rem; color:#166534; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:999px; padding:.28rem .62rem; font-size:.69rem; font-weight:750; }
    .dot { width:6px; height:6px; border-radius:50%; background:#22c55e; display:inline-block; }

    textarea { border-radius: 10px !important; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding: .75rem .9rem; }
    div[data-testid="stMetricLabel"] p { font-size:.68rem; color:#718096; }
    div[data-testid="stMetricValue"] { font-size:1.25rem; }
    .stButton > button { border-radius:9px; font-weight:700; }
    .stTabs [data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { padding-left:.05rem; padding-right:.05rem; }
    .footer { color:#9aa5b5; font-size:.7rem; text-align:center; margin-top:3rem; }
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

# All four trained models are always part of the application. There is no
# user-facing model selector: the dashboard runs every available model and
# highlights the strongest one by stored QWK.

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">AES / Research Dashboard</div>'
        '<div class="brand-sub">Automatic Essay Scoring</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)
    if available_models:
        best_model = str(comparison.loc[comparison["qwk"].idxmax(), "model"])
        st.markdown(
            f'<div class="side-status"><strong>4 regression models</strong><br>'
            f'Ridge · LinearSVR · ElasticNet · SGDRegressor<br><br>'
            f'Current strongest QWK<br><strong>{best_model}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="side-status">No trained model results found.</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Data</div>', unsafe_allow_html=True)
    if SCRAPED_DATA.exists():
        try:
            n_scraped = len(pd.read_csv(SCRAPED_DATA))
            st.markdown(f'<span class="status"><span class="dot"></span> Scraped dataset loaded</span>', unsafe_allow_html=True)
            st.caption(f"{n_scraped:,} records · UOL Banco de Redações")
        except Exception:
            st.caption("Scraped dataset exists but could not be read.")
    else:
        st.caption("Scraped dataset not found locally.")

if not available_models:
    st.error("No trained model results were found. Run `src/compare_models.py` after preparing the scored dataset.")
    st.stop()

best_model = str(comparison.loc[comparison["qwk"].idxmax(), "model"])

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    f'''
    <div class="topline">
      <div>
        <div class="kicker">NLP · Automated Assessment</div>
        <div class="title">Automatic Essay Scoring</div>
        <div class="subtitle">A TF-IDF regression system for estimating essay scores and evaluating model agreement with human assessments.</div>
      </div>
      <div class="system-meta"><b>Production models</b><br>4 regressors · word + character TF-IDF<br>Best stored QWK: <b>{best_model}</b></div>
    </div>
    ''',
    unsafe_allow_html=True,
)

tab_score, tab_models, tab_data, tab_about = st.tabs(["Score an Essay", "Evaluation", "Dataset", "Methodology"])

with tab_score:
    left, right = st.columns([1.48, .82], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Essay input</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Paste the complete essay. All four trained regressors will be evaluated automatically.</div>', unsafe_allow_html=True)
        essay = st.text_area(
            "Essay text",
            height=445,
            label_visibility="collapsed",
            placeholder="Paste essay text here…",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    predictions: list[dict[str, object]] = []
    pred_df = pd.DataFrame()

    if essay.strip():
        for name in available_models:
            artifact = load_model(name)
            if artifact is None:
                continue
            raw = float(np.asarray(artifact["pipeline"].predict([essay])).ravel()[0])
            rounded = float(np.clip(np.rint(raw * 2) / 2, artifact["score_min"], artifact["score_max"]))
            predictions.append({"Model": name, "Raw score": raw, "Predicted score": rounded})
        pred_df = pd.DataFrame(predictions)

    with right:
        if not pred_df.empty:
            primary_row = pred_df[pred_df["Model"] == best_model]
            primary = float(primary_row["Predicted score"].iloc[0]) if not primary_row.empty else float(pred_df["Predicted score"].median())
            consensus = float(pred_df["Predicted score"].median())
            artifact = load_model(best_model)
            lo = float(artifact["score_min"]) if artifact else 0.0
            hi = float(artifact["score_max"]) if artifact else 10.0
            position = float(np.clip((primary - lo) / max(hi - lo, 1e-9), 0, 1))
            st.markdown(
                f'<div class="score-wrap"><div class="score-label">Primary estimate</div>'
                f'<div class="score">{primary:g}</div>'
                f'<div class="score-model">{best_model} · strongest stored QWK</div>'
                f'<div class="score-scale">Score range: {lo:g}–{hi:g}</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(position, text=f"Position on score scale · {lo:g} to {hi:g}")
            st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Cross-model estimate</div>', unsafe_allow_html=True)
            st.markdown(f"**Median prediction: {consensus:g}**")
            st.caption("All four regressors are run automatically; the primary estimate comes from the model with the strongest stored QWK.")
        else:
            st.markdown(
                '<div class="score-wrap"><div class="score-label">Primary estimate</div>'
                '<div class="score">—</div><div class="score-model">Waiting for essay input</div>'
                '<div class="score-scale">Paste an essay to generate predictions.</div></div>',
                unsafe_allow_html=True,
            )

    if essay.strip() and not pred_df.empty:
        stats = essay_stats(essay)
        st.markdown('<div class="section-head"><h2>Essay diagnostics</h2><span>Input-level statistics</span></div>', unsafe_allow_html=True)
        cols = st.columns(7)
        items = [
            ("Words", f"{int(stats['Words']):,}"),
            ("Characters", f"{int(stats['Characters']):,}"),
            ("Sentences", f"{int(stats['Sentences']):,}"),
            ("Paragraphs", f"{int(stats['Paragraphs']):,}"),
            ("Words / sentence", f"{stats['Avg. words / sentence']:.1f}"),
            ("Word length", f"{stats['Avg. word length']:.1f}"),
            ("Unique-word ratio", f"{stats['Unique-word ratio']:.2f}"),
        ]
        for col, (label, value) in zip(cols, items):
            col.metric(label, value)

        st.markdown('<div class="section-head"><h2>Model estimates</h2><span>All trained regressors</span></div>', unsafe_allow_html=True)
        display_df = pred_df.copy()
        display_df["Predicted score"] = display_df["Predicted score"].map(lambda x: f"{x:g}")
        display_df["Raw score"] = display_df["Raw score"].map(lambda x: f"{x:.3f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        chart_df = pred_df.set_index("Model")[["Predicted score"]]
        st.bar_chart(chart_df, use_container_width=True)
        st.info("Individual predictions are estimates. QWK, MAE and RMSE are evaluation metrics requiring reference scores and therefore apply to the test set, not a single essay.")
    elif essay.strip():
        st.warning("Essay received, but one or more trained model artifacts could not be loaded.")

with tab_models:
    st.markdown('<div class="section-head"><h2>Model evaluation</h2><span>Held-out test set</span></div>', unsafe_allow_html=True)
    best_qwk = comparison.loc[comparison["qwk"].idxmax()]
    best_mae = comparison.loc[comparison["mae"].idxmin()]
    best_rmse = comparison.loc[comparison["rmse"].idxmin()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best QWK", f"{best_qwk['qwk']:.3f}", str(best_qwk["model"]))
    c2.metric("Lowest MAE", f"{best_mae['mae']:.3f}", str(best_mae["model"]))
    c3.metric("Lowest RMSE", f"{best_rmse['rmse']:.3f}", str(best_rmse["model"]))

    st.markdown('<div class="section-head"><h2>Comparison</h2><span>Ridge · LinearSVR · ElasticNet · SGDRegressor</span></div>', unsafe_allow_html=True)
    table = comparison[["model", "qwk", "mae", "rmse", "n_test"]].rename(columns={"model":"Model", "qwk":"QWK", "mae":"MAE", "rmse":"RMSE", "n_test":"Test essays"})
    st.dataframe(table.round({"QWK":3, "MAE":3, "RMSE":3}), hide_index=True, use_container_width=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### QWK")
        st.bar_chart(comparison.set_index("model")[["qwk"]], use_container_width=True)
    with b:
        st.markdown("#### Error")
        st.bar_chart(comparison.set_index("model")[["mae", "rmse"]], use_container_width=True)

with tab_data:
    st.markdown('<div class="section-head"><h2>Dataset registry</h2><span>Local project data</span></div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    if SCRAPED_DATA.exists():
        try:
            scraped = pd.read_csv(SCRAPED_DATA)
            with a:
                st.markdown('<div class="panel-tight">', unsafe_allow_html=True)
                st.markdown('<div class="eyebrow">Web scraped</div>', unsafe_allow_html=True)
                st.markdown(f"### {len(scraped):,} essays")
                st.caption("UOL Banco de Redações · data/scraped/uol_essays.csv")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-head"><h2>Scraped data preview</h2><span>Source records</span></div>', unsafe_allow_html=True)
            cols = [c for c in ["essay", "score", "prompt", "source_url", "source_name"] if c in scraped.columns]
            st.dataframe(scraped[cols], hide_index=True, use_container_width=True)
        except Exception as exc:
            a.error(f"Could not read scraped dataset: {exc}")
    else:
        with a:
            st.warning("Scraped dataset not found.")

    if ASAP_DATA.exists():
        try:
            asap = pd.read_csv(ASAP_DATA)
            with b:
                st.markdown('<div class="panel-tight">', unsafe_allow_html=True)
                st.markdown('<div class="eyebrow">Reference dataset</div>', unsafe_allow_html=True)
                st.markdown(f"### {len(asap):,} essays")
                st.caption("ASAP-AES · data/processed/asap.csv")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as exc:
            b.error(f"Could not read ASAP dataset: {exc}")
    else:
        with b:
            st.warning("ASAP-AES dataset not found.")

with tab_about:
    st.markdown('<div class="section-head"><h2>Methodology</h2><span>Model architecture</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Scoring pipeline</div>', unsafe_allow_html=True)
    st.markdown("### Essay → TF-IDF → Regression → Score")
    st.write("The system transforms essay text into word-level and character-level TF-IDF features and feeds that representation into four regularized linear regression models.")
    st.markdown('</div>', unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Feature representation")
        st.write("Word n-grams capture lexical patterns and phrase structure. Character n-grams capture subword patterns, spelling variation and stylistic signals.")
        st.markdown("#### Models")
        st.write("Ridge, LinearSVR, ElasticNet and SGDRegressor are evaluated using the same underlying feature representation so their performance can be compared directly.")
    with b:
        st.markdown("#### Evaluation")
        st.write("Quadratic weighted kappa measures agreement with human scores. MAE and RMSE quantify absolute and squared prediction error respectively.")
        st.markdown("#### Data integrity")
        st.write("The training data must use a consistent scoring rubric. Historical web-scraped sources can contain different score scales and should be normalized before being combined.")

st.markdown('<div class="footer">Automatic Essay Scoring · NLP research dashboard</div>', unsafe_allow_html=True)
