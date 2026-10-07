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

st.set_page_config(page_title="AES | Automatic Essay Scoring", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{--navy:#172033;--blue:#315efb;--text:#172033;--muted:#6b7280;--border:#e5e7eb;--surface:#fff;--bg:#f5f6f8}
.stApp{background:var(--bg);color:var(--text)}
.block-container{max-width:1440px;padding:1.5rem 2.25rem 3rem}
[data-testid="stSidebar"]{background:var(--navy);border-right:none}
[data-testid="stSidebar"] *{color:#dbe3ef}
[data-testid="stSidebar"] .stCaption{color:#9aa9bd}
.side-brand{padding:.2rem 0 1.25rem}.side-brand-title{color:#fff;font-size:1.05rem;font-weight:800;letter-spacing:-.02em}.side-brand-sub{color:#8fa0b8;font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;margin-top:.3rem}
.side-heading{color:#8292aa;font-size:.65rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin:1.35rem 0 .55rem}.side-line{height:1px;background:#2b3850;margin:.2rem 0 1rem}.side-info{font-size:.78rem;line-height:1.6;color:#aebbd0}.side-info strong{color:#f2f5f9}
.app-header{display:flex;justify-content:space-between;align-items:center;padding:.15rem 0 1.2rem;border-bottom:1px solid var(--border);margin-bottom:1.35rem}.app-kicker{color:var(--blue);font-size:.67rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.app-title{color:var(--text);font-size:1.65rem;font-weight:780;letter-spacing:-.035em;margin-top:.18rem}.app-desc{color:var(--muted);font-size:.82rem;margin-top:.22rem}.header-meta{text-align:right;color:#7b8492;font-size:.72rem;line-height:1.55}.header-meta strong{color:#394354}
.workspace-label{color:#657084;font-size:.67rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.5rem}
.input-shell{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.05rem .75rem}.input-title{font-size:.9rem;font-weight:750;color:var(--text)}.input-help{font-size:.72rem;color:#8992a0;margin:.15rem 0 .65rem}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.15rem 1.2rem}.result-card.primary{border-top:3px solid var(--blue)}.result-label{color:#687386;font-size:.66rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}.result-score{color:var(--text);font-size:4rem;font-weight:800;line-height:.95;letter-spacing:-.06em;margin:.55rem 0 .35rem}.result-model{color:#697487;font-size:.73rem}.result-note{color:#9aa2ae;font-size:.68rem;margin-top:.15rem}
.section-title{color:var(--text);font-size:.98rem;font-weight:780;margin:1.45rem 0 .65rem}.section-caption{color:#8b94a2;float:right;font-size:.7rem;font-weight:500}
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:.7rem .8rem}div[data-testid="stMetricLabel"] p{color:#778191;font-size:.64rem;font-weight:700}div[data-testid="stMetricValue"]{color:var(--text);font-size:1.12rem}
.stButton>button{border-radius:7px;font-weight:750;min-height:2.45rem}.stTextArea textarea{border-radius:7px!important;border-color:#dfe3e9!important;font-size:.88rem!important;line-height:1.55!important}.stTextArea textarea:focus{border-color:#9db2ff!important;box-shadow:0 0 0 1px #9db2ff!important}
.stTabs [data-baseweb="tab-list"]{gap:1.3rem;border-bottom:1px solid var(--border)}.stTabs [data-baseweb="tab"]{font-size:.78rem;font-weight:650;padding:.55rem .1rem}.stDataFrame{border:1px solid var(--border);border-radius:8px;overflow:hidden}.footer{color:#a0a7b2;font-size:.67rem;text-align:center;border-top:1px solid var(--border);padding-top:1rem;margin-top:2.5rem}
</style>
""", unsafe_allow_html=True)


def essay_stats(text: str) -> dict[str, float]:
    words = WORD_RE.findall(text)
    sentences = [s for s in SENTENCE_RE.split(text) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    wc, sc = len(words), len(sentences)
    return {"Words":float(wc),"Characters":float(len(text)),"Sentences":float(sc),"Paragraphs":float(len(paragraphs)),
            "Avg. words / sentence":float(wc/max(sc,1)),"Avg. word length":float(sum(map(len,words))/max(wc,1)),
            "Unique-word ratio":float(len({w.lower() for w in words})/max(wc,1))}


@st.cache_resource
def load_model(name: str):
    path = MODELS / f"scraped_{name.lower()}.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    path = RESULTS / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_dataset(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    return pd.read_csv(path) if path.exists() else None


comparison = load_comparison()
available_models = comparison["model"].tolist() if comparison is not None and not comparison.empty else []

if not available_models:
    st.error("No trained model results were found. Run \x60src/compare_models.py\x60 after preparing the scored dataset.")
    st.stop()

best_model = str(comparison.loc[comparison["qwk"].idxmax(), "model"])

with st.sidebar:
    st.markdown('<div class="side-brand"><div class="side-brand-title">AES</div><div class="side-brand-sub">Automatic Essay Scoring</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-heading">Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-info"><strong>Essay scoring workspace</strong><br>Evaluate an essay with every trained regression model and inspect the stored benchmark results.</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-heading">Models</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="side-info"><strong>{len(available_models)} regressors</strong><br>Ridge · LinearSVR · ElasticNet · SGDRegressor<br><br>Best stored QWK<br><strong>{best_model}</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-heading">Data</div>', unsafe_allow_html=True)
    scraped = load_dataset(str(SCRAPED_DATA))
    asap = load_dataset(str(ASAP_DATA))
    if scraped is not None:
        st.markdown(f'<div class="side-info"><strong>{len(scraped):,}</strong> scraped records<br>UOL Banco de Redações</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-info">Scraped corpus not loaded.</div>', unsafe_allow_html=True)
    if asap is not None:
        st.caption(f"ASAP-AES reference: {len(asap):,} essays")

st.markdown(f"""
<div class="app-header">
<div><div class="app-kicker">NLP / Automated Assessment</div><div class="app-title">Automatic Essay Scoring</div><div class="app-desc">Score essays, compare model predictions, and inspect evaluation results.</div></div>
<div class="header-meta"><strong>TF-IDF regression</strong><br>Word + character features<br>{len(available_models)} trained models</div>
</div>
""", unsafe_allow_html=True)

tab_score, tab_eval, tab_data, tab_method = st.tabs(["Score","Model evaluation","Datasets","Methodology"])

with tab_score:
    st.markdown('<div class="workspace-label">Scoring workspace</div>', unsafe_allow_html=True)
    left, right = st.columns([1.7,1.0], gap="large")
    with left:
        st.markdown('<div class="input-shell"><div class="input-title">Essay</div><div class="input-help">Paste the complete essay. All trained models are evaluated automatically.</div></div>', unsafe_allow_html=True)
        essay = st.text_area("Essay text", height=390, label_visibility="collapsed", placeholder="Paste essay text here...", key="essay_input")
        score_clicked = st.button("Score essay", type="primary", use_container_width=True)

    if score_clicked and not essay.strip():
        st.warning("Paste an essay before scoring.")
        st.session_state["last_predictions"] = None
    elif score_clicked:
        predictions = []
        for name in available_models:
            artifact = load_model(name)
            if artifact is None:
                continue
            raw = float(np.asarray(artifact["pipeline"].predict([essay])).ravel()[0])
            rounded = float(np.clip(np.rint(raw*2)/2, artifact["score_min"], artifact["score_max"]))
            predictions.append({"Model":name,"Raw score":raw,"Predicted score":rounded})
        st.session_state["last_predictions"] = predictions
        st.session_state["last_essay"] = essay

    pred_df = pd.DataFrame(st.session_state.get("last_predictions") or [])
    scored_essay = st.session_state.get("last_essay", essay)

    with right:
        if not pred_df.empty:
            primary_row = pred_df[pred_df["Model"] == best_model]
            primary = float(primary_row["Predicted score"].iloc[0]) if not primary_row.empty else float(pred_df["Predicted score"].median())
            consensus = float(pred_df["Predicted score"].median())
            artifact = load_model(best_model)
            lo = float(artifact["score_min"]) if artifact else 0.0
            hi = float(artifact["score_max"]) if artifact else 10.0
            position = float(np.clip((primary-lo)/max(hi-lo,1e-9),0,1))
            st.markdown(f'<div class="result-card primary"><div class="result-label">Estimated score</div><div class="result-score">{primary:g}</div><div class="result-model">{best_model} · highest stored QWK</div><div class="result-note">Model score range: {lo:g}–{hi:g}</div></div>', unsafe_allow_html=True)
            st.progress(position, text=f"Score position · {lo:g}–{hi:g}")
            c1,c2=st.columns(2)
            c1.metric("Model consensus",f"{consensus:g}")
            c2.metric("Models evaluated",len(pred_df))
        else:
            st.markdown('<div class="result-card primary"><div class="result-label">Estimated score</div><div class="result-score">—</div><div class="result-model">Ready when you are</div><div class="result-note">Enter an essay and select “Score essay”.</div></div>', unsafe_allow_html=True)

    if not pred_df.empty:
        stats = essay_stats(scored_essay)
        st.markdown('<div class="section-title">Essay diagnostics <span class="section-caption">Input-level statistics</span></div>', unsafe_allow_html=True)
        cols=st.columns(7)
        items=[("Words",f"{int(stats['Words']):,}"),("Characters",f"{int(stats['Characters']):,}"),("Sentences",f"{int(stats['Sentences']):,}"),("Paragraphs",f"{int(stats['Paragraphs']):,}"),("Words / sentence",f"{stats['Avg. words / sentence']:.1f}"),("Avg. word length",f"{stats['Avg. word length']:.1f}"),("Unique-word ratio",f"{stats['Unique-word ratio']:.2f}")]
        for col,(label,value) in zip(cols,items): col.metric(label,value)

        st.markdown('<div class="section-title">Model predictions <span class="section-caption">Every trained regressor</span></div>', unsafe_allow_html=True)
        display_df=pred_df.copy()
        display_df["Predicted score"]=display_df["Predicted score"].map(lambda x:f"{x:g}")
        display_df["Raw score"]=display_df["Raw score"].map(lambda x:f"{x:.3f}")
        st.dataframe(display_df,hide_index=True,use_container_width=True)
        st.markdown('<div class="section-title">Prediction spread <span class="section-caption">Rounded model estimates</span></div>', unsafe_allow_html=True)
        st.bar_chart(pred_df.set_index("Model")[["Predicted score"]],use_container_width=True)
        st.caption("The primary estimate uses the model with the strongest stored QWK. QWK, MAE and RMSE are test-set metrics and do not measure a single essay directly.")
    else:
        st.info("The score workspace is ready. Predictions will appear here after you score an essay.")

with tab_eval:
    st.markdown('<div class="workspace-label">Benchmark results</div>', unsafe_allow_html=True)
    best_qwk=comparison.loc[comparison["qwk"].idxmax()]
    best_mae=comparison.loc[comparison["mae"].idxmin()]
    best_rmse=comparison.loc[comparison["rmse"].idxmin()]
    c1,c2,c3=st.columns(3)
    c1.metric("Best QWK",f"{best_qwk['qwk']:.3f}",str(best_qwk["model"]))
    c2.metric("Lowest MAE",f"{best_mae['mae']:.3f}",str(best_mae["model"]))
    c3.metric("Lowest RMSE",f"{best_rmse['rmse']:.3f}",str(best_rmse["model"]))
    st.markdown("### Model comparison")
    table=comparison[["model","qwk","mae","rmse","n_test"]].rename(columns={"model":"Model","qwk":"QWK","mae":"MAE","rmse":"RMSE","n_test":"Test essays"})
    st.dataframe(table.round({"QWK":3,"MAE":3,"RMSE":3}),hide_index=True,use_container_width=True)
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("#### Quadratic weighted kappa")
        st.bar_chart(comparison.set_index("model")[["qwk"]],use_container_width=True)
    with b:
        st.markdown("#### Prediction error")
        st.bar_chart(comparison.set_index("model")[["mae","rmse"]],use_container_width=True)

with tab_data:
    st.markdown('<div class="workspace-label">Dataset registry</div>', unsafe_allow_html=True)
    a,b=st.columns(2,gap="large")
    with a:
        if scraped is not None:
            st.markdown("### Web-scraped corpus")
            st.metric("Essays",f"{len(scraped):,}")
            st.caption("UOL Banco de Redações · data/scraped/uol_essays.csv")
        else: st.warning("Web-scraped dataset not found.")
    with b:
        if asap is not None:
            st.markdown("### ASAP-AES reference")
            st.metric("Essays",f"{len(asap):,}")
            st.caption("Reference corpus · data/processed/asap.csv")
        else: st.warning("ASAP-AES dataset not found.")
    if scraped is not None:
        st.markdown("### Scraped records")
        cols=[c for c in ["essay","score","prompt","source_url","source_name"] if c in scraped.columns]
        st.dataframe(scraped[cols],hide_index=True,use_container_width=True)
        st.warning("Scraped sources can use different historical scoring rubrics. Do not combine them for training until their score scales have been verified and normalized.")

with tab_method:
    st.markdown('<div class="workspace-label">System design</div>', unsafe_allow_html=True)
    st.markdown("### Essay → TF-IDF → Regression → Score")
    st.write("The application converts essay text into word-level and character-level TF-IDF features and evaluates four regression models using the same representation.")
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown("#### Feature representation")
        st.write("Word n-grams capture lexical and phrase-level patterns. Character n-grams capture subword structure, spelling variation and stylistic signals.")
        st.markdown("#### Models")
        st.write("Ridge, LinearSVR, ElasticNet and SGDRegressor provide four comparable linear baselines. The application runs all available trained models automatically.")
    with b:
        st.markdown("#### Evaluation")
        st.write("Quadratic weighted kappa measures agreement with human scores, while MAE and RMSE quantify prediction error.")
        st.markdown("#### Data integrity")
        st.write("Training data must use a consistent scoring rubric. Web-scraped historical sources require scale verification before they are merged with other corpora.")

st.markdown('<div class="footer">AES · Automatic Essay Scoring · NLP research dashboard</div>', unsafe_allow_html=True)
