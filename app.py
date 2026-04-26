"""Professional Streamlit app: upload a brain CT (DICOM/JPG/PNG) → model prediction."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

IMG_SIZE = 224
WINDOWS = [(40, 80), (40, 380), (600, 2800)]
CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]
MODEL_PATH = Path(__file__).resolve().parent / "stroke_transfer_Efficient_model.h5"

st.set_page_config(
    page_title="NeuroScan AI — Brain CT Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #080C14;
    --surface:   #0E1520;
    --surface2:  #141C2E;
    --border:    #1E2D45;
    --accent:    #00C2FF;
    --accent2:   #7B5EA7;
    --danger:    #FF4E6A;
    --warn:      #F5A623;
    --ok:        #00D68F;
    --text:      #E8EDF5;
    --muted:     #5A6A85;
    --mono:      'DM Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --display:   'Syne', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,194,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(123,94,167,0.06) 0%, transparent 60%),
        var(--bg) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

.block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    margin-bottom: 0.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,194,255,0.08);
    border: 1px solid rgba(0,194,255,0.2);
    border-radius: 100px;
    padding: 5px 14px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin-bottom: 1.4rem;
    text-transform: uppercase;
}
.hero-badge span { font-size: 8px; }
.hero h1 {
    font-family: var(--display);
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 0.8rem;
}
.hero h1 em {
    font-style: normal;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    font-family: var(--sans);
    font-size: 1rem;
    font-weight: 300;
    color: var(--muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Upload zone ── */
.upload-zone {
    background: var(--surface);
    border: 1.5px dashed var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
    margin-bottom: 1rem;
}
.upload-zone:hover { border-color: var(--accent); }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
}
.card-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.card-value {
    font-family: var(--display);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
}
.card-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
    font-family: var(--mono);
}

/* ── Result banner ── */
.result-banner {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: 1.5rem 0;
    border: 1px solid;
}
.result-banner.bleeding  { background: rgba(255,78,106,0.08); border-color: rgba(255,78,106,0.3); }
.result-banner.ischemia  { background: rgba(245,166,35,0.08); border-color: rgba(245,166,35,0.3); }
.result-banner.normal    { background: rgba(0,214,143,0.08);  border-color: rgba(0,214,143,0.3); }
.result-icon { font-size: 2.4rem; flex-shrink: 0; }
.result-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.result-label.bleeding { color: var(--danger); }
.result-label.ischemia { color: var(--warn); }
.result-label.normal   { color: var(--ok); }
.result-name {
    font-family: var(--display);
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
}
.result-conf {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--muted);
    margin-top: 6px;
}

/* ── Probability bars ── */
.prob-row { margin-bottom: 1rem; }
.prob-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    font-size: 13px;
}
.prob-name { font-weight: 500; color: var(--text); }
.prob-val  { font-family: var(--mono); font-size: 12px; color: var(--muted); }
.prob-track {
    height: 6px;
    background: var(--surface2);
    border-radius: 100px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.fill-bleeding { background: linear-gradient(90deg, #FF4E6A, #FF8FA3); }
.fill-ischemia { background: linear-gradient(90deg, #F5A623, #FFCE6B); }
.fill-normal   { background: linear-gradient(90deg, #00D68F, #5AFFC5); }

/* ── Section label ── */
.section-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Image panel ── */
.img-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}
.img-card-header {
    padding: 0.9rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-r { background: #FF4E6A; }
.dot-y { background: #F5A623; }
.dot-g { background: #00D68F; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(245,166,35,0.06);
    border: 1px solid rgba(245,166,35,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
    margin-top: 2rem;
}
.disclaimer strong { color: var(--warn); }

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
[data-testid="stImage"] img {
    border-radius: 0 0 14px 14px !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMetric"] { display: none; }

div[data-testid="stStatusWidget"] { display: none; }
footer { display: none !important; }
#MainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge"><span>●</span> AI-Powered Analysis</div>
  <h1>Neuro<em>Scan</em> AI</h1>
  <p>Upload a brain CT slice for instant classification of bleeding, ischemia, or normal findings.</p>
</div>
""", unsafe_allow_html=True)


# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload Scan</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a DICOM, JPG, or PNG file here",
    type=["dcm", "jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0; color: #5A6A85; font-size:13px; font-family:'DM Mono',monospace;">
        Accepted formats: .dcm &nbsp;·&nbsp; .jpg &nbsp;·&nbsp; .png
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Preprocessing ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising model…")
def load_model():
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"eff_preprocess": eff_preprocess, "preprocess_input": eff_preprocess},
        compile=False,
        safe_mode=False,
    )


def _resize_unit(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def preprocess_dicom(buf: bytes) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(buf))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    chs = []
    for c, w in WINDOWS:
        lo, hi = c - w / 2, c + w / 2
        chs.append(_resize_unit(np.clip((hu - lo) / (hi - lo), 0.0, 1.0)))
    return np.stack(chs, axis=-1)


def preprocess_image(buf: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(buf)).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], axis=-1)


try:
    buf = uploaded.read()
    ext = Path(uploaded.name).suffix.lower().lstrip(".")
    tensor = preprocess_dicom(buf) if ext == "dcm" else preprocess_image(buf)
except Exception as e:
    st.error(f"**Failed to read file:** {e}")
    st.stop()


# ── Layout: image + results ───────────────────────────────────────────────────
col_img, col_res = st.columns([1, 1.2], gap="large")

with col_img:
    st.markdown('<div class="section-label">Input Scan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="img-card">
      <div class="img-card-header">
        <span><span class="dot dot-r"></span>&nbsp;<span class="dot dot-y"></span>&nbsp;<span class="dot dot-g"></span></span>
        <span>Brain Window · 224×224</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.image(tensor[..., 0], use_container_width=True)
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-top:0;border-radius:0 0 16px 16px;
                padding:0.7rem 1.2rem;display:flex;justify-content:space-between;">
      <span style="font-family:'DM Mono',monospace;font-size:11px;color:#5A6A85;">FILE</span>
      <span style="font-family:'DM Mono',monospace;font-size:11px;color:#5A6A85;">{uploaded.name}</span>
    </div>
    """, unsafe_allow_html=True)

with col_res:
    st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)

    try:
        model = load_model()
    except Exception as e:
        st.error(f"**Model load failed:** {e}")
        st.stop()

    with st.spinner("Running inference…"):
        probs = model.predict(np.expand_dims(tensor, 0), verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx]) * 100

    css_class = pred_name.lower()
    icons = {"Bleeding": "🔴", "Ischemia": "🟡", "Normal": "🟢"}
    icon = icons[pred_name]

    st.markdown(f"""
    <div class="result-banner {css_class}">
      <div class="result-icon">{icon}</div>
      <div>
        <div class="result-label {css_class}">Primary Finding</div>
        <div class="result-name">{pred_name}</div>
        <div class="result-conf">Confidence: {pred_conf:.1f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.4rem;">Class Probabilities</div>', unsafe_allow_html=True)

    colors = {"Bleeding": "fill-bleeding", "Ischemia": "fill-ischemia", "Normal": "fill-normal"}
    for name, prob in zip(CLASS_NAMES, probs):
        pct = float(prob) * 100
        fill_class = colors[name]
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-header">
            <span class="prob-name">{name}</span>
            <span class="prob-val">{pct:.1f}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill {fill_class}" style="width:{pct:.1f}%"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Stats row
    st.markdown('<div class="section-label" style="margin-top:1.4rem;">Scan Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    file_kb = len(buf) / 1024
    with c1:
        st.markdown(f"""
        <div class="card">
          <div class="card-label">File Size</div>
          <div class="card-value">{file_kb:.0f}<span style="font-size:1rem;font-weight:400"> KB</span></div>
          <div class="card-sub">{ext.upper()} format</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card">
          <div class="card-label">Resolution</div>
          <div class="card-value">224<span style="font-size:1rem;font-weight:400">px</span></div>
          <div class="card-sub">Normalised input</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card">
          <div class="card-label">Windows</div>
          <div class="card-value">3<span style="font-size:1rem;font-weight:400"> ch</span></div>
          <div class="card-sub">Brain · Stroke · Bone</div>
        </div>""", unsafe_allow_html=True)


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  <strong>⚠ Clinical Disclaimer:</strong> This tool is intended for research and educational purposes only.
  It is not a certified medical device and must not be used as a substitute for professional clinical judgment.
  Always consult a qualified radiologist or physician for diagnosis and treatment decisions.
</div>
""", unsafe_allow_html=True)