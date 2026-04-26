"""Minimal Streamlit app: upload a brain CT (DICOM/JPG/PNG) → model prediction."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

IMG_SIZE = 224
WINDOWS = [(40, 80), (40, 380), (600, 2800)]   # brain, stroke/subdural, bone
CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]   # alphabetical → LabelEncoder order
MODEL_PATH = Path(__file__).resolve().parent / "stroke_transfer_Efficient_model.h5"

st.set_page_config(page_title="Brain CT Stroke Classifier", page_icon="🧠")
st.title("🧠 Brain CT Stroke Classifier")
st.caption("Upload a brain CT slice (DICOM, JPG, or PNG) to get a prediction.")


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={
            "eff_preprocess": eff_preprocess,
            "preprocess_input": eff_preprocess,
        },
        compile=False,
        safe_mode=False,
    )


def _resize_unit(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
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
    img = Image.open(io.BytesIO(buf)).convert("L").resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], axis=-1)


uploaded = st.file_uploader(
    "Upload scan", type=["dcm", "jpg", "jpeg", "png"], accept_multiple_files=False
)

if uploaded is None:
    st.info("⬆️ Choose a file to begin.")
    st.stop()

try:
    buf = uploaded.read()
    ext = Path(uploaded.name).suffix.lower().lstrip(".")
    tensor = preprocess_dicom(buf) if ext == "dcm" else preprocess_image(buf)
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to read file: {e}")
    st.stop()

st.image(tensor[..., 0], caption="Input (brain window)", clamp=True, use_container_width=True)

try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Model load failed: {e}")
    st.stop()

with st.spinner("Predicting…"):
    probs = model.predict(np.expand_dims(tensor, 0), verbose=0)[0]

pred_idx = int(np.argmax(probs))
st.subheader(f"Prediction: **{CLASS_NAMES[pred_idx]}**  ({probs[pred_idx]*100:.1f}%)")

st.bar_chart({"probability": dict(zip(CLASS_NAMES, [float(p) for p in probs]))})
