"""Minimal Streamlit app: upload a brain CT (DICOM/JPG/PNG) → model prediction + Grad-CAM."""

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

st.set_page_config(page_title="Brain CT Stroke Classifier", page_icon="🧠", layout="wide")
st.title("🧠 Brain CT Stroke Classifier")
st.caption("Upload a brain CT slice (DICOM, JPG, or PNG) to get a prediction with Grad-CAM explainability.")


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


def _find_last_conv_layer(model) -> str:
    """Return the name of the last layer with a 4-D output (a conv feature map)."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("No 4-D conv layer found in model.")


@st.cache_resource(show_spinner=False)
def _gradcam_model(_model, layer_name: str):
    import tensorflow as tf

    return tf.keras.models.Model(
        inputs=_model.inputs,
        outputs=[_model.get_layer(layer_name).output, _model.output],
    )


def compute_gradcam(model, tensor: np.ndarray, class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap in [0,1] at the model's input resolution."""
    import tensorflow as tf

    layer_name = _find_last_conv_layer(model)
    grad_model = _gradcam_model(model, layer_name)

    x = tf.convert_to_tensor(np.expand_dims(tensor, 0), dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)              # (1, h, w, c)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))    # (c,)
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # (h, w)
    cam = tf.nn.relu(cam).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    return np.asarray(cam_img, dtype=np.float32) / 255.0


def overlay_heatmap(base_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-colored heatmap onto a grayscale base image."""
    import matplotlib.cm as cm

    base = np.clip(base_gray, 0, 1)
    base_rgb = np.stack([base] * 3, axis=-1)
    color = cm.get_cmap("jet")(heatmap)[..., :3]    # (H, W, 3) in [0,1]
    return np.clip(base_rgb * (1 - alpha) + color * alpha, 0, 1)


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

# ---------------------------------------------------------------------------
# Grad-CAM explainability
# ---------------------------------------------------------------------------
st.markdown("### Grad-CAM explainability")

target_choice = st.selectbox(
    "Explain which class?",
    options=list(range(len(CLASS_NAMES))),
    index=pred_idx,
    format_func=lambda i: f"{CLASS_NAMES[i]}  ({probs[i] * 100:.1f}%)",
)
alpha = st.slider("Heatmap intensity", 0.0, 1.0, 0.45, 0.05)

with st.spinner("Computing Grad-CAM…"):
    try:
        heatmap = compute_gradcam(model, tensor, int(target_choice))
        base = tensor[..., 0]
        overlay = overlay_heatmap(base, heatmap, alpha=alpha)
    except Exception as e:  # noqa: BLE001
        st.error(f"Grad-CAM failed: {e}")
        st.stop()

c1, c2, c3 = st.columns(3)
c1.image(base, caption="Input (brain window)", clamp=True, use_container_width=True)
c2.image(heatmap, caption="Activation heatmap", clamp=True, use_container_width=True)
c3.image(
    overlay,
    caption=f"Overlay → {CLASS_NAMES[int(target_choice)]}",
    clamp=True,
    use_container_width=True,
)

st.markdown(
    """
#### What the colored zones mean

Grad-CAM shows **which regions of the brain most influenced the model's score for the
selected class**. Colors are a "heat" scale — they do **not** indicate severity, only
the model's attention.

| Color | Activation | Meaning |
|---|---:|---|
| 🔴 **Red / yellow** | High (≈ 0.7 – 1.0) | Strong evidence — these pixels contributed the most to the predicted score. For *Bleeding* / *Ischemia*, this is roughly *where the model thinks the lesion is*. For *Normal*, this is the tissue the model used to rule stroke out. |
| 🟢 **Green / cyan** | Medium (≈ 0.3 – 0.7) | Moderate contribution — supporting context. Often surrounding tissue, midline, or symmetric counterparts the model is comparing against. |
| 🔵 **Blue / dark** | Low (≈ 0.0 – 0.3) | Little to no contribution — background, skull, or regions the model effectively ignored for this class. |

**How to read it**
- A red blob localized **inside brain tissue** is what you want for a stroke prediction.
⚠️ Grad-CAM is an explanation of the model, **not a diagnosis**. Do not use clinically.
"""
)
