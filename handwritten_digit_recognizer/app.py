from pathlib import Path
import json

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "mnist_cnn.keras"
METRICS_PATH = APP_DIR / "models" / "training_metrics.json"

st.set_page_config(
    page_title="DigitIQ | Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1440px;}
.hero {padding: 1.35rem 1.55rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 18px; background: linear-gradient(115deg, rgba(59,130,246,.15), rgba(139,92,246,.10));}
.hero-kicker {font-size: .76rem; letter-spacing: .13em; font-weight: 700; opacity: .7;}
.subtle {color: #8b929c; font-size: .9rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 13px; border-radius: 13px;}
.result-digit {font-size: 5rem; line-height: 1; font-weight: 800; text-align: center; padding: .3rem;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: str, modified_time: float):
    return tf.keras.models.load_model(model_path)

def model_available():
    return MODEL_PATH.exists()

def prepare_image(image: Image.Image) -> np.ndarray:
    """Convert a drawing/photo to MNIST-like 28x28 white-background, black-ink input."""
    img = ImageOps.exif_transpose(image).convert("L")
    arr = np.asarray(img).astype("float32")

    # MNIST uses light digits on a dark background; infer polarity from border pixels.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:
        arr = 255.0 - arr

    # Remove low-intensity background noise, then crop to the digit.
    arr[arr < 24] = 0
    ys, xs = np.where(arr > 25)
    if len(xs) == 0:
        raise ValueError("No visible digit found. Draw or upload a darker digit.")

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = Image.fromarray(np.uint8(np.clip(arr[y0:y1, x0:x1], 0, 255)))

    # Fit the digit inside 20x20 while preserving aspect ratio, then center on 28x28.
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    final = np.asarray(canvas).astype("float32") / 255.0
    return final.reshape(1, 28, 28, 1)

def show_prediction(model, image, label):
    try:
        prepared = prepare_image(image)
        probs = model.predict(prepared, verbose=0)[0]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])

        st.markdown(f"<div class='subtle'>{label}</div>", unsafe_allow_html=True)
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            st.image((prepared[0, :, :, 0] * 255).astype("uint8"),
                     caption="Processed 28 × 28 model input", width=230,
                     clamp=True)
        with right:
            st.markdown(f"<div class='result-digit'>{predicted}</div>", unsafe_allow_html=True)
            st.markdown("**Predicted digit**")
            st.metric("Model confidence", f"{confidence:.2%}")
            if confidence < 0.70:
                st.warning("Low confidence. Try a clearer, centered digit with a single stroke path.")
            else:
                st.success("The model is confident about this prediction.")

        prob_df = pd.DataFrame({"Digit": list(range(10)), "Probability": probs})
        fig = px.bar(prob_df, x="Digit", y="Probability", range_y=[0, 1],
                     text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
                     title="Probability across all 10 digits")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=330, margin=dict(t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not recognize this image: {exc}")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">COMPUTER VISION • DEEP LEARNING • MNIST</div>
  <h1 style="margin:.3rem 0;">🔢 DigitIQ — Handwritten Digit Recognizer</h1>
  <div>Draw or upload a handwritten digit. A Convolutional Neural Network analyzes the image and predicts a digit from 0 to 9.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Workspace")
    st.markdown("**Model status**")
    if model_available():
        st.success("CNN model found")
    else:
        st.error("CNN model not trained yet")
    st.caption("Train the model once using the setup instructions in README.md.")
    st.divider()
    st.markdown("**Input tips**")
    st.caption("Draw one large digit, centered in the canvas. Avoid writing multiple digits or adding circles and arrows.")

if not model_available():
    st.warning("The trained CNN model is not available yet.")
    st.markdown("""
    Open the VS Code terminal in this project folder and run:

    ```powershell
    py -m pip install -r requirements.txt
    py train_model.py
    ```

    Training downloads MNIST, trains the CNN and saves `models/mnist_cnn.keras`.
    Then restart or refresh this Streamlit app.
    """)
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

metrics = {}
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

m1, m2, m3 = st.columns(3)
m1.metric("Architecture", "CNN")
m2.metric("Dataset", "MNIST")
m3.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.2%}" if metrics else "Train model")

tab_draw, tab_upload, tab_about = st.tabs(["✍️ Draw a digit", "🖼️ Upload an image", "📊 Model & learning"])

with tab_draw:
    st.subheader("Write a digit on the canvas")
    st.caption("Use your mouse or touchscreen. Draw one digit from 0 to 9.")
    col_canvas, col_action = st.columns([1.15, 1], gap="large")
    with col_canvas:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=320,
            width=320,
            drawing_mode="freedraw",
            key="digit_canvas",
            display_toolbar=True,
            update_streamlit=True,
        )
    with col_action:
        st.markdown("### Recognition")
        st.markdown("The model converts your drawing into a 28 × 28 grayscale image before prediction.")
        if st.button("Recognize digit", type="primary", use_container_width=True):
            if canvas_result.image_data is None:
                st.warning("Draw a digit first.")
            else:
                rgba = np.asarray(canvas_result.image_data).astype("uint8")
                # Canvas background is black; strokes are white. Convert to PIL for preprocessing.
                drawing = Image.fromarray(rgba, mode="RGBA").convert("RGB").convert("L")
                show_prediction(model, drawing, "Live canvas result")

with tab_upload:
    st.subheader("Upload a handwritten digit")
    st.caption("Supported formats: PNG, JPG and JPEG. Upload one digit per image.")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", width=260)
            if st.button("Predict uploaded digit", type="primary"):
                show_prediction(model, image, "Uploaded image result")
        except Exception as exc:
            st.error(f"Unable to open image: {exc}")

with tab_about:
    st.subheader("Model overview")
    a, b, c = st.columns(3)
    a.metric("Input shape", "28 × 28")
    b.metric("Output classes", "10")
    c.metric("Dataset", "70,000 images")
    st.markdown("""
    **CNN architecture**
    1. Convolution layer — detects basic image patterns and edges.
    2. Max pooling — reduces spatial dimensions.
    3. Second convolution block — learns more complex digit patterns.
    4. Flatten + Dense — converts learned features into class scores.
    5. Softmax — returns probabilities for digits 0 through 9.
    """)
    if metrics:
        st.markdown("**Training results**")
        t1, t2, t3 = st.columns(3)
        t1.metric("Test loss", f"{metrics.get('test_loss', 0):.4f}")
        t2.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
        t3.metric("Training epochs", str(metrics.get("epochs", "—")))
        history_path = APP_DIR / "models" / "training_history.csv"
        if history_path.exists():
            hist = pd.read_csv(history_path)
            if {"epoch", "accuracy", "val_accuracy"}.issubset(hist.columns):
                long_hist = hist.melt(id_vars="epoch",
                                      value_vars=["accuracy", "val_accuracy"],
                                      var_name="Metric", value_name="Accuracy")
                fig = px.line(long_hist, x="epoch", y="Accuracy", color="Metric",
                              markers=True, title="Training vs validation accuracy")
                st.plotly_chart(fig, use_container_width=True)
    st.info("MNIST is a benchmark dataset of handwritten digits. Strong test accuracy on MNIST does not guarantee equally strong results on phone photos, unusual handwriting, or other writing styles.")

st.markdown("---")
st.markdown("<div class='subtle'>DigitIQ • Educational computer-vision prototype • Predictions are model estimates.</div>",
            unsafe_allow_html=True)
