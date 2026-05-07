import os

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from model import CLASS_NAMES, IMAGE_SIZE

st.set_page_config(
    page_title="Multiclass Image Classifier",
    page_icon="🧠",
    layout="centered",
)


st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        color: #534ab7;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #5f5e5a;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        background: #eeedfe;
        border: 1.5px solid #afa9ec;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #888780;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .prediction-class {
        font-size: 2rem;
        font-weight: 700;
        color: #534ab7;
    }
    .confidence-text {
        font-size: 1rem;
        color: #3c3489;
        font-weight: 500;
    }
    .top3-item {
        background: #f5f5f0;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
    }
    .winner-item {
        background: #eeedfe;
        border: 0.5px solid #afa9ec;
    }
    .info-box {
        background: #e1f5ee;
        border: 0.5px solid #5dcaa5;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        font-size: 0.88rem;
        color: #085041;
        margin-top: 1rem;
    }
    .error-box {
        background: #fcebeb;
        border: 0.5px solid #f09595;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        color: #791f1f;
        font-size: 0.9rem;
    }
    div[data-testid="stProgress"] > div {
        background-color: #534ab7 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join("saved_model", "cnn_rnn_cifar10.keras")
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)   # shape (1, 32, 32, 3)


def conf_color(pct: float) -> str:
    if pct >= 70: return "🟢"
    if pct >= 40: return "🟡"
    return "🔴"


st.markdown('<p class="main-title">🧠 Multiclass Image Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload an image to classify it into one of the '
    '<b>10 CIFAR-10 classes</b>: '
    'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.</p>',
    unsafe_allow_html=True,
)


model = load_model()

if model is None:
    st.markdown("""
    <div class="error-box">
        ⚠️  <b>Model not found.</b>  Please run <code>python train.py</code> first to train
        and save the model, then restart this app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.sidebar:
    st.header("📊 Model Info")
    st.write("**Architecture:** CNN + LSTM")
    st.write("**Dataset:** CIFAR-10")
    st.write("**Classes:** 10")
    st.write("**Input size:** 32 × 32 × 3")

    total_params = model.count_params()
    st.write(f"**Parameters:** {total_params:,}")

    st.markdown("---")
    st.header("📂 Classes")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"`{i}` {name.capitalize()}")

    st.markdown("---")
    st.caption("CNN extracts spatial features → LSTM classifies the sequence → Softmax outputs probabilities.")

st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader(
    label="Choose an image file",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="Max file size: 200 MB. Supports PNG, JPG, WEBP, BMP.",
)

if uploaded_file is not None:

    pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Uploaded Image**")
        st.image(pil_img, use_column_width=True)
        st.caption(f"Size: {pil_img.width} × {pil_img.height} px  |  Mode: {pil_img.mode}")

    with col2:
        with st.spinner("Classifying …"):
            try:
                inp   = preprocess_image(pil_img)
                probs = model.predict(inp, verbose=0)[0]         # shape (10,)

                top_idx    = int(np.argmax(probs))
                confidence = float(probs[top_idx]) * 100
                top3_idx   = np.argsort(probs)[::-1][:3]

                st.markdown(f"""
                <div class="prediction-box">
                    <p class="prediction-label">Predicted Class</p>
                    <p class="prediction-class">{CLASS_NAMES[top_idx].upper()}</p>
                    <p class="confidence-text">{conf_color(confidence)} Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(confidence))

                st.markdown("**Top 3 Predictions**")
                for rank, idx in enumerate(top3_idx):
                    prob_pct = float(probs[idx]) * 100
                    css_class = "top3-item winner-item" if rank == 0 else "top3-item"
                    medal = ["🥇", "🥈", "🥉"][rank]
                    st.markdown(f"""
                    <div class="{css_class}">
                        <span>{medal} <b>{CLASS_NAMES[idx].capitalize()}</b></span>
                        <span>{prob_pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(prob_pct))

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Prediction failed: {e}</div>',
                            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Full Class Probabilities")

    prob_dict = {CLASS_NAMES[i].capitalize(): float(probs[i]) * 100 for i in range(10)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

    st.bar_chart(sorted_probs, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        ℹ️ <b>Note:</b> This model is trained on CIFAR-10 (32×32 images).
        Photos of real-world objects may give lower confidence scores because the model
        was trained on small, centred images. For best results, upload clear images of
        a single object on a plain background.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Upload an image above to get a prediction.")

    st.markdown("---")
    st.markdown("### 🏷️ What can this model classify?")
    cols = st.columns(5)
    icons = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚛"]
    for i, (name, icon) in enumerate(zip(CLASS_NAMES, icons)):
        cols[i % 5].markdown(f"{icon} **{name.capitalize()}**")
