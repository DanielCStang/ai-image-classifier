import streamlit as st
import torch
import pandas as pd
import numpy as np
import io

from PIL import Image
from datetime import datetime
from transformers import pipeline
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Fast AI Image Classifier",
    page_icon="⚡",
    layout="wide"
)

# =========================
# SESSION STATE
# =========================

if "model" not in st.session_state:
    st.session_state.model = None

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# MODEL OPTIONS (FAST FIRST)
# =========================

MODELS = {
    "⚡ Fast (MobileNet)": "google/mobilenet_v2_1.0_224",
    "⚖️ Balanced (ResNet50)": "microsoft/resnet-50",
    "🎯 Accurate (ViT)": "google/vit-base-patch16-224"
}

# =========================
# LOAD MODEL (CACHED)
# =========================

@st.cache_resource
def load_model(model_name: str):

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "image-classification",
        model=model_name,
        device=device
    )

# =========================
# IMAGE PREPROCESS
# =========================

def preprocess(image: Image.Image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    # FAST: reduce resolution BEFORE inference
    image.thumbnail((224, 224))

    return image

# =========================
# PREDICT
# =========================

def predict(image, model, top_k=3):

    image = preprocess(image)

    with torch.inference_mode():
        preds = model(image, top_k=top_k)

    return preds

# =========================
# CHART (FAST PLOTLY)
# =========================

def plot_results(preds):

    df = pd.DataFrame(preds)

    fig = px.bar(
        df,
        x="score",
        y="label",
        orientation="h",
        color="score",
        color_continuous_scale="viridis"
    )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig

# =========================
# UI HEADER
# =========================

st.title("⚡ Fast AI Image Classifier")
st.caption("Optimized for speed + deployment (Streamlit Cloud / Hugging Face Spaces)")

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("Settings")

    model_choice = st.selectbox("Model", list(MODELS.keys()))
    top_k = st.slider("Top K", 1, 5, 3)
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)

    if st.button("🚀 Load Model"):

        st.session_state.model = load_model(MODELS[model_choice])
        st.success("Model loaded!")

# =========================
# MAIN UI
# =========================

tab1, tab2 = st.tabs(["📷 Classify", "📂 History"])

# =========================
# TAB 1 - CLASSIFY
# =========================

with tab1:

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded and st.session_state.model:

        image = Image.open(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        with col2:

            if st.button("Run Classification"):

                preds = predict(image, st.session_state.model, top_k)

                st.subheader("Results")

                for i, p in enumerate(preds):

                    icon = "🟢" if p["score"] >= threshold else "🟡"

                    st.write(
                        f"{i+1}. {icon} **{p['label']}** - {p['score']:.2%}"
                    )

                st.plotly_chart(plot_results(preds), use_container_width=True)

                # store compressed history (FAST MEMORY)
                thumb = image.copy()
                thumb.thumbnail((128, 128))

                buffer = io.BytesIO()
                thumb.save(buffer, format="JPEG", quality=60)

                st.session_state.history.append({
                    "name": uploaded.name,
                    "time": datetime.now(),
                    "preds": preds,
                    "img": buffer.getvalue()
                })

    elif uploaded:
        st.warning("Load model first")

# =========================
# TAB 2 - HISTORY
# =========================

with tab2:

    st.header("History")

    if not st.session_state.history:
        st.info("No images yet")

    for item in reversed(st.session_state.history):

        with st.expander(f"{item['name']} - {item['time'].strftime('%H:%M:%S')}"):

            img = Image.open(io.BytesIO(item["img"]))
            st.image(img, width=200)

            for p in item["preds"][:3]:
                st.write(f"**{p['label']}** - {p['score']:.2%}")
