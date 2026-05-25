import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import torch

from PIL import Image
from datetime import datetime
from transformers import pipeline

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "classifier" not in st.session_state:
    st.session_state.classifier = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "analyzed_images" not in st.session_state:
    st.session_state.analyzed_images = []

# =====================================================
# MODEL CONFIG
# =====================================================

MODEL_NAME = "google/mobilenet_v2_1.0_224"

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1
    )

    return classifier

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image for faster inference
    MAX_SIZE = (224, 224)
    image.thumbnail(MAX_SIZE)

    return image

# =====================================================
# CLASSIFICATION
# =====================================================

def classify_image(image, classifier, top_k=3):

    processed_image = preprocess_image(image)

    with torch.inference_mode():

        predictions = classifier(
            processed_image,
            top_k=top_k
        )

    return predictions

# =====================================================
# PREDICTION CHART
# =====================================================

def create_prediction_chart(predictions):

    labels = [pred["label"] for pred in predictions]
    scores = [pred["score"] for pred in predictions]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(
        labels,
        scores,
        color=["#4CAF50", "#2196F3", "#FFC107"]
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Top Predictions")

    for bar in bars:

        width = bar.get_width()

        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center"
        )

    plt.tight_layout()

    return fig

# =====================================================
# HEADER
# =====================================================

st.title("🖼️ AI Image Classification Dashboard")

st.markdown("""
Upload an image and classify it using a Hugging Face AI model.
Optimized for Codecademy performance.
""")

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Top Predictions",
        min_value=1,
        max_value=5,
        value=3
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1
    )

    if st.button("🚀 Load AI Model"):

        with st.spinner("Loading model..."):

            st.session_state.classifier = load_model()

            st.session_state.model_loaded = True

        st.success("Model loaded successfully!")

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs([
    "📷 Image Classifier",
    "📂 History"
])

# =====================================================
# IMAGE CLASSIFIER TAB
# =====================================================

with tab1:

    st.header("Classify an Image")
    
    with st.form("upload_form"):

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

        submit_button = st.form_submit_button(
            "🔍 Classify Image"
        )

    if uploaded_file and submit_button:

        if not st.session_state.model_loaded:

            st.error("Please load the AI model first.")

        else:

            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            # =========================================
            # DISPLAY IMAGE
            # =========================================

            with col1:

                st.image(
                    image,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            # =========================================
            # CLASSIFICATION
            # =========================================

            with col2:

                with st.spinner("Classifying image..."):

                    predictions = classify_image(
                        image,
                        st.session_state.classifier,
                        top_k
                    )

                st.subheader("🎯 Predictions")

                for i, pred in enumerate(predictions):

                    emoji = (
                        "🟢"
                        if pred["score"] >= confidence_threshold
                        else "🟡"
                    )

                    st.write(
                        f"{i+1}. "
                        f"{emoji} "
                        f"**{pred['label']}** "
                        f"- {pred['score']:.2%}"
                    )

                chart = create_prediction_chart(predictions)

                st.pyplot(chart)

                # =====================================
                # SAVE HISTORY
                # =====================================

                thumbnail = image.copy()
                thumbnail.thumbnail((128, 128))

                buffer = io.BytesIO()

                thumbnail.save(
                    buffer,
                    format="JPEG",
                    quality=60
                )

                st.session_state.analyzed_images.append({
                    "name": uploaded_file.name,
                    "timestamp": datetime.now(),
                    "predictions": predictions,
                    "image_bytes": buffer.getvalue()
                })

# =====================================================
# HISTORY TAB
# =====================================================

with tab2:

    st.header("📂 Classification History")

    if not st.session_state.analyzed_images:

        st.info("No images classified yet.")

    else:

        st.write(
            f"Total Images Classified: "
            f"{len(st.session_state.analyzed_images)}"
        )

        for img_data in reversed(
            st.session_state.analyzed_images
        ):

            with st.expander(

                f"{img_data['name']} - "
                f"{img_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"

            ):

                col1, col2 = st.columns([1, 2])

                # =====================================
                # IMAGE
                # =====================================

                with col1:

                    image = Image.open(
                        io.BytesIO(img_data["image_bytes"])
                    )

                    st.image(
                        image,
                        use_container_width=True
                    )

                # =====================================
                # PREDICTIONS
                # =====================================

                with col2:

                    st.write("### Top Predictions")

                    for i, pred in enumerate(
                        img_data["predictions"]
                    ):

                        st.write(
                            f"{i+1}. "
                            f"**{pred['label']}** "
                            f"- {pred['score']:.2%}"
                        )

# =====================================================
# DOWNLOAD RESULTS
# =====================================================

if st.session_state.analyzed_images:

    st.divider()

    st.subheader("📥 Export Results")

    export_rows = []

    for img_data in st.session_state.analyzed_images:

        top_pred = img_data["predictions"][0]

        export_rows.append({
            "Image": img_data["name"],
            "Prediction": top_pred["label"],
            "Confidence": f"{top_pred['score']:.2%}",
            "Timestamp": img_data["timestamp"]
        })

    export_df = pd.DataFrame(export_rows)

    csv = export_df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="classification_results.csv",
        mime="text/csv"
    )

# =====================================================
# CLEAR HISTORY
# =====================================================

if st.session_state.analyzed_images:

    if st.button("🗑️ Clear History"):

        st.session_state.analyzed_images.clear()

        st.success("History cleared!")