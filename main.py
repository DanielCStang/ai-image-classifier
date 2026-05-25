import streamlit as st
import torch
import pandas as pd
import numpy as np
import io

from PIL import Image
from datetime import datetime
from transformers import pipeline
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Image Classification Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================

if "model" not in st.session_state:
    st.session_state.model = None

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# MODEL OPTIONS
# =========================================================

MODELS = {
    "⚡ Fast (MobileNet)": "google/mobilenet_v2_1.0_224",
    "⚖️ Balanced (ResNet50)": "microsoft/resnet-50",
    "🎯 Accurate (ViT)": "google/vit-base-patch16-224"
}

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(model_name: str):

    device = 0 if torch.cuda.is_available() else -1

    try:
        model = pipeline(
            "image-classification",
            model=model_name,
            device=device
        )
        return model

    except Exception as e:

        st.warning(f"Primary model failed: {e}")

        try:
            fallback_model = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=-1
            )

            st.success("Fallback model loaded successfully")
            return fallback_model

        except Exception as fallback_error:
            st.error(f"Fallback model also failed: {fallback_error}")
            return None

# =========================================================
# IMAGE PREPROCESS
# =========================================================

def preprocess(image: Image.Image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize for faster inference
    image.thumbnail((224, 224))

    return image

# =========================================================
# PREDICT
# =========================================================

def predict(image, model, top_k=5):

    try:
        image = preprocess(image)

        with torch.inference_mode():
            preds = model(image, top_k=top_k)

        return preds

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# =========================================================
# PLOTLY CHART
# =========================================================

def plot_results(preds):

    df = pd.DataFrame(preds)

    fig = px.bar(
        df,
        x="score",
        y="label",
        orientation="h",
        color="score",
        color_continuous_scale="viridis",
        text="score"
    )

    fig.update_traces(
        texttemplate='%{text:.2%}',
        textposition='outside'
    )

    fig.update_layout(
        title="Top Predictions",
        xaxis_title="Confidence Score",
        yaxis_title="Class",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig

# =========================================================
# ANALYTICS DASHBOARD
# =========================================================

def analytics_dashboard():

    history = st.session_state.history

    if not history:
        st.info("No classified images yet.")
        return

    all_data = []

    for item in history:

        for pred in item["preds"]:

            all_data.append({
                "Image": item["name"],
                "Label": pred["label"],
                "Score": pred["score"],
                "Timestamp": item["time"]
            })

    df = pd.DataFrame(all_data)

    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("📈 Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", len(history))

    with col2:
        st.metric("Total Predictions", len(df))

    with col3:
        avg_conf = df["Score"].mean()
        st.metric("Avg Confidence", f"{avg_conf:.2%}")

    with col4:
        top_class = df["Label"].mode().iloc[0]
        st.metric("Most Common Class", top_class)

    # =====================================================
    # CHARTS
    # =====================================================

    st.subheader("📊 Analytics")

    col1, col2 = st.columns(2)

    with col1:

        class_counts = (
            df["Label"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        class_counts.columns = ["Label", "Count"]

        fig1 = px.bar(
            class_counts,
            x="Count",
            y="Label",
            orientation="h",
            title="Top Predicted Classes",
            color="Count"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        fig2 = px.histogram(
            df,
            x="Score",
            nbins=20,
            title="Confidence Score Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # DETAILED TABLE
    # =====================================================

    st.subheader("📋 Detailed Results")

    detailed = []

    for item in history:

        top_pred = item["preds"][0]

        detailed.append({
            "Image": item["name"],
            "Top Prediction": top_pred["label"],
            "Confidence": f"{top_pred['score']:.2%}",
            "Timestamp": item["time"].strftime("%Y-%m-%d %H:%M:%S")
        })

    results_df = pd.DataFrame(detailed)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # CSV EXPORT
    # =====================================================

    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Results CSV",
        data=csv_buffer.getvalue(),
        file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # =====================================================
    # CLEAR BUTTON
    # =====================================================

    if st.button("🗑️ Clear All Results"):

        st.session_state.history = []
        st.rerun()

# =========================================================
# UI HEADER
# =========================================================

st.title("🖼️ AI Image Classification Dashboard")

st.markdown("""
Classify images using Hugging Face AI models with:
- ⚡ Fast inference
- 📊 Analytics dashboard
- 📂 Classification history
- 📥 CSV export
- 🚀 Deployment optimization
""")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Choose Model",
        list(MODELS.keys())
    )

    top_k = st.slider(
        "Top K Predictions",
        1,
        10,
        5
    )

    threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.2
    )

    if st.button("🚀 Load Model", type="primary"):

        with st.spinner("Loading model..."):

            st.session_state.model = load_model(
                MODELS[model_choice]
            )

        if st.session_state.model:
            st.success("Model loaded successfully!")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📷 Classify",
    "📂 History",
    "📊 Analytics"
])

# =========================================================
# TAB 1 - CLASSIFY
# =========================================================

with tab1:

    st.header("Image Classification")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        with col2:

            if st.session_state.model:

                if st.button("🔍 Run Classification"):

                    with st.spinner("Classifying image..."):

                        preds = predict(
                            image,
                            st.session_state.model,
                            top_k
                        )

                    if preds:

                        st.subheader("🎯 Predictions")

                        for i, p in enumerate(preds):

                            icon = (
                                "🟢"
                                if p["score"] >= threshold
                                else "🟡"
                            )

                            st.write(
                                f"{i+1}. {icon} "
                                f"**{p['label']}** "
                                f"- {p['score']:.2%}"
                            )

                        st.plotly_chart(
                            plot_results(preds),
                            use_container_width=True
                        )

                        # =====================================
                        # STORE COMPRESSED HISTORY
                        # =====================================

                        thumb = image.copy()
                        thumb.thumbnail((128, 128))

                        buffer = io.BytesIO()

                        thumb.save(
                            buffer,
                            format="JPEG",
                            quality=60
                        )

                        st.session_state.history.append({
                            "name": uploaded.name,
                            "time": datetime.now(),
                            "preds": preds,
                            "img": buffer.getvalue()
                        })

            else:
                st.warning("Please load a model first.")

# =========================================================
# TAB 2 - HISTORY
# =========================================================

with tab2:

    st.header("📂 Classification History")

    history = st.session_state.history

    if not history:
        st.info("No classified images yet.")

    else:

        st.write(f"Total images classified: {len(history)}")

        for item in reversed(history):

            with st.expander(
                f"{item['name']} - "
                f"{item['time'].strftime('%Y-%m-%d %H:%M:%S')}"
            ):

                col1, col2 = st.columns([1, 2])

                with col1:

                    img = Image.open(
                        io.BytesIO(item["img"])
                    )

                    st.image(img, width=200)

                with col2:

                    st.write("### Top Predictions")

                    for p in item["preds"][:3]:

                        st.write(
                            f"• **{p['label']}** "
                            f"- {p['score']:.2%}"
                        )

# =========================================================
# TAB 3 - ANALYTICS
# =========================================================

with tab3:

    st.header("📊 Results & Analytics")

    analytics_dashboard()

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: gray;'>

Built with ❤️ using:
- Streamlit
- Hugging Face Transformers
- Plotly
- PyTorch

Optimized for:
⚡ Fast inference
🚀 Cloud deployment
📊 Analytics dashboards

</div>
""", unsafe_allow_html=True)
