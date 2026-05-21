import os
import json
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ======================================================
# Page configuration
# ======================================================
st.set_page_config(
    page_title="SkinCare AI",
    page_icon="🧴",
    layout="wide"
)


# ======================================================
# Custom CSS
# ======================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #F8FBFF 0%, #FFFFFF 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EEF6FF 0%, #FFFFFF 100%);
        border-right: 1px solid #DBEAFE;
    }

    .sidebar-title {
        font-size: 28px;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 22px;
        letter-spacing: -0.5px;
    }

    .sidebar-card {
        background: #FFFFFF;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }

    .sidebar-section-title {
        font-size: 18px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 12px;
    }

    .sidebar-text {
        font-size: 15px;
        color: #334155;
        line-height: 1.8;
    }

    .sidebar-badge {
        display: inline-block;
        background: #DBEAFE;
        color: #1D4ED8;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        margin: 4px 4px 4px 0;
    }

    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 100%);
        padding: 32px;
        border-radius: 24px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 30px;
    }

    .main-title {
        font-size: 56px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 24px;
        color: #4B5563;
        margin-bottom: 18px;
        font-weight: 500;
    }

    .description-text {
        font-size: 17px;
        color: #374151;
        line-height: 1.7;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 18px;
    }

    .class-card {
        background: #FFFFFF;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        text-align: center;
        font-weight: 800;
        font-size: 18px;
        color: #1F2937;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        transition: 0.2s ease-in-out;
    }

    .class-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10);
    }

    .upload-box {
        background: #FFFFFF;
        padding: 22px;
        border-radius: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }

    .result-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        padding: 26px;
        border-radius: 20px;
        border-left: 7px solid #10B981;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.16);
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 26px;
        font-weight: 850;
        color: #065F46;
        margin-bottom: 14px;
    }

    .result-text {
        font-size: 18px;
        color: #064E3B;
        margin-bottom: 8px;
    }

    .low-confidence-box {
        background: #FEF2F2;
        color: #991B1B;
        padding: 18px;
        border-radius: 16px;
        border-left: 6px solid #EF4444;
        margin-top: 15px;
    }

    .success-confidence-box {
        background: #EFF6FF;
        color: #1E40AF;
        padding: 18px;
        border-radius: 16px;
        border-left: 6px solid #3B82F6;
        margin-top: 15px;
    }

    .footer {
        color: #6B7280;
        font-size: 14px;
        text-align: center;
        margin-top: 45px;
        padding-bottom: 20px;
    }

    .stButton > button {
        border-radius: 14px;
        height: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# Paths
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2_phase1_best_model.keras"
)

CLASS_INDICES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_indices.json"
)


# ======================================================
# Load model and class labels
# ======================================================
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


@st.cache_data
def load_class_labels():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    return {v: k for k, v in class_indices.items()}


model = load_trained_model()
index_to_class = load_class_labels()


# ======================================================
# Image preprocessing
# ======================================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array


# ======================================================
# Prediction function
# ======================================================
def predict_skin_condition(image: Image.Image):
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    predicted_class = index_to_class[predicted_index]

    return predicted_class, confidence, predictions


# ======================================================
# Sidebar
# ======================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">🧴 SkinCare AI</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-section-title">Project Overview</div>
            <div class="sidebar-text">
                <b>Project:</b> SkinCare AI<br>
                <b>Task:</b> Image Classification<br>
                <b>Domain:</b> Computer Vision<br>
                <b>Approach:</b> Deep Learning
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-section-title">Final Model</div>
            <div class="sidebar-text">
                <b>Architecture:</b><br>
                MobileNetV2 Transfer Learning<br><br>
                <b>Selected model:</b><br>
                Phase 1 without fine-tuning
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-section-title">Model Performance</div>
            <div class="sidebar-text">
                <b>Test Accuracy:</b> 70.65%<br>
                <b>Macro F1-score:</b> 0.72<br>
                <b>Weighted F1-score:</b> 0.70
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-section-title">Supported Classes</div>
            <span class="sidebar-badge">Acne</span>
            <span class="sidebar-badge">Eczema</span>
            <span class="sidebar-badge">Psoriasis</span>
            <span class="sidebar-badge">Rosacea</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# Header
# ======================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">🧴 SkinCare AI</div>
        <div class="subtitle">CNN-Based Preliminary Skin Disease Image Classification</div>
        <div class="description-text">
            This application uses a <b>MobileNetV2 Transfer Learning</b> model to classify
            dermatological images into four selected skin disease categories.
            The model was trained and evaluated as part of a Deep Learning and Computer Vision project.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ======================================================
# Supported classes
# ======================================================
st.markdown('<div class="section-title">Supported Classes</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="class-card">Acne</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="class-card">Eczema</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="class-card">Psoriasis</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="class-card">Rosacea</div>', unsafe_allow_html=True)


# ======================================================
# Upload section
# ======================================================
st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="upload-box">
        Upload a dermatological image in JPG, JPEG or PNG format, then click the prediction button.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)


# ======================================================
# Prediction interface
# ======================================================
if uploaded_file is None:
    st.info("Please upload an image to start prediction.")

else:
    image = Image.open(uploaded_file)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)

        if st.button("Predict Skin Condition", use_container_width=True):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probabilities = predict_skin_condition(image)

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-title">Prediction Result</div>
                    <div class="result-text"><b>Predicted Class:</b> {predicted_class}</div>
                    <div class="result-text"><b>Confidence:</b> {confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Confidence Score")
            st.progress(confidence)

            st.markdown("### Class Probabilities")

            for idx, prob in enumerate(probabilities):
                class_name = index_to_class[idx]
                st.write(f"**{class_name}:** {prob * 100:.2f}%")
                st.progress(float(prob))

            if confidence < 0.60:
                st.markdown(
                    """
                    <div class="low-confidence-box">
                        The confidence score is relatively low. Please interpret the result carefully.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="success-confidence-box">
                        The model produced a confident prediction.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ======================================================
# Footer
# ======================================================
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        SkinCare AI — Deep Learning Project | MobileNetV2 Transfer Learning | Souha Guezguez
    </div>
    """,
    unsafe_allow_html=True
)