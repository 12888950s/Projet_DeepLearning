import os
import json
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="SkinCare AI",
    page_icon="🧴",
    layout="wide"
)


# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        font-weight: 800;
        color: #2E2E3A;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 22px;
        color: #555;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: #F8F9FA;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #E5E7EB;
        margin-bottom: 15px;
    }

    .class-card {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        text-align: center;
        font-weight: 600;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }

    .prediction-box {
        background-color: #ECFDF5;
        padding: 22px;
        border-radius: 15px;
        border-left: 6px solid #10B981;
        margin-top: 20px;
    }

    .warning-box {
        background-color: #FFF7ED;
        color: #92400E;
        padding: 18px;
        border-radius: 15px;
        border-left: 6px solid #F59E0B;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .footer {
        color: #888;
        font-size: 14px;
        margin-top: 40px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Paths
# =========================
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


# =========================
# Load model and classes
# =========================
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


@st.cache_data
def load_class_labels():
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)

    index_to_class = {v: k for k, v in class_indices.items()}
    return index_to_class


model = load_trained_model()
index_to_class = load_class_labels()


# =========================
# Preprocessing
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def predict_skin_condition(image):
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    predicted_class = index_to_class[predicted_index]

    return predicted_class, confidence, predictions[0]


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("📌 Project Info")

    st.markdown(
        """
        **Project:** SkinCare AI  
        **Model:** MobileNetV2 Transfer Learning  
        **Task:** Image Classification  
        **Classes:** 4  
        """
    )

    st.markdown("---")

    st.markdown("### Model Performance")
    st.metric("Test Accuracy", "70.65%")
    st.metric("Macro F1-score", "0.72")
    st.metric("Weighted F1-score", "0.70")

    st.markdown("---")

    st.markdown(
        """
        ### Classes
        - Acne
        - Eczema
        - Psoriasis
        - Rosacea
        """
    )


# =========================
# Header
# =========================
st.markdown('<div class="main-title">🧴 SkinCare AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">CNN-Based Preliminary Skin Disease Image Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
    This application uses a <b>MobileNetV2 Transfer Learning</b> model to classify
    dermatological images into four skin disease categories.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Classes cards
# =========================
st.markdown("### Supported Classes")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="class-card">Acne</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="class-card">Eczema</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="class-card">Psoriasis</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="class-card">Rosacea</div>', unsafe_allow_html=True)


# =========================
# Medical warning
# =========================
st.markdown(
    """
    <div class="warning-box">
    ⚠️ <b>Medical Disclaimer:</b> This application is for educational purposes only.
    It does not replace a professional medical diagnosis by a dermatologist.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Upload and prediction
# =========================
st.markdown("### Upload Image")

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True)

    with right_col:
        st.markdown("### Prediction Panel")

        if st.button("🔍 Predict Skin Condition", use_container_width=True):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_predictions = predict_skin_condition(image)

            st.markdown(
                f"""
                <div class="prediction-box">
                <h3>Prediction Result</h3>
                <p><b>Predicted Class:</b> {predicted_class}</p>
                <p><b>Confidence:</b> {confidence * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Confidence Score")
            st.progress(confidence)

            st.markdown("### Class Probabilities")

            for idx, prob in enumerate(all_predictions):
                class_name = index_to_class[idx]
                st.write(f"**{class_name}:** {prob * 100:.2f}%")
                st.progress(float(prob))

            if confidence < 0.60:
                st.warning(
                    "The confidence score is relatively low. "
                    "The prediction should be interpreted carefully."
                )
            else:
                st.success("The model produced a confident prediction.")

else:
    st.info("Please upload an image to start prediction.")


# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    '<div class="footer">SkinCare AI — Deep Learning Project | MobileNetV2 Transfer Learning | Souha Guezguez</div>',
    unsafe_allow_html=True
)