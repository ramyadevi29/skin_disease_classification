
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
import requests
from PIL import Image

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="SkinCare AI",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.header {
    background: linear-gradient(135deg, #e8f4ff, #f7fbff);
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.header h1 {
    color: #17324d;
    font-size: 42px;
    margin-bottom: 8px;
}

.header p {
    color: #526779;
    font-size: 17px;
}

.result-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #dce5ee;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}

/* =================================================
   PIPELINE BOX DESIGN
   ================================================= */

.info-box {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #dce5ee;
    text-align: center;
    color: #17324d;
    min-height: 180px;
}

.info-box h2 {
    color: #17324d;
    font-size: 38px;
    margin-bottom: 10px;
}

.info-box h3 {
    color: #17324d;
    font-size: 18px;
    margin-bottom: 10px;
}

.info-box p {
    color: #526779;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #718096;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = (
    "skin_disease_convnext_tiny_improved (1).keras"
)

CLASS_PATH = "class_names.json"

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =====================================================
# LOAD CLASS NAMES
# =====================================================

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# =====================================================
# SEVERITY ASSESSMENT FUNCTION
# =====================================================

def assess_severity(image):

    image_array = np.array(image)

    image_array = cv2.resize(
        image_array,
        (224, 224)
    )

    hsv = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2HSV
    )

    # Detect stronger reddish regions

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([12, 255, 255])

    lower_red2 = np.array([168, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    red_mask = mask1 + mask2

    # Remove small noise

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Calculate affected area

    affected_pixels = np.count_nonzero(
        red_mask
    )

    total_pixels = (
        red_mask.shape[0] *
        red_mask.shape[1]
    )

    affected_percentage = (
        affected_pixels /
        total_pixels
    ) * 100

    # Rule-based severity

    if affected_percentage < 8:

        severity = "Mild"

    elif affected_percentage < 70:

        severity = "Moderate"

    else:

        severity = "Severe"

    return severity, affected_percentage

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="header">

<h1>🩺 SkinCare AI</h1>

<p>
AI-Based Skin Disease Classification using
ConvNeXt-Tiny and Transfer Learning
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# IMAGE UPLOAD
# =====================================================

st.markdown("## 🔍 Skin Image Analysis")

st.write(
    "Upload a skin image to classify the possible skin disease "
    "and view the confidence score."
)

uploaded_file = st.file_uploader(
    "📷 Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# IMAGE + PREDICTION + SEVERITY
# =====================================================

if uploaded_file is not None:

    # Open uploaded image

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("---")

    image_col, result_col = st.columns(2)

    # =================================================
    # DISPLAY IMAGE
    # =================================================

    with image_col:

        st.markdown(
            "### 📷 Uploaded Image"
        )

        st.image(
            img,
            use_container_width=True
        )

    # =================================================
    # MODEL PREDICTION
    # =================================================

    with result_col:

        st.markdown(
            "### 🧠 Prediction Result"
        )

        img_resized = img.resize(
            (224, 224)
        )

        img_array = np.array(
            img_resized
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_disease = class_names[
            predicted_index
        ]

        confidence = (
            predictions[0][predicted_index] *
            100
        )

        # Result display

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.write(
            "**Predicted Disease**"
        )

        st.subheader(
            predicted_disease
        )

        st.write(
            "**Confidence Score**"
        )

        st.subheader(
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # =================================================
    # DISPLAY SEVERITY
    # =================================================

    severity, affected_percentage = assess_severity(
        img
    )

    st.markdown("---")

    st.markdown(
        "## 🩺 Severity Assessment"
    )

    severity_col, area_col = st.columns(2)

    with severity_col:

        st.write(
            "**Severity Level**"
        )

        if severity == "Mild":

            st.success(
                "🟢 Mild"
            )

        elif severity == "Moderate":

            st.warning(
                "🟡 Moderate"
            )

        else:

            st.error(
                "🔴 Severe"
            )

    with area_col:

        st.write(
            "**Estimated Affected Area**"
        )

        st.subheader(
            f"{affected_percentage:.2f}%"
        )

    # =================================================
    # DECISION & RECOMMENDATION
    # =================================================

    st.markdown("---")

    st.markdown(
        "## 💡 Decision & Recommendation"
    )

    if severity == "Mild":

        st.success(
            "🟢 Mild condition detected"
        )

        st.write(
            "### 🏠 Home Care Tips"
        )

        st.write(
            "• Keep the affected skin clean and dry."
        )

        st.write(
            "• Avoid scratching or irritating the affected area."
        )

        st.write(
            "• Use gentle, fragrance-free skin-care products."
        )

        st.write(
            "• Monitor the affected area for changes."
        )

        st.info(
            "If the condition becomes worse or does not improve, "
            "consult a dermatologist."
        )

    elif severity == "Moderate":

        st.warning(
            "🟡 Moderate condition detected"
        )

        st.write(
            "### 🏠 Home Care & Monitoring"
        )

        st.write(
            "• Keep the affected area clean and dry."
        )

        st.write(
            "• Avoid scratching, rubbing, or irritating the skin."
        )

        st.write(
            "• Use gentle skin-care products."
        )

        st.write(
            "• Monitor the affected area regularly."
        )

        st.info(
            "Consider consulting a dermatologist for further evaluation."
        )

    else:

        st.error(
            "🔴 Severe condition detected"
        )

        st.write(
            "### 🩺 Doctor Consultation"
        )

        st.write(
            "• Dermatologist consultation is recommended."
        )

        st.write(
            "• Avoid self-treating the affected area."
        )

        st.write(
            "• Seek professional evaluation for appropriate treatment."
        )

# =====================================================
# NEARBY DERMATOLOGIST
# =====================================================

st.markdown("---")

st.markdown(
    "## 📍 Nearby Dermatologist"
)

st.write(
    "Enter your city or location to find nearby dermatologists."
)

# Put your NEW OpenCage API key here
OPENCAGE_API_KEY = "92f5c00ab7f042fb8c9a32baa231ce84"


def get_coordinates(place_name):

    url = "https://api.opencagedata.com/geocode/v1/json"

    params = {
        "q": place_name,
        "key": OPENCAGE_API_KEY,
        "limit": 1,
        "countrycode": "in"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if data["results"]:

            geometry = data["results"][0]["geometry"]

            latitude = geometry["lat"]
            longitude = geometry["lng"]

            location_name = data["results"][0]["formatted"]

            return (
                latitude,
                longitude,
                location_name
            )

        return None, None, None

    except Exception as e:

        st.error(
            f"Unable to get location: {e}"
        )

        return None, None, None


place = st.text_input(
    "Enter your city or location",
    placeholder="Example: Coimbatore"
)


if st.button("🔎 Find Dermatologists"):

    if place.strip() == "":

        st.warning(
            "Please enter your location."
        )

    else:

        latitude, longitude, location_name = (
            get_coordinates(place)
        )

        if latitude is not None:

            st.success(
                f"Location found: {location_name}"
            )

            st.write(
                "Latitude:",
                latitude
            )

            st.write(
                "Longitude:",
                longitude
            )

            maps_url = (
                "https://www.google.com/maps/search/"
                "dermatologist+near+"
                f"{latitude},{longitude}"
            )

            st.markdown(
                f"[🗺️ Find Nearby Dermatologists]({maps_url})"
            )

        else:

            st.error(
                "Location not found. "
                "Please enter a valid city or place."
            )

# =====================================================
# AI SKIN-CARE PROGRESS TRACKING
# =====================================================

st.markdown("---")

st.markdown(
    "## 📊 AI Skin-Care Progress Tracking"
)

st.write(
    "Upload a previous and current skin image to compare "
    "their deep image features using Cosine Similarity."
)

# =====================================================
# FEATURE EXTRACTION MODEL
# =====================================================

@st.cache_resource
def create_feature_model():

    # Find the 768-dimensional feature layer

    feature_layer = None

    for layer in reversed(model.layers):

        try:

            output_shape = layer.output.shape

            if (
                len(output_shape) == 2
                and output_shape[-1] == 768
            ):

                feature_layer = layer
                break

        except Exception:
            continue

    if feature_layer is None:

        raise ValueError(
            "768-dimensional ConvNeXt feature layer "
            "could not be found in the model."
        )

    return tf.keras.Model(
        inputs=model.input,
        outputs=feature_layer.output
    )


feature_model = create_feature_model()

# =====================================================
# FEATURE EXTRACTION FUNCTION
# =====================================================

def extract_features(image):

    image_resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image_resized
    ).astype(np.float32)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    features = feature_model.predict(
        image_array,
        verbose=0
    )

    return features.flatten()

# =====================================================
# COSINE SIMILARITY FUNCTION
# =====================================================

def calculate_cosine_similarity(
    feature1,
    feature2
):

    feature1 = np.asarray(
        feature1,
        dtype=np.float32
    )

    feature2 = np.asarray(
        feature2,
        dtype=np.float32
    )

    numerator = np.dot(
        feature1,
        feature2
    )

    denominator = (
        np.linalg.norm(feature1) *
        np.linalg.norm(feature2)
    )

    if denominator == 0:

        return 0.0

    similarity = (
        numerator /
        denominator
    )

    return float(similarity)

# =====================================================
# IMAGE UPLOAD
# =====================================================

st.markdown(
    "### 📷 Upload Images"
)

previous_image_file = st.file_uploader(
    "Upload Previous Skin Image",
    type=["jpg", "jpeg", "png"],
    key="previous_skin_image"
)

current_image_file = st.file_uploader(
    "Upload Current Skin Image",
    type=["jpg", "jpeg", "png"],
    key="current_skin_image"
)

# =====================================================
# PROGRESS COMPARISON
# =====================================================

if (
    previous_image_file is not None
    and current_image_file is not None
):

    previous_image = Image.open(
        previous_image_file
    ).convert("RGB")

    current_image = Image.open(
        current_image_file
    ).convert("RGB")

    # =================================================
    # DISPLAY IMAGES
    # =================================================

    previous_col, current_col = st.columns(2)

    with previous_col:

        st.markdown(
            "### 🗓️ Previous Scan"
        )

        st.image(
            previous_image,
            use_container_width=True
        )

    with current_col:

        st.markdown(
            "### 🗓️ Current Scan"
        )

        st.image(
            current_image,
            use_container_width=True
        )

    # =================================================
    # EXTRACT DEEP FEATURES
    # =================================================

    previous_features = extract_features(
        previous_image
    )

    current_features = extract_features(
        current_image
    )

    # =================================================
    # COSINE SIMILARITY
    # =================================================

    similarity = calculate_cosine_similarity(
        previous_features,
        current_features
    )

    similarity_percentage = (
        similarity * 100
    )

    # =================================================
    # DISPLAY SIMILARITY
    # =================================================

    st.markdown("---")

    st.markdown(
        "### 🔬 Feature Similarity"
    )

    similarity_col, status_col = st.columns(2)

    with similarity_col:

        st.metric(
            "Cosine Similarity",
            f"{similarity_percentage:.2f}%"
        )

        st.progress(
            max(
                0.0,
                min(
                    similarity,
                    1.0
                )
            )
        )

    # =================================================
    # PROGRESS STATUS
    # =================================================

    with status_col:

        if similarity_percentage >= 80:

            st.info(
                "🔵 High Similarity"
            )

            st.write(
                "The current image has high similarity "
                "to the previous image."
            )

        elif similarity_percentage >= 50:

            st.warning(
                "🟡 Moderate Similarity"
            )

            st.write(
                "The current image has moderate similarity "
                "to the previous image."
            )

        else:

            st.success(
                "🟢 Low Similarity"
            )

            st.write(
                "The current image has lower similarity "
                "to the previous image."
            )

    # =================================================
    # EXPLANATION
    # =================================================

    st.info(
        "Cosine Similarity compares the deep image features "
        "extracted by ConvNeXt-Tiny. It indicates image-feature "
        "similarity and should not be considered a medical diagnosis."
    )

 
