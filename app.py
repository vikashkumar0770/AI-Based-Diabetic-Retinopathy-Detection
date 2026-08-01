import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from class_names import CLASS_NAMES

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Diabetic Retinopathy Detection",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.main{
    background-color:#0f172a;
}

h1,h2,h3{
    color:white;
}

div[data-testid="stMetricValue"]{
    color:#00ff88;
}

div[data-testid="stMetricLabel"]{
    color:white;
}

.stButton>button{
    width:100%;
    border-radius:12px;
    height:50px;
    font-size:18px;
}

.reportview-container{
    background:#0f172a;
}

.block-container{
    padding-top:1rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_my_model():
    return load_model("best_model.keras")

model = load_my_model()

# Normalize CLASS_NAMES to a list regardless of whether class_names.py
# defines it as a dict {0: "...", 1: "..."} or a list ["...", "..."]
if isinstance(CLASS_NAMES, dict):
    CLASS_NAMES_LIST = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
else:
    CLASS_NAMES_LIST = list(CLASS_NAMES)

# ==========================================
# Grad-CAM Helper Functions
# ==========================================

def find_last_conv_layer(keras_model):
    """
    Recursively searches (including inside nested sub-models like a wrapped
    EfficientNetB0) for the last Conv2D-type layer and returns the layer object.
    """
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.Model):
            found = find_last_conv_layer(layer)
            if found is not None:
                return found
        elif isinstance(layer, conv_types):
            return layer
    return None


def build_grad_model(model, conv_layer):
    """
    Builds a model mapping the ORIGINAL model's inputs to
    [conv_layer output, model output]. Works even if conv_layer
    belongs to a nested sub-model, as long as that sub-model's
    input tensor is traceable back to the outer model's input.
    """
    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )


def make_gradcam_heatmap(img_array, model, grad_model, pred_index=None):
    """Generates a Grad-CAM heatmap for the given image and predicted class."""

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    # (grad_model is built once by the caller via build_grad_model)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4):
    """Overlays the Grad-CAM heatmap on top of the original retina image."""
    base_image = np.array(original_image.resize((224, 224)))

    heatmap_resized = cv2.resize(heatmap, (base_image.shape[1], base_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    try:
        jet = cm.get_cmap("jet")
    except AttributeError:
        jet = cm.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    superimposed_img = jet_heatmap * alpha + base_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    return superimposed_img


def draw_circle_on_hotspot(original_image, heatmap, threshold=0.6):
    """Draws a circle around the region with the highest activation (disease hotspot)."""
    base_image = np.array(original_image.resize((224, 224))).copy()

    heatmap_resized = cv2.resize(heatmap, (base_image.shape[1], base_image.shape[0]))
    binary_mask = np.uint8(heatmap_resized >= threshold) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(base_image, center, max(radius, 15), (255, 0, 0), 3)

    return base_image

# ==========================================
# Disease Information
# ==========================================

disease_info = {

0: {
"name": "No Diabetic Retinopathy",
"hindi": "रेटिना सामान्य दिखाई दे रही है। किसी प्रकार की डायबिटिक रेटिनोपैथी नहीं मिली।",
"english": "No Diabetic Retinopathy Detected.",
"severity": "Normal",
"recommendation_hindi": "✔ नियमित Eye Checkup कराते रहें।",
"recommendation_english": "✔ Continue Regular Eye Checkup."
},

1: {
"name": "Mild Diabetic Retinopathy",
"hindi": "प्रारंभिक स्तर की डायबिटिक रेटिनोपैथी पाई गई है।",
"english": "Mild Diabetic Retinopathy Detected.",
"severity": "Low",
"recommendation_hindi": "✔ Blood Sugar नियंत्रित रखें।",
"recommendation_english": "✔ Maintain Blood Sugar."
},

2: {
"name": "Moderate Diabetic Retinopathy",
"hindi": "मध्यम स्तर की डायबिटिक रेटिनोपैथी पाई गई है।",
"english": "Moderate Diabetic Retinopathy Detected.",
"severity": "Medium",
"recommendation_hindi": "✔ Eye Specialist से सलाह लें।",
"recommendation_english": "✔ Consult Ophthalmologist."
},

3: {
"name": "Severe Diabetic Retinopathy",
"hindi": "गंभीर स्तर की डायबिटिक रेटिनोपैथी पाई गई है।",
"english": "Severe Diabetic Retinopathy Detected.",
"severity": "High",
"recommendation_hindi": "✔ तुरंत उपचार कराएं।",
"recommendation_english": "✔ Immediate Treatment Recommended."
},

4: {
"name": "Proliferative Diabetic Retinopathy",
"hindi": "रेटिना में अत्यधिक गंभीर अवस्था की डायबिटिक रेटिनोपैथी पाई गई है।",
"english": "Proliferative Diabetic Retinopathy Detected.",
"severity": "Critical",
"recommendation_hindi": "✔ तुरंत अस्पताल जाएँ।",
"recommendation_english": "✔ Immediate Hospital Visit Required."
}

}

# ==========================================
# Title
# ==========================================

st.title("👁 AI Based Diabetic Retinopathy Detection")

st.markdown("""
### Upload Retina Fundus Image

AI will analyse the Retina and generate a professional bilingual medical report.

---

### रेटिना इमेज अपलोड करें

AI आपकी रेटिना इमेज का विश्लेषण करके हिंदी एवं अंग्रेज़ी दोनों भाषाओं में रिपोर्ट तैयार करेगा।

""")

uploaded_file = st.file_uploader(
    "Choose Retina Image",
    type=["png", "jpg", "jpeg"]
)

# ==========================================
# Image Upload & Prediction
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼 Uploaded Retina Image")
        st.image(image, use_container_width=True)

    # Image Preprocessing
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    probabilities = prediction[0] * 100

    info = disease_info[predicted_class]

    with col2:
        st.subheader("🤖 AI Diagnosis")
        st.success(info["name"])
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)
        st.metric("Severity", info["severity"])

    st.divider()

    st.subheader("📊 Prediction Probability")

    probability_df = pd.DataFrame({
        "Disease": CLASS_NAMES_LIST,
        "Probability (%)": np.round(probabilities, 2)
    })

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        probability_df.set_index("Disease")
    )

    st.divider()

    # ==========================================
    # Grad-CAM: Mark suspicious region (only if disease detected)
    # ==========================================

    st.subheader("🔍 AI Attention Map")

    if predicted_class != 0:
        try:
            conv_layer = find_last_conv_layer(model)
            if conv_layer is None:
                raise ValueError("No Conv2D-type layer found in the model.")

            grad_model = build_grad_model(model, conv_layer)
            heatmap = make_gradcam_heatmap(
                img_array, model, grad_model, pred_index=predicted_class
            )

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**Heatmap Overlay**")
                heatmap_img = overlay_heatmap_on_image(image, heatmap)
                st.image(heatmap_img, use_container_width=True)

            with col4:
                st.markdown("**Marked Suspicious Region**")
                circled_img = draw_circle_on_hotspot(image, heatmap, threshold=0.6)
                st.image(circled_img, use_container_width=True)

            st.caption("🔴 Red circle us region ko highlight karta hai jaha model ko sabse zyada abnormality dikhi.")

        except Exception as e:
            st.warning(f"Attention map generate nahi ho payi: {e}")

    else:
        st.info("✅ Image normal hai — koi suspicious region marking ki zaroorat nahi.")

    st.divider()

    # ==========================================
    # Professional Medical Report
    # ==========================================

    st.header("🩺 AI Medical Report")

    left, right = st.columns(2)

    with left:
        st.markdown("## 📄 हिंदी रिपोर्ट")
        st.info(info["hindi"])
        st.markdown("### 💊 सुझाव")
        st.success(info["recommendation_hindi"])

    with right:
        st.markdown("## 📄 English Report")
        st.info(info["english"])
        st.markdown("### 💊 Recommendation")
        st.success(info["recommendation_english"])

    st.divider()

    # ==========================================
    # Prediction Summary
    # ==========================================

    st.subheader("📋 AI Prediction Summary")

    summary = f"""
Disease Detected : {info['name']}

Confidence : {confidence:.2f} %

Severity : {info['severity']}

This AI prediction is intended only for educational purposes.

Please consult a qualified Ophthalmologist for final diagnosis.
"""

    st.text_area("Summary", summary, height=180)

    # ==========================================
    # Download Report
    # ==========================================

    report = f"""
==========================================
AI DIABETIC RETINOPATHY DETECTION REPORT
==========================================

Disease:
{info["name"]}

Confidence:
{confidence:.2f} %

Severity:
{info["severity"]}

------------------------------------------

Hindi Report

{info["hindi"]}

------------------------------------------

English Report

{info["english"]}

------------------------------------------

Recommendation (Hindi)

{info["recommendation_hindi"]}

------------------------------------------

Recommendation (English)

{info["recommendation_english"]}

==========================================
"""

    st.download_button(
        "📥 Download Medical Report",
        report,
        file_name="Diabetic_Retinopathy_Report.txt",
        mime="text/plain"
    )

    st.divider()

    st.markdown("""
---
### 👨‍⚕️ AI Medical Assistant

This application uses **EfficientNetB0** for Diabetic Retinopathy Detection.

**Disclaimer:** This AI prediction is only for educational and research purposes.

Always consult an Ophthalmologist for medical advice.

Developed using TensorFlow • Streamlit • Python

© 2025
""")
