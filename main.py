import streamlit as st
import numpy as np
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    InceptionV3
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from PIL import Image

# ----------------------------------------------------
# MODEL LOADING FUNCTION
# ----------------------------------------------------
def load_model_and_preprocess(model_name):
    """Load model and matching preprocess function"""
    if model_name == "MobileNetV2":
        return MobileNetV2(weights="imagenet"), mobilenet_preprocess, (224, 224)
    elif model_name == "ResNet50":
        return ResNet50(weights="imagenet"), resnet_preprocess, (224, 224)
    elif model_name == "InceptionV3":
        return InceptionV3(weights="imagenet"), inception_preprocess, (299, 299)
    else:
        raise ValueError("Invalid model name")

# Cache models so they don’t reload every time
@st.cache_resource
def load_cached_model(model_name):
    return load_model_and_preprocess(model_name)

# ----------------------------------------------------
# IMAGE CLASSIFICATION FUNCTION
# ----------------------------------------------------
def classify_image(model, preprocess_input, uploaded_file, target_size):
    """Predict top 3 classes for an uploaded image"""
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]
        return decoded_preds
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None

# ----------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------
def main():
    st.set_page_config(page_title="🧠 Image Classifier", layout="centered")
    st.title("🧠 AI Image Classifier")
    st.write("Upload an image and choose a deep learning model to classify it!")

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose a model",
        ["MobileNetV2", "ResNet50", "InceptionV3"]
    )

    # Load the chosen model
    with st.spinner(f"Loading {model_choice}..."):
        model, preprocess_input, target_size = load_cached_model(model_choice)

    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                predictions = classify_image(model, preprocess_input, uploaded_file, target_size)

            if predictions:
                st.success("✅ Prediction Complete!")
                st.subheader("Top Predictions:")

                for pred_class, name, score in predictions:
                    readable_name = name.replace("_", " ").title()
                    st.write(f"**{readable_name}** — {score*100:.2f}% confidence")
                    st.progress(int(score * 100))

    st.markdown("---")
    st.caption("Built with TensorFlow · Streamlit · 🧠 AI Image Classifier")

# ----------------------------------------------------
# RUN APP
# ----------------------------------------------------
if __name__ == "__main__":
    main()
