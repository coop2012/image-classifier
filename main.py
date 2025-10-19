
import numpy as np
import streamlit as st
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as mobilenet_decode
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
from tensorflow.keras.applications.inception_v3 import decode_predictions as inception_decode

from PIL import Image
from PIL import ImageDraw, ImageFont

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)

    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=6)[0]
        return decoded_predictions
    
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="Image Classifier", layout="centered")
    st.title("AI-Powered Image Classifier")
    st.write("Upload an image, and the model will classify it for you.")
    model_choice = st.selectbox(
        "Choose a model",
        ["MobileNetV2", "ResNet50", "InceptionV3"]
    )
    @st.cache_resource
    def load_cached_model(model_name):
        return load_model(model_name)

    model = load_cached_model(model_choice)
    
    model = load_cached_model()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption='Uploaded Image', use_container_width=True
        )

        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analysing..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        label_name = label.replace("_", " ").title()
                        st.write(f"**{label_name}: {score:.2%}**")
                        st.progress(int(score * 100))




if __name__ == "__main__":
    main()