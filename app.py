import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Dog vs Cat Classifier", layout="centered")

# Title and description
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>Dog vs Cat Image Classifier</h1>
    <p style='text-align: center;'>Upload an image of a dog or cat, and I'll tell you what it is!</p>
    <hr style='border: 1px solid #f0f0f0;'>
""", unsafe_allow_html=True)

# Load the model only once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cat_dog_classifier.keras")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a Dog or Cat Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and resize
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((128, 128))
    
    # Show image with fixed size
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

    # Preprocess and predict
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

    # Show result
    st.markdown(f"""
        <div style='text-align: center; margin-top: 20px;'>
            <h2>Prediction: <span style='color: #4CAF50;'>{label}</span></h2>
        </div>
    """, unsafe_allow_html=True)