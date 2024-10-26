import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load the pre-trained model
with open('fire_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to preprocess the uploaded image
def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to get the prediction label
def get_prediction_label(prediction):
    return "Fire" if prediction < 0.5 else "Not Fire"

# Streamlit UI
st.title("🔥 Fire Detection Web App")
st.write("Upload an image to check if it contains fire.")

# Image upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png", "bmp"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Display the result
    label = get_prediction_label(prediction)
    st.write(f"Prediction: **{label}**")
