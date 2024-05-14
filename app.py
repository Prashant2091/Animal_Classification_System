import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import subprocess

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'animal_model_trained.hdf5'
    if not os.path.isfile(model_path):
        subprocess.run(['curl --output animal_model_trained.hdf5 "https://github.com/Prashant2091/Animal_Classification_System/raw/main/animal_model_trained.hdf5"'], shell=True)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except OSError as e:
        return None

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to get prediction
def get_prediction(image, model):
    image = preprocess_image(image)
    predictions = model.predict(image)
    class_index = np.argmax(predictions[0])
    class_name = class_names[class_index]
    probability = predictions[0][class_index] * 100
    return class_name, probability

# Streamlit app title
st.title('Animal Classifier')

# Load the model
model = load_model()

# Define class names
class_names = ['butterfly', 'cow', 'elephant', 'sheep', 'squirrel']

# File uploader widget
uploaded_file = st.file_uploader("Upload an image of an animal", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    test_image = Image.open(uploaded_file)
    st.image(test_image, caption="Input Image", width=400)

    if model is not None:
        # Make a prediction
        class_name, probability = get_prediction(test_image, model)

        # Show the prediction result
        st.write(f"Prediction: {class_name}")
        st.write(f"Probability: {probability:.2f}%")
    else:
        st.error("Error loading the model. Please try again later.")
else:
    st.text('Waiting for upload....')
