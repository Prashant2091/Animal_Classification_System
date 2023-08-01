'''import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
import subprocess
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)

@st.experimental_singleton
def load_model():
    model_path = 'animal_model_trained.hdf5'
    if not os.path.isfile(model_path):
        subprocess.run(['curl --output animal_model_trained.hdf5 "https://github.com/Prashant2091/Animal_Classification_System/raw/main/animal_model_trained.hdf5"'], shell=True)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except OSError as e:
        st.error(f"Error loading the model: {str(e)}")
        return None
st.title('Animal Classifier')
file_uploader_key = hash("Upload an image of an animal")
file = st.file_uploader("Upload an image of an animal", type=["jpg", "png"], key=file_uploader_key)
#file = st.file_uploader("Upload an image of an animal", type=["jpg", "png"])
if file is not None:
 label = model.predict(uploaded_file)
 # Process the prediction result as needed
 # ...
else:
 st.text('Waiting for upload....')
def predict_class(image, model):
 image = tf.cast(image, tf.float32)
 image = tf.image.resize(image, [188, 188])
 #image = np.expand_dims(image, axis = 0)
 image = image[np.newaxis,...]
 prediction = model.predict(image)
 return prediction
st.title('Animal Classifier')
file = st.file_uploader("Upload an image of an animal", type=["jpg", "png"])
if file is None:
 st.text('Waiting for upload....')
else:
 slot = st.empty()
 slot.text('Running inference....')
 test_image = Image.open(file)
 st.image(test_image, caption="Input Image", width = 400)
 #pred = predict_class(np.asarray(test_image), model)
 #pred=model.predict(test_image).argmax()
class_names = ['butterfly', 'cow', 'elephant', 'sheep', 'squirrel']
if model is not None:
 pred = model.predict(test_image).argmax()
else:
 st.write("Model not loaded properly. Check the model loading process.")
if model is not None:  # Add this condition to avoid NameError
 result = class_names[np.argmax(pred)]
 output = 'The image is a ' + result
 slot.text('Done')
 st.success(output)
if st.button('predict'):
 st.write("Result...")
if file is not None:
 label = model.predict(file)
 label = label.item()
 res = class_names.get(label)
 st.markdown(res)
else:
 st.text('Waiting for upload....')  '''
import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request

# Function to load the model
@st.cache(allow_output_mutation=True)
@st.experimental_singleton
def load_model():
    model_path = 'animal_model_trained.hdf5'
    if not os.path.isfile(model_path):
        subprocess.run(['curl --output animal_model_trained.hdf5 "https://github.com/Prashant2091/Animal_Classification_System/raw/main/animal_model_trained.hdf5"'], shell=True)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except OSError as e:
        return None

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
        # Preprocess the image
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make a prediction
        pred = model.predict(image).argmax()

        # Show the prediction result
        class_name, probability = get_prediction(image, model)

        # Show the prediction result
        st.write(f"Prediction: {class_name}")
        st.write(f"Probability: {probability:.2f}%")
        #st.write(f"Prediction: {result}")
    else:
        st.error("Error loading the model. Check the model loading process.")
else:
    st.text('Waiting for upload....')





  





