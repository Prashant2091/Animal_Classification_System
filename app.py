# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 00:44:37 2022

@author: Prashant
"""
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)

@st.experimental_singleton
def load_model():
    if not os.path.isfile('model.hdf5'):
        urllib.request.urlretrieve('https://github.com/Prashant2091/Animal_Classification_System/blob/main/animal_model_trained.hdf5', 'model.hdf5')
    return tensorflow.keras.models.load_model('model.h5')

def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [188, 188])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Animal Classifier')

file = st.file_uploader("Upload an image of an animal", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['butterfly', 'cow', 'elephant', 'sheep', 'squirrel']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)

