# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 00:44:37 2022

@author: Prashant
"""
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
    
  if not os.path.isfile('model.hdf5'):
    subprocess.run(['curl --output model.hdf5 "https://github.com/Prashant2091/Animal_Classification_System/blob/main/animal_model_trained.hdf5"'], shell=True)
    return tf.keras.models.load_model('model.hdf5', compile=False)

def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [188, 188])

	#image = np.expand_dims(image, axis = 0)
	image = image[np.newaxis,...]

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
	#pred = predict_class(np.asarray(test_image), model)
	pred=model.predict(test_image).argmax()
	class_names = ['butterfly', 'cow', 'elephant', 'sheep', 'squirrel']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)'''

import os
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
#import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time
fig = plt.figure()
st.title('Animal Classifier amongst butterfly,cow,elephant,sheep,squirrel')
st.markdown("Prediction :{butterfly,cow,elephant,sheep,squirrel}")
#def main():
#file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
image = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

class_btn = st.button("Classify")
if image is not None:    
        image = Image.open(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
if class_btn:
        if image is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                 plt.imshow(image)
                 plt.axis("off")
                 predictions = predict(image)
                 time.sleep(1)
                 st.success('Classified')
                 st.write(predictions)
		
def load_model():
    
  if not os.path.isfile('model.hdf5'):
    subprocess.run(['curl --output model.hdf5 "https://github.com/Prashant2091/Animal_Classification_System/blob/main/animal_model_trained.hdf5"'], shell=True)
    return tf.keras.models.load_model('model.hdf5', compile=False)		
#model = load_model()		
def predict(image):
    #classifier_model = "animal_model_trained.hdf5"
      
    #model = load_model(classifier_model)
    model = load_model()  
    test_image = image.resize((188,188))
    #test_image = image[np.newaxis,...]
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['butterfly', 'cow', 'elephant', 'sheep', 'squirrel']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 

    return image
#predict(image)
