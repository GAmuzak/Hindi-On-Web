import streamlit as st
#import codecs
#import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
def inference_model():
	model=Sequential([
	Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', input_shape=(224,224,3)),
	MaxPool2D(pool_size=(2,2), strides=2),
    
    	Conv2D(filters=64, kernel_size=(4,4), activation='relu', padding='same'),
    	MaxPool2D(pool_size=(2,2), strides=2),
    
    	Conv2D(filters=128, kernel_size=(4,4), activation='relu', padding='same'),
   	MaxPool2D(pool_size=(2,2), strides=2),
    
   	Flatten(),
   	Dense(units=58, activation='softmax')
	])
	return model
	
def getmap(pred):
	mapping={
	0: 'va or व', 1: '5 or ५', 2: 'ae or ए', 3: '3', 4: 'sa', 5: 'bha', 6: 'da', 7: '7', 8: 'cha', 9: '2', 10: 'la', 11: 'ana', 12: 'ka', 
	13: 'pa', 14: 'aa', 15: 'gha', 16: 'tra', 17: 'ta', 18: 'tha', 19: 'chha', 20: 'ang', 21: 'yna', 22: 'taa', 23: 'dhaa', 24: '6',
	25: 'na', 26: 'ga', 27: 'o', 28: 'gya', 29: '8', 30: 'kha', 31: 'ma', 32: 'thaa', 33: 'ra', 34: 'aha', 35: 'ai', 36: 'a', 
	37: 'sha',38: 'oo', 39: 'au', 40: 'pha', 41: 'ee', 42: '4', 43: 'dha', 44: '9', 45: 'u', 46: 'daa', 47: 'i', 48: '1', 49: 'ha', 
	50: 'rda', 51: 'ba', 52: 'ksha', 53: 'jha', 54: 'katasha', 55: '0', 56: 'ja', 57:'ya'
	}
	
	return mapping[pred]

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

st.title("Hindi character recognition")

st.write("Draw a hindi character below and click on Predict button")
st.write("\n")
st.write("To clear the digit, uncheck checkbox, double click on the digit or refresh the page")
st.write("To draw the digit, check the checkbox")
drawing_mode = st.checkbox("Draw?",True)
image_data = st_canvas(15, '#000', '#FFF', height=400,width=400, drawing_mode=drawing_mode, key="canvas")
check=st.button("Predict")
st.set_option('deprecation.showfileUploaderEncoding', False)

if image_data is not None:
	if check:
		cv2.imwrite("test.jpg",image_data)
		im = np.array(Image.open("test.jpg").convert('RGB').resize((224,224),Image.BICUBIC))
		st.title("Input Image")
		st.image(im)
		im=im/255.0
		im_preprocess=np.expand_dims(im, axis=0)
	
		infer_model=inference_model()
		
		infer_model.load_weights("../weights/prototype.h5")
		
		output=infer_model.predict(x=im_preprocess)
		final_class=np.argmax(output, axis=1)[0]   
		st.write("Prediction: The letter is a ",getmap(final_class))
