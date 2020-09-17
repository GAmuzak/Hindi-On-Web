import streamlit as st
from PIL import Image
import numpy as np
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
	mapping={0: 'va', 1: '5', 2: 'ae', 3: '3', 4: 'sa', 5: 'bha', 6: 'da', 7: '7', 8: 'cha', 9: '2', 10: 'la', 11: 'ana', 12: 'ka', 13: 'pa', 14: 'aa', 15: 'gha', 16: 'tra', 17: 'ta', 18: 'tha', 19:
	'chha', 20: 'ang', 21: 'yna', 22: 'taa', 23: 'dhaa', 24: '6', 25: 'na', 26: 'ga', 27: 'o', 28: 'gya', 29: '8', 30: 'kha', 31: 'ma', 32: 'thaa', 33: 'ra', 34: 'aha', 35: 'ai', 36: 'a', 37: 'sha',38:
	'oo', 39: 'au', 40: 'pha', 41: 'ee', 42: '4', 43: 'dha', 44: '9', 45: 'u', 46: 'daa', 47: 'i', 48: '1', 49: 'ha', 50: 'rda', 51: 'ba', 52: 'ksha', 53: 'jha', 54: 'katasha', 55: '0', 56: 'ja', 57:
	'ya'}
	
	return mapping[pred]

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

st.title("lmao")

st.set_option('deprecation.showfileUploaderEncoding', False)
im_file = st.file_uploader("Upload an Image", type=["jpg","png", "jpeg"])

check=st.button("check")


if(im_file!=None):
	im = np.array(Image.open(im_file).convert('RGB').resize((224,224),Image.BICUBIC))
	st.title("Input Image")
	st.image(im)
	if(check):
		im=im/255.0
		im_preprocess=np.expand_dims(im, axis=0)
	
		infer_model=inference_model()
		
		infer_model.load_weights("../weights/prototype.h5")
		
		output=infer_model.predict(x=im_preprocess)
		final_class=np.argmax(output, axis=1)[0]   
		st.write(getmap(final_class))
