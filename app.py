import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
@st.cache()
def call_model() :
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image_direct(img) :
    img=np.array(img)
    img=tf.convert_to_tensor(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis , :]
    return img
st.title("Image Stylling For you")
# Code to upload an image
content_file=st.file_uploader("Upload Image 1 with content")
# Code to upload an image
style_file=st.file_uploader("Upload Image 2 with style")

model = call_model()
if content_file and style_file :
    content_image = Image.open(content_file)
    style_image=Image.open(style_file)
    
    content_image=load_image_direct(content_image)
    style_image=load_image_direct(style_image)
    
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    
    output_iamge=st.image(stylized_image[0].numpy())
