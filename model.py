import numpy as np
import streamlit as st
import keras
import cv2
import os
from PIL import Image
#
import google.generativeai as genai
api="AIzaSyBCI64AvRnXUGoQTHEKM93f8rMhPdj7MlE"
genai.configure(api_key=api)


class Model:
    def __init__(self):
        self.model = keras.models.load_model(r"J:\VS Code\Python\fish_model.h5")
        self.classes = os.listdir(r"J:\Datasets\Fish dataset\Fish_Dataset")

    def preprocess(self,pil_image):
        image = np.array(pil_image)
        image = cv2.resize(image,(256,256))
        image = image/255.0
        return image

    def predict(self,image):
        image = np.expand_dims(image,0)
        prediction = self.model.predict(image)
        # st.write(prediction)
        class_name = self.classes[np.argmax(prediction)]

        return class_name


class Page:
    def __init__(self):
        st.title("Fish Classifier")

    def image_loader(self):
        image_path = st.file_uploader(label="upload",type=['jpg','jpeg','png','jfif'])
        # st.write(image_path)
        if image_path is not None:
            image = Image.open(image_path)
            st.image(image,caption="image")
            return image

    def classified_name(self,prediction):
        st.write("Class: ",prediction)


def description(name):

    genai_model = genai.GenerativeModel("gemini-1.5-flash")
    response = genai_model.generate_content("explain the fish :"+name)
    st.write(response.text)


page = Page()
image = page.image_loader()

if image is not None:
    model = Model()

    preprocessed_image = model.preprocess(image)
    prediction = model.predict(preprocessed_image)
    page.classified_name(prediction)
    description(prediction)