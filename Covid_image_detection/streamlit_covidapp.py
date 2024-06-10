# -*- coding: utf-8 -*-
import streamlit as st
import requests
from PIL import Image
import numpy as np
import base64
from io import BytesIO

# Title of the Streamlit app
st.title("COVID-19 and Lung Infection Prediction")

# File uploader to upload the lung x-ray images
uploaded_file = st.file_uploader("Upload a lung x-ray image", type=["jpg", "png", "jpeg"])

# If a file is uploaded display the image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        try:
             # Prepare the image file for sending  to the prediction server
            files = {"image": uploaded_file.getvalue()}
            response = requests.post("http://localhost:5000/predict", files=files)
            response.raise_for_status()
            predictions = response.json()
            
            # prediction classes
            classes = ['COVID-19', 'Non-COVID', 'Normal']
            dnn_pred_class = classes[np.argmax(predictions['dnn_prediction'])]
            cnn_pred_class = classes[np.argmax(predictions['cnn_prediction'])]

            # Display the predictions
            st.write(f"DNN Prediction: {dnn_pred_class}")
            st.write(f"CNN Prediction: {cnn_pred_class}")

            if 'dnn_lime' in predictions and 'cnn_lime' in predictions:
                dnn_lime = base64.b64decode(predictions['dnn_lime'])
                cnn_lime = base64.b64decode(predictions['cnn_lime'])

                dnn_lime_image = Image.open(BytesIO(dnn_lime))
                cnn_lime_image = Image.open(BytesIO(cnn_lime))

                st.image(dnn_lime_image, caption='DNN LIME Explanation', use_column_width=True)
                st.image(cnn_lime_image, caption='CNN LIME Explanation', use_column_width=True)
            else:
                st.error("LIME explanation generation failed")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
